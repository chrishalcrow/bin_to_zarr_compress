import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import ipc


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        help="Path to data folder containing binary file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to folder to store your output",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        help="which probe, e.g. 'ProbeA'",
    )
    parser.add_argument(
        "--file_index",
        type=str,
        help="The index of the file you'd like to compute e.g. 13 for the file '..._13.bin'",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=30_000,
        help="Time samples per batch. Default matches the bonsai-onix settings.",
    )
    
    return parser

def pad_short_clock_files(raw_clock_data, clock_batch_samples, ephys_batch_samples, clock_dtype):
    """The arrow final file, which combines ephys, clock and hubclock data, requires that all
    arrays are the same length. If clock files are shorter than ephys fiels, pad them with nans."""

    arrow_dtype = pa.from_numpy_dtype(clock_dtype)
    hubclock_samples = pa.Array.from_buffers(
        arrow_dtype,
        clock_batch_samples,
        [None, pa.py_buffer(raw_clock_data)]
    )
    null_pad = pa.nulls(ephys_batch_samples - clock_batch_samples, type=arrow_dtype)
    padded_hubclock = pa.concat_arrays([hubclock_samples, null_pad])
    return padded_hubclock

def compress_binary_to_arrow(binary_path_string: str, output_folder: str, chunk_size: int):
    """Takes the binary file located at `binary_path_string` and saves a corresponding 
    arrow file in `output_folder`.

    If the binary file is of the form:
        NeuropixelsV2_ProbeB_AmplifierData_65.bin

    The function will also take the files:
        NeuropixelsV2_ProbeB_HubClock_65.bin
        NeuropixelsV2_ProbeB_Clock_65.bin

    And combine all data into a single arrow file at:
        output_folder/NeuropixelsV2_ProbeB_AmplifierData_65.arrow"""

    clock_path = Path(binary_path_string.replace("AmplifierData", "Clock"))
    hubclock_path = Path(binary_path_string.replace("AmplifierData", "HubSyncCounter"))
    binary_path = Path(binary_path_string)
    arrow_path = Path(output_folder) / (binary_path.with_suffix(".arrow")).name

    if not binary_path.is_file():
        raise FileNotFoundError(f"No binary file at {binary_path}")
    if not clock_path.is_file():
        raise FileNotFoundError(f"No binary file at {clock_path}")
    if not hubclock_path.is_file():
        raise FileNotFoundError(f"No binary file at {hubclock_path}")
    if arrow_path.is_file():
        raise FileExistsError(f"An arrow file already exists at {arrow_path}")

    ephys_dtype = np.uint16
    clock_dtype = np.uint64
    num_channels = 384

    # load ephys file - code from spikeinterface
    total_file_size = os.path.getsize(binary_path)
    bytes_per_sample = np.dtype(ephys_dtype).itemsize
    bytes_per_time_step = num_channels * bytes_per_sample
    ephys_time_samples = total_file_size // bytes_per_time_step
    ephys_shape = (ephys_time_samples, num_channels)
    raw_amplifier_data = np.memmap(binary_path, dtype=ephys_dtype, mode="r", shape=ephys_shape)

    # load clock file
    raw_hubclock_data = np.memmap(hubclock_path, dtype=clock_dtype, mode="r")
    raw_clock_data = np.memmap(clock_path, dtype=clock_dtype, mode="r")

    # Make the arrow "schema" - this details the names of the columns and datatypes.
    # Arrow has pandas-like types - ints and floats can be safely cast
    arrow_type = pa.from_numpy_dtype(ephys_dtype)
    fields = [pa.field(f"AmplifierData_{i}", arrow_type) for i in range(num_channels)]
    fields += [
        pa.field('HubClock', pa.from_numpy_dtype(clock_dtype)),
        pa.field('Clock', pa.from_numpy_dtype(clock_dtype)),
    ]
    schema = pa.schema(fields)

    # Usually, you'd "stream" your data directly into an arrow file bit by bit. On the HPC
    # the arrow file might be on a different physical data drive than the 
    # original data, and this makes streaming very slow. Instead, we stream
    # directly into RAM. This is ok because our chunks max out at ~10GB.
    # At the end of the script, we dump the array from RAM into a file all at once. 
    ram_sink = pa.BufferOutputStream()

    # Write data - you do this by writing 1s batches at a time
    write_options = ipc.IpcWriteOptions(
        compression="zstd",  
    )
    with ipc.new_file(ram_sink, schema, options=write_options) as writer:
   
        for start_idx in range(0, ephys_time_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, ephys_time_samples)

            # slicing automatically deals with the case where the ephys file
            # is shorter than the clock file. In this case we cut off the hanging
            # times. This is fine since there's no corresponding data to match
            # the times so we don't want them.
            amplifier_batch = raw_amplifier_data[start_idx:end_idx]
            hubclock_batch = raw_hubclock_data[start_idx:end_idx]
            clock_batch = raw_clock_data[start_idx:end_idx]

            num_ephys_batch_samples = len(amplifier_batch)
            num_hubclock_batch_samples = len(hubclock_batch)
            num_clock_batch_samples = len(clock_batch)

            # Reshape into Arrow layout
            arrays = [
                pa.array(np.ascontiguousarray(amplifier_batch[:, col_idx]))
                for col_idx in range(num_channels)
            ]
            if num_hubclock_batch_samples == num_ephys_batch_samples:
                arrays += [hubclock_batch]
            else:
                padded_hubclock = pad_short_clock_files(
                    hubclock_batch,
                    num_hubclock_batch_samples,
                    num_ephys_batch_samples,
                    clock_dtype,
                )
                arrays += [padded_hubclock]

            if num_clock_batch_samples == num_ephys_batch_samples:
                arrays += [clock_batch]
            else:
                padded_clock = pad_short_clock_files(
                    clock_batch,
                    num_clock_batch_samples,
                    num_ephys_batch_samples,
                    clock_dtype,
                )
                arrays += [padded_clock]

            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(batch)

    completed_ram_buffer = ram_sink.getvalue()

    # Stream entire RAM buffer to file in one write
    with pa.OSFile(str(arrow_path), "wb") as disk_file:
        disk_file.write(completed_ram_buffer)

if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = args.output_folder
    probe_name = args.probe_name
    file_index = args.file_index
    chunk_size = args.chunk_size

    binary_path_string = str(Path(data_folder) / f'NeuropixelsV2_{probe_name}_AmplifierData_{file_index}.bin')

    compress_binary_to_arrow(binary_path_string, output_folder, chunk_size)
