import argparse
import os
from pathlib import Path

import time
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_folder",
    type=str,
    help="Path to data folder containing binary file.",
)
parser.add_argument(
    "output_folder",
    type=str,
    help="Path to output folder.",
)
parser.add_argument(
    "file_index",
    type=str,
    help="which file.",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=30_000,
    help="Time samples per batch. Tune this based on your downstream slice size.",
)

def compress_binary_to_arrow(bindary_path_string, output_folder):


    chunk_size = args.chunk_size

    clock_path = Path(binary_path_string.replace("AmplifierData", "Clock"))
    hubclock_path = Path(binary_path_string.replace("AmplifierData", "HubSyncCounter"))
    binary_path = Path(binary_path_string)
    arrow_path = output_folder / (binary_path.with_suffix(".arrow")).name

    print(f"{arrow_path=}", flush=True)

    # load in the binary files
    if not binary_path.is_file():
        raise FileNotFoundError(f"No binary file at {binary_path}")
    if not clock_path.is_file():
        raise FileNotFoundError(f"No binary file at {clock_path}")
    if not hubclock_path.is_file():
        raise FileNotFoundError(f"No binary file at {hubclock_path}")
    if arrow_path.is_file():
        raise FileExistsError(f"An arrow file already exists at {arrow_path}")

    ephys_dtype = np.uint16
    clock_dtype = np.int64
    num_channels = 384

    total_file_size = os.path.getsize(binary_path)
    bytes_per_sample = np.dtype(ephys_dtype).itemsize
    bytes_per_time_step = num_channels * bytes_per_sample
    time_samples = total_file_size // bytes_per_time_step

    shape = (time_samples, num_channels)
    raw_amplifier_data = np.memmap(binary_path, dtype=ephys_dtype, mode="r", shape=shape)
    raw_hubclock_data = np.memmap(hubclock_path, dtype=clock_dtype, mode="r", shape=(time_samples))
    raw_clock_data = np.memmap(clock_path, dtype=clock_dtype, mode="r", shape=(time_samples))

    # Make the arrow "schema" - this details the names of the columns and the datatypes
    # arrow have pandas style types - ints and floats can be safely cast
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
    # At the end of the script, we the array in RAM into a file all at once. 
    ram_sink = pa.BufferOutputStream()

    # Write data - you do this by writing 1s batches at a time
    write_options = ipc.IpcWriteOptions(
        compression="zstd",  
    )
    #with (pa.OSFile(str(local_temp_arrow), "wb") as sink,
    with ipc.new_file(ram_sink, schema, options=write_options) as writer:
   
        for start_idx in range(0, time_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, time_samples)
   
            amplifier_batch = raw_amplifier_data[start_idx:end_idx]
            hubclock_batch = raw_hubclock_data[start_idx:end_idx]
            clock_batch = raw_clock_data[start_idx:end_idx]
    
            # Reshape into Arrow layout
            arrays = [
                pa.array(np.ascontiguousarray(amplifier_batch[:, col_idx]))
                for col_idx in range(num_channels)
            ]
            arrays += [
                hubclock_batch,
                clock_batch
            ]

            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(batch)
    

    completed_ram_buffer = ram_sink.getvalue()

    # 3. Stream the entire RAM buffer to Ceph scratch in one continuous dump
    with pa.OSFile(str(arrow_path), "wb") as disk_file:
        disk_file.write(completed_ram_buffer)

args = parser.parse_args()

data_folder = args.data_folder
output_folder = Path(args.output_folder)
file_index = args.file_index

binary_path_string = str(Path(data_folder) / f'NeuropixelsV2_ProbeB_AmplifierData_{file_index}.bin')

compress_binary_to_arrow(binary_path_string, output_folder)


