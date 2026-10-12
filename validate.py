import argparse
import os
from pathlib import Path
import numpy as np
from pyarrow import memory_map
from pyarrow.ipc import RecordBatchFileReader

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
        help="Path to output folder.",
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        help="which probe.",
    )
    parser.add_argument(
        "--file_index",
        type=str,
        help="which file.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=30_000,
        help="Time samples per batch. Tune this based on your downstream slice size.",
    )
    
    return parser

def validate_binary_vs_arrow(binary_path_string, output_folder):

    clock_path = Path(binary_path_string.replace("AmplifierData", "Clock"))
    hubclock_path = Path(binary_path_string.replace("AmplifierData", "HubSyncCounter"))
    binary_path = Path(binary_path_string)
    arrow_path = Path(output_folder / (binary_path.with_suffix(".arrow")).name)

    # load in the binary files
    if not binary_path.is_file():
        raise FileNotFoundError(f"No binary file at {binary_path}")
    if not clock_path.is_file():
        raise FileNotFoundError(f"No binary file at {clock_path}")
    if not hubclock_path.is_file():
        raise FileNotFoundError(f"No binary file at {hubclock_path}")
    if not arrow_path.is_file():
        raise FileExistsError(f"No arrow file at {arrow_path}")

    # Infor about the recording
    ephys_dtype = np.uint16
    clock_dtype = np.uint64
    num_channels = 384
    sampling_frequency = 30_000

    # First, validate the arrow and binary files directly

    # load the binary file
    total_file_size = os.path.getsize(binary_path)
    bytes_per_sample = np.dtype(ephys_dtype).itemsize
    bytes_per_time_step = num_channels * bytes_per_sample
    ephys_time_samples = total_file_size // bytes_per_time_step

    shape = (ephys_time_samples, num_channels)

    raw_amplifier_data = np.memmap(binary_path, dtype=ephys_dtype, mode="r", shape=shape)
    raw_hubclock_data = np.memmap(hubclock_path, dtype=clock_dtype, mode="r")
    raw_clock_data = np.memmap(clock_path, dtype=clock_dtype, mode="r")

    # load the arrow file
    source = memory_map(str(arrow_path), "r")
    reader = RecordBatchFileReader(source)
    num_batches = reader.num_record_batches

    initial_batch_index = 0
    for batch_index in range(num_batches):

        batch = reader.get_batch(batch_index)
        batch_size = batch.num_rows

        arrow_samples = batch.to_pandas()

        hubclock_from_arrow = arrow_samples['HubClock'].values
        clock_from_arrow = arrow_samples['Clock'].values
        ephys_from_arraow = arrow_samples.values[:,:-2]

        ephys_matches = np.all(ephys_from_arraow == raw_amplifier_data[initial_batch_index:initial_batch_index+batch_size,:])

        sliced_binary_clock_data = raw_clock_data[initial_batch_index:initial_batch_index+batch_size]
        if len(clock_from_arrow) == len(sliced_binary_clock_data):
            clock_matces = np.all(clock_from_arrow == sliced_binary_clock_data)
        else:
            clock_from_arrow_ints = np.astype(clock_from_arrow[0:len(sliced_binary_clock_data)], np.uint64)
            clock_matces = np.all(clock_from_arrow_ints == sliced_binary_clock_data)
            padded_data = clock_from_arrow[len(sliced_binary_clock_data):]
            padded_data_are_nans = np.all(np.isnan(padded_data))
            if not padded_data_are_nans:
                raise ValueError('Some padding clock data that we expect to be nans are not nans')

        sliced_binary_hubclock_data = raw_hubclock_data[initial_batch_index:initial_batch_index+batch_size]
        if len(hubclock_from_arrow) == len(sliced_binary_hubclock_data):
            hubclock_matches = np.all(hubclock_from_arrow == sliced_binary_hubclock_data)
        else:
            hubclock_matches = np.all(hubclock_from_arrow[:len(sliced_binary_hubclock_data)] == sliced_binary_hubclock_data)
            padded_data = hubclock_from_arrow[len(sliced_binary_hubclock_data):]
            padded_data_are_nans = np.all(np.isnan(padded_data))
            if not padded_data_are_nans:
                raise ValueError('Some padding clock data that we expect to be nans are not nans')

        if not ephys_matches:
            raise ValueError('Ephys does not match')
        if not clock_matces:
            raise ValueError('Clock does not match')
        if not hubclock_matches:
            raise ValueError('HubClock does not match')

        initial_batch_index += batch_size

    # Now we'll validate using the SpikeInterface readers
    # This gives another validation test, and tests the readers

    from aeon.dj_pipeline.utils.spike_sorting_utils import read_bonsai_onix_arrow
    from spikeinterface.extractors import read_binary

    rec_arrow = read_bonsai_onix_arrow(str(arrow_path), sampling_frequency=sampling_frequency)
    rec_binary = read_binary(binary_path, sampling_frequency=sampling_frequency, dtype=ephys_dtype, num_channels=num_channels)
    num_samples = rec_binary.get_num_samples()


    initial_sample_index = 0
    final_sample_index = 0
    while final_sample_index != num_samples:

        # check one second at a time
        final_sample_index = min(initial_sample_index + sampling_frequency, num_samples)

        arrow_data = rec_arrow.get_traces(start_frame=initial_sample_index, end_frame=final_sample_index)
        binary_data = rec_binary.get_traces(start_frame=initial_sample_index, end_frame=final_sample_index)

        data_match = np.all(arrow_data == binary_data)

        if not data_match:
            raise ValueError('ephys does not match for arrow and binary files')

        initial_sample_index = final_sample_index


if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    data_folder = args.data_folder
    output_folder = Path(args.output_folder)
    probe_name = args.probe_name
    file_index = args.file_index

    binary_path_string = str(Path(data_folder) / f'NeuropixelsV2_{probe_name}_AmplifierData_{file_index}.bin')
    arrow_path_string = Path(output_folder) / f'NeuropixelsV2_{probe_name}_AmplifierData_{file_index}.arrow'

    validate_binary_vs_arrow(binary_path_string, output_folder)




