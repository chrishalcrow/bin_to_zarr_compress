import argparse
import os
from pathlib import Path

import numpy as np
import zarr

parser = argparse.ArgumentParser()
parser.add_argument(
    "binary_path",
    type=str,
    help="Path to .bin file.",
)

args = parser.parse_args()
binary_path = Path(args.binary_path)

if not binary_path.is_file():
    raise FileNotFoundError(f'No binary file at {binary_path}')

dtype = np.uint16
num_channels = 384 

zarr_path = str(binary_path).split('.bin')[0] + '.zarr'
zarr_array = zarr.open(zarr_path, mode='r')

zarr_shape = zarr_array.shape
zarr_time_samples, zarr_num_channels = zarr_shape
bytes_per_elec_sample = np.dtype(dtype).itemsize

estimated_raw_data_size = zarr_time_samples*zarr_num_channels*bytes_per_elec_sample
max_raw_data_size = (zarr_time_samples + 1)*zarr_num_channels*bytes_per_elec_sample

binary_file_size = os.path.getsize(binary_path)

if (max_raw_data_size <= binary_file_size) or (binary_file_size < estimated_raw_data_size):
    raise ValueError("File sizes don't make sense.")

raw_data = np.memmap(binary_path, dtype=dtype, mode='r', shape=zarr_shape)

chunk_length = 100_000
checks = []
chunks = np.arange(0,zarr_time_samples,chunk_length)
good_times = True
for chunk in chunks:
    chunk_matches = np.all(zarr_array[chunk:chunk+chunk_length,:] == raw_data[chunk:chunk+chunk_length,:])
    if not chunk_matches:
        raise ValueError('Values of traces do not match in zarr and binary files.')
