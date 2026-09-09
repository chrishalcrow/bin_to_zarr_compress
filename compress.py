import argparse
import os
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec

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

# It is essential the dtype and num_channels is correct.
# These would ideally be inputs from an external source.
dtype = np.uint16
num_channels = 384 

shard_size = 1_000_000
chunk_size = 10_000
if (shard_size % chunk_size) != 0:
    raise ValueError("`chunk_size` must divide `shard_size`")

# This code would restrict the process to one thread
# numcodecs.blosc.set_nthreads(1)
# zarr.config.set({'threading.max_workers': 1, 'async.concurrency': 1})

zarr_path = str(binary_path).split('.bin')[0] + '.zarr'

total_file_size = os.path.getsize(binary_path)

bytes_per_elec_sample = np.dtype(dtype).itemsize
bytes_per_time_sample = num_channels * bytes_per_elec_sample
time_samples = total_file_size // bytes_per_time_sample

shape = (time_samples, num_channels)
shards = (shard_size, num_channels)  
chunks = (chunk_size, num_channels)  

raw_data = np.memmap(binary_path, dtype=dtype, mode='r', shape=shape)

z = zarr.create_array(
    store=zarr_path,
    shape=shape,
    chunks=chunks,
    shards=shards,
    dtype=dtype,
    compressors=[BloscCodec(cname="zstd", clevel=3)]
)

z[:] = raw_data
