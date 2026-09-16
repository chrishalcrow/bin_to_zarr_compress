import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

parser = argparse.ArgumentParser()
parser.add_argument("binary_path", type=str, help="Path to .bin file.")
parser.add_argument(
    "--chunk_size",
    type=int,
    default=30_000,
    help="Time samples per batch. Tune this based on your downstream slice size.",
)

args = parser.parse_args()
binary_path = Path(args.binary_path)

if not binary_path.is_file():
    raise FileNotFoundError(f"No binary file at {binary_path}")

dtype = np.uint16
num_channels = 384
chunk_size = args.chunk_size

arrow_path = binary_path.with_suffix(".arrow")

total_file_size = os.path.getsize(binary_path)
bytes_per_sample = np.dtype(dtype).itemsize
bytes_per_time_step = num_channels * bytes_per_sample
time_samples = total_file_size // bytes_per_time_step

shape = (time_samples, num_channels)
raw_data = np.memmap(binary_path, dtype=dtype, mode="r", shape=shape)

arrow_type = pa.from_numpy_dtype(dtype)
fields = [pa.field(f"AmplitudeData_{i}", arrow_type) for i in range(num_channels)]
schema = pa.schema(fields)

# 3. Stream chunks directly to the Arrow IPC file
with (pa.OSFile(str(arrow_path), "wb") as sink,
    ipc.new_file(sink, schema) as writer):
        for start_idx in range(0, num_channels, chunk_size):
            end_idx = min(start_idx + chunk_size, num_channels)
            
            # Zero-copy row slice from disk
            chunk = raw_data[start_idx:end_idx]

            # Reshape chunk into Arrow layout
            arrays = [
                pa.array(np.ascontiguousarray(chunk[:, col_idx]))
                for col_idx in range(num_channels)
            ]
            
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_batch(batch)
