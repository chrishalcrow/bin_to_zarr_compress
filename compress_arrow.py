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
binary_path_string = args.binary_path
chunk_size = args.chunk_size

clock_path = Path(binary_path_string.replace("AmplifierData", "Clock"))
hubclock_path = Path(binary_path_string.replace("AmplifierData", "HubSyncCounter"))
binary_path = Path(binary_path_string)
arrow_path = binary_path.with_suffix(".arrow")

# load in the binary files

if not binary_path.is_file():
    raise FileNotFoundError(f"No binary file at {binary_path}")
if not clock_path.is_file():
    raise FileNotFoundError(f"No binary file at {clock_path}")
if not hubclock_path.is_file():
    raise FileNotFoundError(f"No binary file at {hubclock_path}")
if arrow_path.is_file():
    raise FileExistsError(f"An arrow file already exists at {arrow_path}.")

ephys_dtype = np.uint16
clock_dtype = np.int64
num_channels = 384

total_file_size = os.path.getsize(binary_path)
bytes_per_sample = np.dtype(ephys_dtype).itemsize
bytes_per_time_step = num_channels * bytes_per_sample
time_samples = total_file_size // bytes_per_time_step

shape = (time_samples, num_channels)
raw_amplifier_data = np.memmap(binary_path, dtype=ephys_dtype, mode="r", shape=shape)
raw_hubclock_data = np.memmap(hubclock_path, dtype=ephys_dtype, mode="r", shape=(time_samples))
raw_clock_data = np.memmap(clock_path, dtype=ephys_dtype, mode="r", shape=(time_samples))

# Make the arrow "schema" - this details the names of the columns and the datatypes

# arrow have pandas style types - ints and floats can be safely cast
arrow_type = pa.from_numpy_dtype(ephys_dtype)
fields = [pa.field(f"AmplifierData_{i}", arrow_type) for i in range(num_channels)]
fields += [
    pa.field('HubClock', pa.from_numpy_dtype(clock_dtype)),
    pa.field('Clock', pa.from_numpy_dtype(clock_dtype)),
]
schema = pa.schema(fields)

# Write data - you do this by writing 1s batches at a time

write_options = ipc.IpcWriteOptions(
    compression="zstd",  
)
with (pa.OSFile(str(arrow_path), "wb") as sink,
    ipc.new_file(sink, schema, options=write_options) as writer):
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
