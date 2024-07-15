"""
Benchmarks the time taken to extend a dataset with more data.

HF dataset: 11.588 ms
  create: 9.651 ms
  concat: 1.128 ms
Numpy memmap: 0.202 ms
"""

import time
from contextlib import contextmanager

import numpy as np
import torch
from datasets import Dataset, Features, Value, concatenate_datasets

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@contextmanager
def report_time(description: str):
    start = time.perf_counter()
    yield
    print(f"{description}: {(time.perf_counter() - start) * 1000:.3f} ms")


n_to_add = 10000

dataset = LeRobotDataset(repo_id="lerobot/pusht").hf_dataset.select_columns("index")
next_index = dataset["index"][-1].item() + 1
memmap = np.memmap(
    "/tmp/index.dat",
    dtype=torch.stack(dataset["index"]).numpy().dtype,
    mode="w+",
    shape=(len(dataset) + n_to_add,),
)

data_to_add = torch.arange(next_index, next_index + n_to_add)

with report_time("HF dataset"):
    with report_time("  create"):
        dataset_to_add = Dataset.from_dict(
            {"index": data_to_add}, features=Features({"index": Value(dtype="int64", id=None)})
        )
    with report_time("  concat"):
        new_dataset = concatenate_datasets([dataset_to_add, dataset_to_add])


with report_time("Numpy memmap"):
    memmap[len(dataset) :] = data_to_add.numpy()
