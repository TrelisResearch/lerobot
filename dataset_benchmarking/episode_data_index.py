"""
Benchmarks time to find the indices of a given episode index in a dataset.

HF dataset: 75.392 ms
Numpy memmep (write mode): 0.086 ms
Numpy memmep (read mode): 0.034 ms
"""

import time
from contextlib import contextmanager

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@contextmanager
def report_time(description: str):
    start = time.perf_counter()
    yield
    print(f"{description}: {(time.perf_counter() - start) * 1000:.3f} ms")


EP_IX = 100


dataset = LeRobotDataset(repo_id="lerobot/pusht")


with report_time("HF dataset"):
    episode_data_indices = torch.where(torch.stack(dataset.hf_dataset["episode_index"]) == EP_IX)[0]


arr = np.memmap("/tmp/episode_index.dat", dtype=np.dtype("int64"), mode="w+", shape=len(dataset))
arr[:] = torch.stack(dataset.hf_dataset["episode_index"]).numpy()

with report_time("Numpy memmep (write mode)"):
    episode_data_indices_ = np.where(arr == EP_IX)[0]

del arr

arr = np.memmap("/tmp/episode_index.dat", dtype=np.dtype("int64"), mode="r+", shape=len(dataset))

with report_time("Numpy memmep (read mode)"):
    episode_data_indices_ = np.where(arr == EP_IX)[0]

assert np.array_equal(episode_data_indices.numpy(), episode_data_indices_)
