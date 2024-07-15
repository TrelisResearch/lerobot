"""
Compares concatenating HF datasets with concatenating them in numpy memmap form. For the memmap, the required
storage space is already pre-allocated. Looks like the HF dataset concatenation is very efficient, I'm guessing
because it relies on some sort of pointer mechanism rather than physically moving data around on the disk.

HF datasets: 4.626 ms
Numpy memmaps: 2789.571 ms
"""

import functools
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@functools.wraps(np.memmap)
def make_memmap_safe(**kwargs) -> np.memmap:
    """Make a numpy memmap with checks on available disk space first."""
    required_space = kwargs["dtype"].itemsize * np.prod(kwargs["shape"])  # bytes
    stats = os.statvfs(Path(kwargs["filename"]).parent)
    available_space = stats.f_bavail * stats.f_frsize  # bytes
    if required_space >= available_space * 0.8:
        raise RuntimeError(f"You're about to take up {required_space} of {available_space} bytes available.")
    return np.memmap(**kwargs)


@contextmanager
def report_time(description: str):
    start = time.perf_counter()
    yield
    print(f"{description}: {(time.perf_counter() - start) * 1000:.3f} ms")


dataset = LeRobotDataset(repo_id="lerobot/pusht_image")

# Make a copy of the dataset in dict[str, memmap] form.
memmap_dataset = {
    k: make_memmap_safe(
        filename=f"/tmp/{k}.dat",
        dtype=dataset[0][k].numpy().dtype,
        mode="w+",
        shape=(len(dataset) * 2, *dataset[0][k].shape),
    )
    for k in dataset.features
}
for k in dataset.features:
    memmap_dataset[k][: len(dataset)] = torch.stack(dataset.hf_dataset[k]).numpy()


with report_time("HF datasets"):
    new_dataset = concatenate_datasets([dataset.hf_dataset, dataset.hf_dataset])


with report_time("Numpy memmaps"):
    for k in memmap_dataset:
        memmap_dataset[k][len(dataset) :] = memmap_dataset[k][: len(dataset)]
