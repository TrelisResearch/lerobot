#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os.path as osp
import random
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Generator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig


def get_safe_torch_device(cfg_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    match cfg_device:
        case "cuda":
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        case "mps":
            assert torch.backends.mps.is_available()
            device = torch.device("mps")
        case "cpu":
            device = torch.device("cpu")
            if log:
                logging.warning("Using CPU, this will be slow.")
        case _:
            device = torch.device(cfg_device)
            if log:
                logging.warning(f"Using custom {cfg_device} device.")

    return device


def get_global_random_state() -> dict[str, Any]:
    """Get the random state for `random`, `numpy`, and `torch`."""
    random_state_dict = {
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state_dict["torch_cuda_random_state"] = torch.cuda.random.get_rng_state()
    return random_state_dict


def set_global_random_state(random_state_dict: dict[str, Any]):
    """Set the random state for `random`, `numpy`, and `torch`.

    Args:
        random_state_dict: A dictionary of the form returned by `get_global_random_state`.
    """
    random.setstate(random_state_dict["random_state"])
    np.random.set_state(random_state_dict["numpy_random_state"])
    torch.random.set_rng_state(random_state_dict["torch_random_state"])
    if torch.cuda.is_available():
        torch.cuda.random.set_rng_state(random_state_dict["torch_cuda_random_state"])


def set_global_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def seeded_context(seed: int) -> Generator[None, None, None]:
    """Set the seed when entering a context, and restore the prior random state at exit.

    Example usage:

    ```
    a = random.random()  # produces some random number
    with seeded_context(1337):
        b = random.random()  # produces some other random number
    c = random.random()  # produces yet another random number, but the same it would have if we never made `b`
    ```
    """
    random_state_dict = get_global_random_state()
    set_global_seed(seed)
    yield None
    set_global_random_state(random_state_dict)


def init_logging():
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def format_big_number(num, precision=0):
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    for suffix in suffixes:
        if abs(num) < divisor:
            return f"{num:.{precision}f}{suffix}"
        num /= divisor

    return num


def _relative_path_between(path1: Path, path2: Path) -> Path:
    """Returns path1 relative to path2."""
    path1 = path1.absolute()
    path2 = path2.absolute()
    try:
        return path1.relative_to(path2)
    except ValueError:  # most likely because path1 is not a subpath of path2
        common_parts = Path(osp.commonpath([path1, path2])).parts
        return Path(
            "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
        )


def init_hydra_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Initialize a Hydra config given only the path to the relevant config file.

    For config resolution, it is assumed that the config file's parent is the Hydra config dir.
    """
    # TODO(alexander-soare): Resolve configs without Hydra initialization.
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Hydra needs a path relative to this file.
    hydra.initialize(
        str(_relative_path_between(Path(config_path).absolute().parent, Path(__file__).absolute().parent)),
        version_base="1.2",
    )
    cfg = hydra.compose(Path(config_path).stem, overrides)
    return cfg


def print_cuda_memory_usage():
    """Use this function to locate and debug memory leak."""
    import gc

    gc.collect()
    # Also clear the cache if you want to fully release the memory
    torch.cuda.empty_cache()
    print("Current GPU Memory Allocated: {:.2f} MB".format(torch.cuda.memory_allocated(0) / 1024**2))
    print("Maximum GPU Memory Allocated: {:.2f} MB".format(torch.cuda.max_memory_allocated(0) / 1024**2))
    print("Current GPU Memory Reserved: {:.2f} MB".format(torch.cuda.memory_reserved(0) / 1024**2))
    print("Maximum GPU Memory Reserved: {:.2f} MB".format(torch.cuda.max_memory_reserved(0) / 1024**2))


class Timer:
    """Timer with a context manager that allows you to accrue timing statistics.

    Use like:

    ```
    with Timer.time("description"):
        foo()

    print(Timer.render_timing_statistics)
    ```

    A convenient feature is that timing statistics accrue for a given description. For example, this will
    end up computing statistics for 100 calls to `foo`.

    ```
    for _ in range(100):
        with Timer.time("description"):
            foo()
    ```

    Note that the timing statistics are stored in a class attribute meaning that this class is NOT thread
    safe.
    """

    # Store timing results as:
    # {
    #     "description1": {HEADER: value, ANOTHER_HEADER: another value, ...},
    #     "description2": {HEADER: value, ANOTHER_HEADER: another value, ...},
    #     ....
    # }
    timing_statistics: dict[str, dict[str, int | float]] = {}
    _timing_session_id_iterator = count()

    AVG_TIME_HEADER = "Avg. time (s)"
    PROCESS_HEADER = "Routine"
    TOTAL_TIME_HEADER = "Total time (s)"
    COUNT_HEADER = "Count"
    LATEST_TIME_HEADER = "Latest time (s)"

    @classmethod
    @contextmanager
    def time(cls, description: str):
        """Time everything within the context and store it under a given description.

        If there are already timing statics for this description, update them.
        """
        start_time = time.perf_counter()
        yield
        time_taken = time.perf_counter() - start_time
        # Update or initialize statistics for this description.
        if description in cls.timing_statistics:
            cls.timing_statistics[description][cls.COUNT_HEADER] += 1
            cls.timing_statistics[description][cls.TOTAL_TIME_HEADER] += time_taken
            cls.timing_statistics[description][cls.AVG_TIME_HEADER] = (
                cls.timing_statistics[description][cls.TOTAL_TIME_HEADER]
                / cls.timing_statistics[description][cls.COUNT_HEADER]
            )
            cls.timing_statistics[description][cls.LATEST_TIME_HEADER] = time_taken
        else:
            cls.timing_statistics[description] = {
                cls.AVG_TIME_HEADER: time_taken,
                cls.COUNT_HEADER: 1,
                cls.TOTAL_TIME_HEADER: time_taken,
                cls.LATEST_TIME_HEADER: time_taken,
            }

    @classmethod
    def reset(cls, description: str | list[str] | None = None):
        """Clear all timing statistics or just some timing statistics according to `description`."""
        if isinstance(description, str):
            del cls.timing_statistics[description]
        elif isinstance(description, list):
            for d in description:
                del cls.timing_statistics[d]
        else:
            cls.timing_statistics = {}
