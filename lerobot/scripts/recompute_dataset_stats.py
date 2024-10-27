"""
Script to recompute dataset statistics.
"""

import argparse
from pathlib import Path

from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.online_buffer import LeRobotDatasetV2, LeRobotDatasetV2ImageMode
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.utils.utils import init_logging
from lerobot.scripts.utils import say

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--fps", type=float, default=60.0, help="FPS of the dataset")
    args = parser.parse_args()

    init_logging()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory {dataset_dir} does not exist")

    # Load the dataset
    dataset = LeRobotDatasetV2(
        dataset_dir,
        fps=args.fps,
        image_mode=LeRobotDatasetV2ImageMode.VIDEO,
    )

    say("Computing dataset statistics.")
    stats = compute_stats(dataset)
    
    # Backup old stats file if it exists
    stats_path = dataset.storage_dir / "stats.safetensors"
    if stats_path.exists():
        backup_path = stats_path.with_suffix(".safetensors.backup")
        stats_path.rename(backup_path)
        print(f"Backed up old stats file to {backup_path}")

    # Save new stats
    save_file(flatten_dict(stats), stats_path)
    print(f"Saved new stats to {stats_path}")

    say("Done")