import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from lerobot.common.datasets.online_buffer import LeRobotDatasetV2, LeRobotDatasetV2ImageMode
from lerobot.common.datasets.utils import unflatten_dict
from lerobot.common.datasets.lerobot_dataset import DATA_DIR

def inspect_dataset(dataset_path: str):
    # Initialize dataset
    dataset = LeRobotDatasetV2(
        Path(dataset_path),
        image_mode=LeRobotDatasetV2ImageMode.VIDEO,
    )
    
    # Load stats
    stats = unflatten_dict(load_file(dataset.storage_dir / "stats.safetensors"))
    
    print("\n=== Dataset Overview ===")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Number of samples: {dataset.num_samples}")
    print(f"FPS: {dataset.fps}")
    
    print("\n=== Available Keys ===")
    for key in dataset._data.keys():
        if key != "_next_index":
            data = dataset._data[key]
            print(f"\n{key}:")
            print(f"  Shape: {data.shape}")
            if isinstance(data, (torch.Tensor, np.ndarray, np.memmap)):
                if isinstance(data, torch.Tensor):
                    data_np = data.numpy()
                else:
                    data_np = np.array(data)
                
                print(f"  dtype: {data_np.dtype}")
                print(f"  min: {np.min(data_np):.3f}")
                print(f"  max: {np.max(data_np):.3f}")
                print(f"  mean: {np.mean(data_np.astype(float)):.3f}")
                if np.isnan(data_np).any():
                    print("  WARNING: Contains NaN values!")
                if np.isinf(data_np).any():
                    print("  WARNING: Contains Inf values!")
    
    print("\n=== Stats ===")
    for key, value in stats.items():
        print(f"\n{key}:")
        for stat_name, stat_value in value.items():
            print(f"  {stat_name}: {stat_value}")
    
    # Sample a few episodes
    print("\n=== Sample Episode Info ===")
    episode_indices = np.array(dataset._data['episode_index'])
    unique_episodes = np.unique(episode_indices)
    
    print(f"Found {len(unique_episodes)} unique episodes")
    
    for ep_idx in unique_episodes:  # Show all episodes
        ep_mask = episode_indices == ep_idx
        n_frames = np.sum(ep_mask)
        print(f"\nEpisode {ep_idx}:")
        print(f"  Number of frames: {n_frames}")
        
        # Show rewards for this episode
        rewards = np.array(dataset._data['next.reward'][ep_mask])
        success = np.array(dataset._data['next.success'][ep_mask])
        print(f"  Total reward: {np.sum(rewards):.3f}")
        print(f"  Success: {success[-1]}")
        
        # Show some state information
        states = np.array(dataset._data['observation.state'][ep_mask])
        print(f"  Initial state: {states[0]}")
        print(f"  Final state: {states[-1]}")
        
        # Show action information
        actions = np.array(dataset._data['action'][ep_mask])
        print(f"  Action mean: {np.mean(actions, axis=0)}")
        print(f"  Action std: {np.std(actions, axis=0)}")

if __name__ == "__main__":
    inspect_dataset(str(DATA_DIR / "push_cube"))
