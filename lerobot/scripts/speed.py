from time import perf_counter

import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config



if __name__ == "__main__":
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    policy = make_policy(
        init_hydra_config("lerobot/configs/default.yaml", overrides=["policy=tdmpc_real", "env=koch_real", "+horizon=10"]),
        dataset_stats={
            "observation.images.main": {"mean": torch.tensor(0), "std": torch.tensor(1)},
            "observation.state": {"min": torch.tensor(0), "max": torch.tensor(1)},
            "action": {"min": torch.tensor(0), "max": torch.tensor(1)},
        }
    ).to(device)
    
    observation_batch = {
        "observation.images.main": torch.zeros((1, 1, 3, 66, 88)).to(device),
        "observation.state": torch.zeros((1, 1, 6)).to(device),
    }
    for _ in range(100):
        start = perf_counter()
        with torch.inference_mode():
            policy.run_inference(observation_batch)
        print(perf_counter() - start)
