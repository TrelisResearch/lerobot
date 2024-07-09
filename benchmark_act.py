import torch
from torch.utils.benchmark import Timer
from torch.utils.data.dataloader import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

DEVICE = torch.device("cuda")

delta_timestamps = {"action": [i / 50.0 for i in range(100)]}

dataset = LeRobotDataset(repo_id="lerobot/aloha_sim_transfer_cube_human", delta_timestamps=delta_timestamps)
dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)
batch = next(iter(dataloader))
batch = {k: v.to(device=DEVICE) for k, v in batch.items()}

act_cfg = ACTConfig(use_vae=False)
act = ACTPolicy(act_cfg, dataset_stats=dataset.stats).to(device=DEVICE)

timer = Timer(stmt="act(batch)", globals={"act": act, "batch": batch})

# Real measurement.
result = timer.blocked_autorange(min_run_time=2)

print(result)
