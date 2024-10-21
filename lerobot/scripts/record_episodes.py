"""
For LeRobot Hackathon.

LeRobot already has a script for recording datasets but I made this one for two reasons:
1. Most importantly, I want to gather the data the same way it is done during the online rollout part of the
   training loop (ie, via the `rollout` function).
2. It uses LeRobotDatasetV2.

Tips:
- Aim to have the gripper pointing down most of the time. This removes a potential source of variability.
- Always start the arm near the edge of the workspace, on the side opposite to the goal region. So, if the
  goal is to push the cube left, start the arm just inside the workspace, all the way to the right.
- When you start recording, have the cube on the right, as the first episode will have a goal of pushing it
  left.
- Record about half of the episodes with the cube starting right near the gripper (hint: you don't really even
  need to reset the cube manually, as the previous episode will have put the cube where you want it for the
  current episode). Record the other half of the episodes with the cube starting in random places between the
  two goal regions.
- Use the visualization to verify that the reward is being calculated correctly.
- If you have the patience, gather 100 episodes. This is what I did a few weeks ago (I am doing a run now and
  it looks like it's going to work with 40 but maybe with a slightly slower start).
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.online_buffer import LeRobotDatasetV2, LeRobotDatasetV2ImageMode
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.scripts.eval_real import rollout, say


class TeleopPolicy(torch.nn.Module):
    """
    HACK: Wrap leader arm in a policy module so that we can use it in the `rollout` function.
    """

    def __init__(self, robot: ManipulatorRobot):
        super().__init__()
        self.robot = robot
        self._dummy_param = torch.nn.Parameter(torch.tensor(0), requires_grad=False)

    @property
    def input_keys(self) -> list[str]:
        return ["observation.state"]

    @property
    def n_obs_steps(self) -> int:
        return 1

    def run_inference(self, observation_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns relative joint angles"""
        # unsqueeze twice for batch and temporal dimension
        return (
            torch.from_numpy(self.robot.leader_arms["main"].read("Present_Position"))
            .unsqueeze(0)
            .unsqueeze(0)
        ) - observation_batch["observation.state"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--overwrite-dataset", action="store_true")
    args = parser.parse_args()

    init_logging()

    dataset_dir = Path(args.dataset_dir)
    robot_cfg = init_hydra_config("lerobot/configs/robot/koch_.yaml")
    robot: ManipulatorRobot = make_robot(init_hydra_config("lerobot/configs/robot/koch_.yaml"))
    robot.connect()
    if dataset_dir.exists():
        if args.overwrite_dataset:
            shutil.rmtree(dataset_dir)
        else:
            msg = "Found existing dataset directory. Loading it up."
            print(msg)
            say(msg, blocking=True)
    dataset = LeRobotDatasetV2(
        dataset_dir, fps=robot_cfg.cameras.main.fps, image_mode=LeRobotDatasetV2ImageMode.VIDEO
    )
    episode_ix = 0 if len(dataset) == 0 else dataset.get_unique_episode_indices().max() + 1
    policy_cfg = init_hydra_config("lerobot/configs/policy/tdmpc_real.yaml")
    while True:
        if episode_ix >= args.num_episodes:
            break
        goal = "left" if episode_ix % 2 == 0 else "right"
        msg = f"{episode_ix}."
        say(msg, blocking=True)
        print(msg)
        episode_data = rollout(
            robot,
            TeleopPolicy(robot),
            robot_cfg.cameras.main.fps,
            warmup_s=0,
            n_pad_episode_data=policy_cfg.policy.horizon - 1,
            manual_reset=True,
            visualize_img=True,
            goal=goal,
        )
        say("Episode finished. Press the return key to proceed.")
        while True:
            res = input(
                "Press return key to proceed, or 'n' then the return key to re-record the last episode, or "
                "'q' then the return key to stop recording.\n"
            )
            if res.lower() not in ["", "n", "q"]:
                print("Invalid input. Try again.")
            else:
                break
        if res.lower() in ["", "q"]:
            episode_ix += 1
            dataset.add_episodes(episode_data)
            if res.lower() == "q":
                break
        elif res.lower() == "n":
            continue

    robot.disconnect()

    say("Dataset recording finished. Computing dataset statistics.")
    stats = compute_stats(dataset)
    stats_path = dataset.storage_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    say("Done")
