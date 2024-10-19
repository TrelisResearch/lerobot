import argparse
import logging
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.common.rl import calc_reward_cube_push
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config, init_logging
from lerobot.common.vision import GoalSetter

init_logging()


def compute_state_value(tdmpc: TDMPCPolicy, observation_dict):
    observation_batch = deepcopy(observation_dict)
    for name in observation_batch:
        if name.startswith("observation.image"):
            observation_batch[name] = observation_batch[name].type(torch.float32) / 255
            observation_batch[name] = observation_batch[name].permute(2, 0, 1).contiguous()
        observation_batch[name] = observation_batch[name].unsqueeze(0)
        observation_batch[name] = observation_batch[name].cuda()
    observation_batch = tdmpc.normalize_inputs(observation_batch)

    # NOTE: Order of observations matters here.
    encode_keys = []
    if tdmpc.expected_image_key is not None:
        encode_keys.append("observation.images.main")
    if tdmpc._use_env_state:
        encode_keys.append("observation.environment_state")
    encode_keys.append("observation.state")
    z = tdmpc.model.encode({k: observation_batch[k] for k in encode_keys})

    return tdmpc.model.V(z).item()


def run_inference(tdmpc: TDMPCPolicy, observation_dict):
    observation_batch = deepcopy(observation_dict)
    for name in observation_batch:
        if name.startswith("observation.image"):
            observation_batch[name] = observation_batch[name].type(torch.float32) / 255
            observation_batch[name] = observation_batch[name].permute(2, 0, 1).contiguous()
        observation_batch[name] = observation_batch[name].unsqueeze(0).unsqueeze(0)
        observation_batch[name] = observation_batch[name].cuda()
    action_traj = tdmpc.run_inference(observation_batch).cpu()
    return action_traj.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    robot: ManipulatorRobot = make_robot(init_hydra_config("lerobot/configs/robot/koch_.yaml"))
    robot.connect()

    tdmpc = None
    goal_direction = "right"
    goal_setter = GoalSetter.from_mask_file(Path(f"outputs/goal_mask_{goal_direction}.npy"))
    goal_mask = goal_setter.get_goal_mask()
    where_goal = np.where(goal_mask > 0)

    # Test it out
    step = 0
    max_relative_action = torch.zeros(6)
    prior_relative_action = np.zeros(6)
    while True:
        start = time.perf_counter()
        # Fow now let's assume that they all have the same timestamp.
        obs_dict, action_dict = robot.teleop_step(record_data=True)
        if tdmpc is not None:
            V = compute_state_value(tdmpc, obs_dict)
            action_traj = (
                np.cumsum(run_inference(tdmpc, obs_dict), axis=0) + obs_dict["observation.state"].numpy()
            )
        else:
            V = 0
        action = action_dict["action"].numpy()
        relative_action = (action_dict["action"] - obs_dict["observation.state"]).numpy()
        img = obs_dict["observation.images.main"].numpy()
        annotated_img = img.copy()
        reward, _, _, info = calc_reward_cube_push(
            img=obs_dict["observation.images.main"].numpy(),
            goal_mask=goal_mask,
            current_joint_pos=obs_dict["observation.state"].numpy(),
            action=relative_action,
            prior_action=prior_relative_action,
        )
        annotated_img = info["annotated_img"]
        annotated_img[where_goal] = (
            annotated_img[where_goal] - (annotated_img[where_goal] - np.array([255, 255, 255])) // 2
        )
        annotated_img = cv2.resize(annotated_img, (640, 480))
        cv2.putText(
            annotated_img,
            org=(10, 25),
            color=(255, 255, 255),
            text=f"{reward=:.3f}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=1,
        )
        cv2.putText(
            annotated_img,
            org=(10, 50),
            color=(255, 255, 255),
            text=f"{V=:.3f}",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=1,
        )
        cv2.imshow("Test", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        elif k == ord("s"):
            goal_direction = "left" if goal_direction == "right" else "right"
            goal_setter = GoalSetter.from_mask_file(Path(f"outputs/goal_mask_{goal_direction}.npy"))
            goal_mask = goal_setter.get_goal_mask()
            where_goal = np.where(goal_mask > 0)
        elapsed = time.perf_counter() - start
        if elapsed > 1 / args.fps:
            logging.warning(f"Loop iteration went overtime: {elapsed=}.")
        else:
            busy_wait((1 / args.fps) - elapsed)

        prior_relative_action = relative_action.copy()
        step += 1

    cv2.destroyAllWindows()
