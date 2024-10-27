"""
For the LeRobot Hackathon.

Adaptation of eval.py (from the main branch). Main changes:
- Uses a real robot for rollout rather than a simulation.
- Adds teleoperation capability. This lets us record a dataset with the same code that is used to record online
  rollouts.
- Parts of the code are specific to the RL task we will be working with.
"""

import argparse
import json
import logging
import random
import threading
import time
from collections import defaultdict
from datetime import datetime as dt
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import torch
from termcolor import colored
from torch import nn
from tqdm import trange

from lerobot.common.datasets.online_buffer import LeRobotDatasetV2
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.rl import (
    calc_reward_cube_push,
    reset_for_cube_push,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.teleoperators.ps5_controller import PS5Controller
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.digital_twin import DigitalTwin
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.common.vision import GoalSetter, segment_hsv
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.utils import say


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


def rollout(
    robot: ManipulatorRobot,
    policy: Policy | None,
    fps: float,
    n_action_buffer: int = 0,
    warmup_s: float = 5.0,
    use_relative_actions: bool = True,
    max_steps: int | None = None,
    visualize_img: bool = False,
    visualize_3d: bool = False,
    enable_intevention: bool = False,
    n_pad_episode_data: int = 0,
    manual_reset: bool = False,
    goal: str | None = None,
) -> dict:
    """
    Run the policy on the robot for one episode and record the episode along the way so it can be added to
    the online data buffer of the training loop.
    """

    # TASK SPECIFIC. Get the left and right goal masks, and the "in-bounds" mask.
    goal_setter_left = GoalSetter.from_mask_file("outputs/goal_mask_left.npy")
    goal_setter_right = GoalSetter.from_mask_file("outputs/goal_mask_right.npy")
    # This is called "center" but I really mean "in-bounds".
    in_bounds_mask = GoalSetter.from_mask_file("outputs/goal_mask_center.npy").get_goal_mask()
    goal_mask_left = goal_setter_left.get_goal_mask()
    goal_mask_right = goal_setter_right.get_goal_mask()

    # TASK SPECIFIC. Use the reward function to determine if the cube is already sitting on one of the goal
    # regions, then set the goal to be the other region. Note though, that if `goal` is provided in the kwargs
    # we go with that rather than this automated approach.
    observation: dict[str, torch.Tensor] = robot.capture_observation()
    reward_left, *_ = calc_reward_cube_push(
        img=observation["observation.images.main"].numpy(),
        goal_mask=goal_mask_left,
        current_joint_pos=observation["observation.state"].numpy(),
        oob_reward=0,
        occlusion_reward=0,
    )
    reward_right, *_ = calc_reward_cube_push(
        img=observation["observation.images.main"].numpy(),
        goal_mask=goal_mask_right,
        current_joint_pos=observation["observation.state"].numpy(),
        oob_reward=0,
        occlusion_reward=0,
    )
    if goal is None:
        if reward_left > -0.5:  # The block is on the left
            goal_mask = goal_mask_right  # Goal is on the right
            goal = "right"
            start_pos = "left"
        elif reward_right > -0.5:  # The block is on the right
            goal_mask = goal_mask_left  # goal is on the left
            goal = "left"
            start_pos = "right"
        elif random.random() > 0.5:
            goal_mask = goal_mask_right  # goal is on the right
            goal = "right"
            start_pos = "left"  # if random.random() < 0.9 else "right"  # make it more likely to start left
        else:
            goal_mask = goal_mask_left  # goal is on the left
            goal = "left"
            start_pos = "right"  # if random.random() < 0.9 else "left"  # make it more likely to start right.
    else:
        if goal == "left":
            goal_mask = goal_mask_left
            start_pos = "right"
        elif goal == "right":
            goal_mask = goal_mask_right
            start_pos = "left"
        else:
            raise ValueError

    # TASK SPECIFIC: Reset the robot position, either manually with teleop (for dataset recording) or
    # autonomously (for online rollouts).
    if manual_reset:
        msg = f"Going {goal}. Reset the environment and robot. Press return in the terminal when ready."
        say(msg)
        keyboard_thread = threading.Thread(target=lambda: input(msg), daemon=True)
        keyboard_thread.start()
        while True:
            start = time.perf_counter()
            robot.teleop_step()
            if not keyboard_thread.is_alive():
                break
            time.sleep(max(0, 1 / fps - (time.perf_counter() - start)))
        say("Go!")
    else:
        say(f"Go {goal}")
        reset_for_cube_push(robot, right=start_pos == "right")

    # TASK SPECIFIC: Check if the cube is out of bounds, and ask for help until it's placed back in bounds.
    while True:
        observation: dict[str, torch.Tensor] = robot.capture_observation()
        cube_mask, _ = segment_hsv(observation["observation.images.main"].numpy())
        if np.count_nonzero(cube_mask & in_bounds_mask) == np.count_nonzero(cube_mask):
            break
        say("Cube is out of bounds! Help.")
        time.sleep(5)

    # Preparing a bunch of stuff for the rollout loop.
    where_goal = torch.where(torch.from_numpy(goal_mask) > 0)

    if visualize_3d:
        digital_twin = DigitalTwin()

    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    policy_rollout_wrapper = PolicyRolloutWrapper(policy, fps=fps, n_action_buffer=n_action_buffer)
    policy_rollout_wrapper.reset()

    # This is the data structure the episode will be "recorded" into.
    episode_data = defaultdict(list)

    if enable_intevention:
        ps5_controller = PS5Controller()

    period = 1 / fps
    to_visualize = {}
    reward = 0
    success = False
    first_follower_pos = None  # Will be held during the warmup
    prior_absolute_action = None
    surrender_control = False
    step = 0
    start_time = time.perf_counter()
    is_warmup = True

    def to_relative_time(t):
        return t - start_time

    # MAIN ROLLOUT LOOP.
    while True:
        over_time = False
        start_step_time = to_relative_time(time.perf_counter())
        is_warmup = start_step_time <= warmup_s
        observation: dict[str, torch.Tensor] = robot.capture_observation()
        print(f"Observation keys: {observation.keys()}")  # Print available keys instead
        annotated_img = None

        # Update the episode data for this frame with indices, the observation, and the reward.
        if not is_warmup:
            episode_data[LeRobotDatasetV2.INDEX_KEY].append(step)
            episode_data[LeRobotDatasetV2.EPISODE_INDEX_KEY].append(0)
            episode_data[LeRobotDatasetV2.TIMESTAMP_KEY].append(start_step_time)
            episode_data[LeRobotDatasetV2.FRAME_INDEX_KEY].append(step)

            for k in observation:
                if k.startswith("observation.image"):
                    img = observation[k].numpy().copy()
                    # HACK use masking to indicate to policy which side needs the cube:
                    img[where_goal] = img[where_goal] // 2 + np.array([127, 127, 127])
                    episode_data[k].append(img)
                else:
                    episode_data[k].append(observation[k].numpy())

            # All data keys that start with "next." get appended to the episode data starting from step 1
            # That way they are offset by 1 relative to keys not starting with "next.".
            if step > 0:
                if len(episode_data["action"]) >= 2:
                    prior_action = episode_data["action"][-2]
                else:
                    prior_action = np.zeros_like(episode_data["action"][-1])
                reward, success, do_terminate, info = calc_reward_cube_push(
                    img=observation["observation.images.main"].numpy(),
                    goal_mask=goal_mask,
                    current_joint_pos=observation["observation.state"].numpy(),
                    action=episode_data["action"][-1],
                    prior_action=prior_action,
                )
                annotated_img = info["annotated_img"]
                print("REWARD:", reward, ", SUCCESS:", success)
                episode_data["next.reward"].append(reward)
                episode_data["next.success"].append(success)
                episode_data["next.done"].append(success or do_terminate)

        if annotated_img is None:
            annotated_img = observation["observation.images.main"].numpy()
        # Draw the goal mask on the annotated image for visualization.
        annotated_img[where_goal] = annotated_img[where_goal] // 2 + np.array([127, 127, 127])

        # Get the follower arms position. This will be used later to convert the policy's output action from
        # relative space to absolute space.
        follower_pos = observation["observation.state"].numpy()
        if first_follower_pos is None:
            first_follower_pos = follower_pos.copy()

        # Just a debug check to make sure that everything we've done so far in this loop iteration didn't
        # take too long. You shouldn't see this warning come up.
        elapsed = to_relative_time(time.perf_counter()) - start_step_time
        if elapsed > period:
            over_time = True
            logging.warning(f"Over time after capturing observation! {elapsed=}")

        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension.
        for name in observation:
            if name.startswith("observation.image"):
                # Small side mission: While we are looping through the image observations, also set up the
                # visualization image.
                if visualize_img:
                    to_visualize[name] = annotated_img
                    to_visualize[name] = cv2.resize(to_visualize[name], (640, 480))
                    to_visualize[name] = cv2.rotate(to_visualize[name], cv2.ROTATE_180)
                    if start_step_time > warmup_s:
                        cv2.putText(
                            to_visualize[name],
                            org=(10, 25),
                            color=(255, 255, 255),
                            text=f"{reward=:.3f}",
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            thickness=1,
                        )
                observation[name] = observation[name].type(torch.float32) / 255
                # HACK use masking to indicate to policy which side needs the cube:
                observation[name][where_goal] = observation[name][where_goal] / 2 + 0.5
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the action with the policy based on the current observation.
        with torch.inference_mode():
            # We don't want to take any longer than this to return a result as then we'd be falling behind the
            # control loop.
            # HACK. subtract 0.025 for a buffer. Careful about this if changing the FPS.
            timeout = (
                period - (to_relative_time(time.perf_counter()) - start_step_time) - 0.025
                if step > 0
                else None
            )
            # HACK. We don't want to warm start the policy as that feature currently assumes warm starting
            # with the most recent step's inference.
            if isinstance(policy, TDMPCPolicy):
                policy.reset()

            # If we're doing teleop don't use the rollout wrapper.
            if isinstance(policy, TeleopPolicy):
                # Returns (seq, batch, action_dim)
                action_sequence = policy.run_inference(observation)
            else:
                # Returns (seq, batch, action_dim)
                action_sequence = policy_rollout_wrapper.provide_observation_get_actions(
                    observation,
                    observation_timestamp=start_step_time,
                    first_action_timestamp=start_step_time,
                    strict_observation_timestamps=step > 0,
                    timeout=timeout,
                )

            if action_sequence is not None:
                action_sequence = action_sequence.squeeze(1)  # remove batch dim
                action = action_sequence[0]

        if action_sequence is not None and visualize_3d:
            digital_twin.set_twin_pose(follower_pos, follower_pos + action_sequence.numpy())

        if visualize_img:
            for name in to_visualize:
                if over_time:
                    purple = np.array([255, 0, 255], dtype=np.uint8)
                    to_visualize[name][:20] = purple
                    to_visualize[name][-20:] = purple
                    to_visualize[name][:, :20] = purple
                    to_visualize[name][:, -20:] = purple
                cv2.imshow(name, cv2.cvtColor(to_visualize[name], cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1)
                if k == ord("p"):
                    cv2.waitKey(0)
                if k == ord("q"):
                    return

        # Decide whether to break out.
        if max_steps is not None and step >= max_steps:
            episode_data["next.done"][-1] = True
        if len(episode_data["next.done"]) > 0 and episode_data["next.done"][-1]:
            # At this point we have collected all keys for the last frame except for the action. Add the last
            # action as zeros
            episode_data["action"] = np.concatenate(
                [episode_data["action"], np.zeros_like(episode_data["action"][-1:])]
            )
            break

        # Order the robot to move
        if is_warmup:
            policy_rollout_wrapper.reset()
            absolute_action = torch.from_numpy(first_follower_pos)
            robot.send_action(absolute_action)
            logging.info("Warming up.")
        else:
            # Special logic for using a ps5 controller to intervene.
            if enable_intevention:
                ps5_reference_joint_pos = (
                    prior_absolute_action.numpy() if prior_absolute_action is not None else first_follower_pos
                )
                ps5_action = ps5_controller.read(ps5_reference_joint_pos.copy())  # absolute
                if ps5_controller.check_flag():
                    surrender_control = False
                if ps5_action is not None or surrender_control:
                    surrender_control = True
                    if ps5_action is None:
                        ps5_action = ps5_reference_joint_pos.copy()
                    action = (
                        torch.from_numpy(ps5_action - follower_pos)
                        if use_relative_actions
                        else torch.from_numpy(ps5_action)
                    )

            if use_relative_actions:
                relative_action = action
                absolute_action = relative_action + torch.from_numpy(follower_pos)
            else:
                absolute_action = action

            # The robot may itself clamp the action, and return the appropriate action.
            absolute_action = robot.send_action(absolute_action)

            if use_relative_actions:
                relative_action = absolute_action - torch.from_numpy(follower_pos)
                episode_data["action"].append(relative_action.numpy())
            else:
                episode_data["action"].append(absolute_action.numpy())

        # Track the prior absolute action for calculating the reward.
        prior_absolute_action = absolute_action.clone()

        # Wait as long as is needed to complete the control loop.
        elapsed = to_relative_time(time.perf_counter()) - start_step_time
        if elapsed > period:
            logging.warning(colored(f"Step took too long! {elapsed=}", "yellow"))
        else:
            busy_wait(period - elapsed - 0.001)

        if visualize_3d and digital_twin.quit_signal_is_set():
            break

        if not is_warmup:
            step += 1
    # ^ Finish rollout loop.

    # Pad the "next" keys with a copy of the last one. They should not be accessed anyway.
    for k in episode_data:
        if k.startswith("next."):
            episode_data[k].append(episode_data[k][-1])

    # Convert to float32 where needed.
    for k in episode_data:
        episode_data[k] = np.stack(episode_data[k])
        if k in ["action", "observation.state", "next.reward", LeRobotDatasetV2.TIMESTAMP_KEY]:
            episode_data[k] = episode_data[k].astype(np.float32)

    # HACK: drop the first frame because of first inference being slow.
    for k in episode_data:
        episode_data[k] = episode_data[k][1:]
    episode_data[LeRobotDatasetV2.FRAME_INDEX_KEY] -= 1
    episode_data[LeRobotDatasetV2.INDEX_KEY] -= 1

    # HACK: Add frames to the episode repeating the last observation, action, reward, and success status.
    # This allows me to effectively increase the magnitude of the success / OOB reward without causing large
    # gradients for the reward network.
    if do_terminate and n_pad_episode_data > 0:
        episode_data[LeRobotDatasetV2.INDEX_KEY] = np.arange(
            len(episode_data[LeRobotDatasetV2.INDEX_KEY]) + n_pad_episode_data
        )
        episode_data[LeRobotDatasetV2.FRAME_INDEX_KEY] = np.arange(
            len(episode_data[LeRobotDatasetV2.FRAME_INDEX_KEY]) + n_pad_episode_data
        )
        episode_data[LeRobotDatasetV2.EPISODE_INDEX_KEY] = np.full(
            (len(episode_data[LeRobotDatasetV2.EPISODE_INDEX_KEY]) + n_pad_episode_data,),
            episode_data[LeRobotDatasetV2.EPISODE_INDEX_KEY][0],
        )
        episode_data[LeRobotDatasetV2.TIMESTAMP_KEY] = np.concatenate(
            [
                episode_data[LeRobotDatasetV2.TIMESTAMP_KEY],
                episode_data[LeRobotDatasetV2.TIMESTAMP_KEY][-1] + (1 + np.arange(n_pad_episode_data)) / fps,
            ]
        )
        observation_keys = [k for k in episode_data if k.startswith("observation.")]
        for k in ["next.reward", "next.success", "next.done", "action", *observation_keys]:
            extra_kwargs = {"mode": "constant", "constant_values": 0} if k == "action" else {"mode": "edge"}
            episode_data[k] = np.pad(
                episode_data[k],
                [(0, n_pad_episode_data)] + [(0, 0)] * (episode_data[k].ndim - 1),
                **extra_kwargs,
            )

    policy_rollout_wrapper.close_thread()

    if visualize_3d:
        digital_twin.close()

    # Reset the timestamp to start form zero.
    episode_data[LeRobotDatasetV2.TIMESTAMP_KEY] -= episode_data[LeRobotDatasetV2.TIMESTAMP_KEY][0]

    return episode_data


def eval_policy(
    robot,
    policy: torch.nn.Module,
    fps: float,
    n_episodes: int,
    n_action_buffer: int = 0,
    warmup_time_s: int = 0,
    use_relative_actions: bool = False,
    max_steps: int | None = None,
    visualize_img: bool = False,
    visualize_3d: bool = False,
    enable_progbar: bool = False,
) -> dict:
    """
    Nothing fancy happening here. Runs `rollout` for `n_episodes` iterations and concatenates all of the
    recorded episode data. Also computes eval metrics for reporting.
    """
    assert isinstance(policy, nn.Module)
    policy.eval()

    start_eval = time.perf_counter()
    episodes_data = []
    sum_rewards = []
    max_rewards = []
    successes = []

    progbar = trange(n_episodes, disable=not enable_progbar)

    for episode_index in progbar:
        with torch.no_grad():
            episode_data = rollout(
                robot,
                policy,
                fps,
                n_action_buffer=n_action_buffer,
                warmup_s=warmup_time_s,
                use_relative_actions=use_relative_actions,
                max_steps=max_steps,
                visualize_img=visualize_img,
                visualize_3d=visualize_3d,
                n_pad_episode_data=policy.config.horizon - 1,
            )
            # Continue the episode and data indices.
            episode_data[LeRobotDatasetV2.EPISODE_INDEX_KEY] += episode_index
            if len(episodes_data) > 0:
                episode_data[LeRobotDatasetV2.INDEX_KEY] += (
                    episodes_data[-1][LeRobotDatasetV2.INDEX_KEY][-1] + 1
                )
            episodes_data.append(episode_data)
            sum_rewards.append(sum(episode_data["next.reward"]))
            max_rewards.append(max(episode_data["next.reward"]))
            successes.append(episode_data["next.success"][-1])

    eval_info = {
        "per_episode": [
            {
                "sum_reward": sum(episode_data["next.reward"]),
                "max_reward": max(episode_data["next.reward"]),
                "success": bool(episode_data["next.success"][-1]),
            }
            for episode_data in episodes_data
        ],
        "episodes": {
            k: np.concatenate([episode_data[k] for episode_data in episodes_data]) for k in episodes_data[0]
        },
        "aggregated": {
            "avg_sum_reward": float(np.mean(sum_rewards)),
            "avg_max_reward": float(np.mean(max_rewards)),
            "pc_success": float(np.mean(successes) * 100),
            "eval_s": time.perf_counter() - start_eval,
            "eval_ep_s": (time.perf_counter() - start_eval) / len(episodes_data),
        },
    }

    # HACK: Bail on autonomous training
    # if eval_info["aggregated"]["pc_success"] == 0:
    #     robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
    #     print("Exited after 0 success")
    #     exit()

    return eval_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fps", type=float)
    parser.add_argument("--n-action-buffer", type=int, default=0)
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch_.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=int,
        default=1,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}"
        ),
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser.add_argument("-n", "--n-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--use-relative-actions", action="store_true")
    parser.add_argument("-v", "--visualize", action="store_true")
    parser.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    init_logging()

    if args.out_dir is None:
        out_dir = Path(f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}")
    else:
        out_dir = Path(args.out_dir)

    pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)

    robot_cfg = init_hydra_config(args.robot_path)
    robot = make_robot(robot_cfg)

    try:
        if not robot.is_connected:
            robot.connect()
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), args.policy_overrides)

        # Check device is available
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_global_seed(hydra_cfg.seed)

        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))

        eval_info = eval_policy(
            robot,
            policy,
            args.fps,
            n_episodes=args.n_episodes,
            n_action_buffer=args.n_action_buffer,
            warmup_time_s=args.warmup_time_s,
            use_relative_actions=args.use_relative_actions,
            max_steps=args.max_steps,
            visualize_img=args.visualize,
            enable_progbar=True,
        )
        pprint(eval_info["aggregated"])

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "eval_info.json", "w") as f:
            json.dump({k: v for k, v in eval_info.items() if k in ["per_episode", "aggregated"]}, f, indent=2)

        logging.info("End of eval")
    finally:
        if robot.is_connected:
            # Disconnect manually to avoid a "Core dump" during process
            # termination due to camera threads not properly exiting.
            robot.disconnect()
