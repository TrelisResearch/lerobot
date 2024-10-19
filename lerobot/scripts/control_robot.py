"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py teleoperate

# Remove the cameras from the robot definition. They are not used in 'teleoperate' anyway.
python lerobot/scripts/control_robot.py teleoperate --robot-overrides '~cameras'
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/koch_test \
    --episode 0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/koch_pick_place_lego \
    --num-episodes 50 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

- Train on this dataset with the ACT policy:
```bash
DATA_DIR=data python lerobot/scripts/train.py \
    policy=act_koch_real \
    env=koch_real \
    dataset_repo_id=$USER/koch_pick_place_lego \
    hydra.run.dir=outputs/train/act_koch_real
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py record \
    --fps 30 \
    --root data \
    --repo-id $USER/eval_act_koch_real \
    --num-episodes 10 \
    --warmup-time-s 2 \
    --episode-time-s 30 \
    --reset-time-s 10
    -p outputs/train/act_koch_real/checkpoints/080000/pretrained_model
```
"""

import argparse
import logging
import shutil
import time
import traceback
from collections import defaultdict
from functools import cache
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image
from safetensors.torch import save_file
from termcolor import colored

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.online_buffer import LeRobotDatasetV2
from lerobot.common.datasets.utils import flatten_dict
from lerobot.common.policies.factory import make_policy
from lerobot.common.rl import calc_reward_cube_push
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot, get_arm_id
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.common.vision import GoalSetter
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.utils import say

########################################################################################
# Utilities
########################################################################################


def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_tensor.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def log_control_info(robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    for name in robot.leader_arms:
        key = f"read_leader_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRlead", robot.logs[key])

    for name in robot.follower_arms:
        key = f"write_follower_{name}_goal_pos_dt_s"
        if key in robot.logs:
            log_dt("dtWfoll", robot.logs[key])

        key = f"read_follower_{name}_pos_dt_s"
        if key in robot.logs:
            log_dt("dtRfoll", robot.logs[key])

    for name in robot.cameras:
        key = f"read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


########################################################################################
# Control modes
########################################################################################


def calibrate(robot: Robot, arms: list[str] | None):
    available_arms = []
    for name in robot.follower_arms:
        arm_id = get_arm_id(name, "follower")
        available_arms.append(arm_id)
    for name in robot.leader_arms:
        arm_id = get_arm_id(name, "leader")
        available_arms.append(arm_id)

    unknown_arms = [arm_id for arm_id in arms if arm_id not in available_arms]

    available_arms_str = " ".join(available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


def teleoperate(robot: Robot, fps: int | None = None, teleop_time_s: float | None = None):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    start_teleop_t = time.perf_counter()
    while True:
        start_loop_t = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
            break


def record(
    robot: Robot,
    policy: torch.nn.Module | None = None,
    hydra_cfg: DictConfig | None = None,
    fps: int | None = None,
    root="data",
    repo_id="lerobot/debug",
    warmup_time_s=2,
    episode_time_s=10,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    force_override=False,
    n_pad_frames=0,
):
    # TODO(rcadene): Add option to record logs
    # TODO(rcadene): Clean this function via decomposition in higher level functions

    assert policy is None

    _, dataset_name = repo_id.split("/")
    if dataset_name.startswith("eval_") and policy is None:
        raise ValueError(
            f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
        )

    if not video:
        raise NotImplementedError()

    if not robot.is_connected:
        robot.connect()

    local_dir = Path(root) / repo_id
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    dataset = LeRobotDatasetV2(local_dir, fps=fps)

    if is_headless():
        logging.info(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )

    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    exit_early = False
    rerecord_episode = False
    stop_recording = False

    # Only import pynput if not in a headless environment
    if not is_headless():
        from pynput import keyboard

        def on_press(key):
            nonlocal exit_early, rerecord_episode, stop_recording
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Exiting loop...")
                    exit_early = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    rerecord_episode = True
                    exit_early = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    stop_recording = True
                    exit_early = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    # Load policy if any
    if policy is not None:
        # Check device is available
        device = get_safe_torch_device(hydra_cfg.device, log=True)

        policy.eval()
        policy.to(device)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        set_global_seed(hydra_cfg.seed)

        # override fps using policy fps
        fps = hydra_cfg.env.fps

    # Execute a few seconds without recording data, to give times
    # to the robot devices to connect and start synchronizing.
    timestamp = 0
    start_warmup_t = time.perf_counter()
    is_warmup_print = False
    while timestamp < warmup_time_s:
        if not is_warmup_print:
            logging.info("Warming up (no data recording)")
            say("Warming up", blocking=True)
            is_warmup_print = True

        start_loop_t = time.perf_counter()

        if policy is None:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

        if not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_warmup_t

    # Start recording all episodes
    while (
        episode_index := (dataset.get_unique_episode_indices()[-1] + 1) if len(dataset) > 0 else 0
    ) < num_episodes:
        direction = "left" if episode_index % 2 == 0 else "right"
        goal_setter = GoalSetter.from_mask_file(f"outputs/goal_mask_{direction}.npy")
        logging.info(f"Recording episode {episode_index}")
        say(f"Recording episode {episode_index}. Go {direction}", blocking=False)

        goal_mask = goal_setter.get_goal_mask()
        where_goal = np.where(goal_mask > 0)

        ep_dict = defaultdict(list)
        frame_index = 0
        timestamp = 0
        start_episode_t = time.perf_counter()
        prior_relative_action = None
        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()

            if policy is None:
                observation, action = robot.teleop_step(record_data=True)
            else:
                observation = robot.capture_observation()

            assert len(action) == 1

            # To relative action
            action["action"] = action["action"] - observation["observation.state"]

            image_keys = [k for k in observation if k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX)]
            assert len(image_keys) == 1

            if not is_headless():
                for key in image_keys:
                    this_relative_action = action["action"].numpy()
                    reward, success, do_terminate, info = calc_reward_cube_push(
                        img=observation[key].numpy(),
                        goal_mask=goal_mask,
                        current_joint_pos=observation["observation.state"].numpy(),
                        action=this_relative_action,
                        prior_action=prior_relative_action,
                    )
                    prior_relative_action = this_relative_action.copy()
                    annotated_img = info["annotated_img"]
                    annotated_img[where_goal] = (
                        annotated_img[where_goal]
                        - (annotated_img[where_goal] - np.array([255, 255, 255])) // 2
                    )
                    # Hflip to better orient the viewer
                    annotated_img = annotated_img[:, ::-1]
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
                    cv2.imshow(key, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                    ep_dict["next.reward"].append(reward.astype(np.float32))
                    ep_dict["next.success"].append(success)
                    if success:
                        say("Goal accomplished.", blocking=True)
                        exit_early = True
                    elif do_terminate:
                        say("Failed.", blocking=True)
                        exit_early = True
                cv2.waitKey(1)

            for key in observation:
                data = observation[key].numpy()
                if key in image_keys:
                    # Highlight the goal region.
                    data[where_goal] = data[where_goal] // 2 + np.array([127, 127, 127], dtype=np.uint8)
                ep_dict[key].append(data)

            for key in action:
                ep_dict[key].append(action[key].numpy().astype(np.float32))

            frame_index += 1

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_episode_t
            if exit_early:
                exit_early = False
                break

        if not stop_recording:
            logging.info("Reset the environment")
            say("Reset the environment")
            while not exit_early:
                start = time.perf_counter()
                robot.teleop_step()
                busy_wait(1 / fps - (time.perf_counter() - start))
                if exit_early:
                    exit_early = False
                    break

        if rerecord_episode:
            say("Re-recording the last episode.")
            rerecord_episode = False
            continue

        # Compile episode data and add to episode dictionary.
        num_frames = frame_index
        ep_dict[LeRobotDatasetV2.EPISODE_INDEX_KEY] = np.full((num_frames,), episode_index, dtype=int)
        ep_dict[LeRobotDatasetV2.INDEX_KEY] = np.arange(num_frames)
        ep_dict[LeRobotDatasetV2.FRAME_INDEX_KEY] = np.arange(num_frames)
        ep_dict[LeRobotDatasetV2.TIMESTAMP_KEY] = np.arange(num_frames, dtype=np.float32) / fps
        for k in chain(action, observation, ["next.reward"]):
            ep_dict[k] = np.stack(ep_dict[k])
            if not k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX):
                ep_dict[k] = ep_dict[k].astype(np.float32)
        ep_dict["next.success"] = np.array(ep_dict["next.success"])
        done = np.zeros(num_frames, dtype=bool)
        done[-1] = True
        ep_dict["next.done"] = done

        # HACK: Add frames to the episode repeating the last observation, action, reward, and success status.
        # This allows me to effectively increase the magnitude of the success / OOB reward without causing
        # large gradients for the reward network.
        if do_terminate and n_pad_frames > 0:
            ep_dict[LeRobotDatasetV2.INDEX_KEY] = np.arange(
                len(ep_dict[LeRobotDatasetV2.INDEX_KEY]) + n_pad_frames
            )
            ep_dict[LeRobotDatasetV2.FRAME_INDEX_KEY] = np.arange(
                len(ep_dict[LeRobotDatasetV2.FRAME_INDEX_KEY]) + n_pad_frames
            )
            ep_dict[LeRobotDatasetV2.EPISODE_INDEX_KEY] = np.full(
                (len(ep_dict[LeRobotDatasetV2.EPISODE_INDEX_KEY]) + n_pad_frames,),
                ep_dict[LeRobotDatasetV2.EPISODE_INDEX_KEY][0],
            )
            ep_dict[LeRobotDatasetV2.TIMESTAMP_KEY] = np.concatenate(
                [
                    ep_dict[LeRobotDatasetV2.TIMESTAMP_KEY],
                    ep_dict[LeRobotDatasetV2.TIMESTAMP_KEY][-1] + (1 + np.arange(n_pad_frames)) / fps,
                ]
            )
            observation_keys = [k for k in ep_dict if k.startswith("observation.")]
            for k in ["next.reward", "next.success", "next.done", "action", *observation_keys]:
                extra_kwargs = (
                    {"mode": "constant", "constant_values": 0} if k == "action" else {"mode": "edge"}
                )
                ep_dict[k] = np.pad(
                    ep_dict[k], [(0, n_pad_frames)] + [(0, 0)] * (ep_dict[k].ndim - 1), **extra_kwargs
                )

        dataset.add_episodes(ep_dict)

        is_last_episode = stop_recording or len(dataset.get_unique_episode_indices()) == num_episodes

        # Wait if necessary
        if not is_last_episode:
            for _ in tqdm.trange(reset_time_s, desc="Waiting"):
                time.sleep(1)
                if exit_early:
                    exit_early = False
                    break

        if is_last_episode:
            logging.info("Done recording")
            say("Done recording", blocking=True)
            if not is_headless():
                listener.stop()
            break

    robot.disconnect()
    if not is_headless():
        cv2.destroyAllWindows()

    num_episodes = episode_index

    if run_compute_stats:
        logging.info("Computing dataset statistics")
        say("Computing dataset statistics")
        stats = compute_stats(dataset)
        stats_path = dataset.storage_dir / "stats.safetensors"
        save_file(flatten_dict(stats), stats_path)
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    if push_to_hub:
        # TODO(now)
        pass

    logging.info("Exiting")
    say("Exiting", blocking=True)
    return dataset


# def replay(robot: Robot, episode: int, fps: int | None = None, root="data", repo_id="lerobot/debug"):
#     # TODO(rcadene): Add option to record logs
#     local_dir = Path(root) / repo_id
#     if not local_dir.exists():
#         raise ValueError(local_dir)

#     dataset = LeRobotDataset(repo_id, root=root)
#     items = dataset.hf_dataset.select_columns("action")
#     from_idx = dataset.episode_data_index["from"][episode].item()
#     to_idx = dataset.episode_data_index["to"][episode].item()

#     if not robot.is_connected:
#         robot.connect()

#     logging.info("Replaying episode")
#     say("Replaying episode", blocking=True)
#     for idx in range(from_idx, to_idx):
#         start_episode_t = time.perf_counter()

#         action = items[idx]["action"]
#         robot.send_action(action)

#         dt_s = time.perf_counter() - start_episode_t
#         busy_wait(1 / fps - dt_s)

#         dt_s = time.perf_counter() - start_episode_t
#         log_control_info(robot, dt_s, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    base_parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser_calib = subparsers.add_parser("calibrate", parents=[base_parser])
    parser_calib.add_argument(
        "--arms",
        type=str,
        nargs="*",
        help="List of arms to calibrate (e.g. `--arms left_follower right_follower left_leader`)",
    )

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--warmup-time-s",
        type=int,
        default=10,
        help="Number of seconds before starting data collection. It allows the robot devices to warmup and synchronize.",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--reset-time-s",
        type=int,
        default=60,
        help="Number of seconds for resetting the environment after each episode.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser_record.add_argument(
        "--n-pad-frames",
        type=int,
        default=0,
        help="Number of padding frames to add to the end of each terminated episode. These are used to "
        "effectively multiply the final reward by `n_pad_frames+1`.",
    )

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    robot_overrides = args.robot_overrides
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["robot_overrides"]

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot(robot_cfg)

    if control_mode == "calibrate":
        calibrate(robot, **kwargs)

    elif control_mode == "teleoperate":
        teleoperate(robot, **kwargs)

    elif control_mode == "record":
        pretrained_policy_name_or_path = args.pretrained_policy_name_or_path
        policy_overrides = args.policy_overrides
        del kwargs["pretrained_policy_name_or_path"]
        del kwargs["policy_overrides"]

        policy_cfg = None
        if pretrained_policy_name_or_path is not None:
            pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
            policy_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
            policy = make_policy(hydra_cfg=policy_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
            record(robot, policy, policy_cfg, **kwargs)
        else:
            record(robot, **kwargs)

    elif control_mode == "replay":
        # TODO(now)
        raise AssertionError
        # replay(robot, **kwargs)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
