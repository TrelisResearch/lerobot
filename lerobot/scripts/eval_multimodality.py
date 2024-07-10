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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py -p lerobot/diffusion_pusht eval.n_episodes=10
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.

```
python lerobot/scripts/eval.py \
    -p outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    eval.n_episodes=10
```

Note that in both examples, the repo/folder should contain at least `config.json`, `config.yaml` and
`model.safetensors`.

Note the formatting for providing the number of episodes. Generally, you may provide any number of arguments
with `qualified.parameter.name=value`. In this case, the parameter eval.n_episodes appears as `n_episodes`
nested under `eval` in the `config.yaml` found at
https://huggingface.co/lerobot/diffusion_pusht/tree/main.
"""

import argparse
import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime as dt
from pathlib import Path
from typing import Callable

import cv2
import einops
import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset, Features, Image, Sequence, Value, concatenate_datasets
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from PIL import Image as PILImage
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.policies.rollout_wrapper import PolicyRolloutWrapper
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed


def rollout(
    env: gym.vector.VectorEnv,
    policy: Policy,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    do_simulate_latency: bool = False,
    n_action_buffer: int = 0,
    enable_progbar: bool = False,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
        do_simulate_latency: Whether to simulate latency between observation and action execution. Here, we
            choose to simulate a specific setting where the observation and action happen on the same clock,
            cycle. Therefore, any action chosen on the basis of an action needs to wait till at least the
            next clock cycle.
        n_action_buffer: Controls pre-emptive inference which is useful for rolling out in real-time (or
            when simulating latency). See PolicyRolloutWrapper for more details.
        enable_progbar: Enable a progress bar over rollout steps.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    fps = env.unwrapped.metadata["render_fps"]
    policy_rollout_wrapper = PolicyRolloutWrapper(policy, fps=fps, n_action_buffer=n_action_buffer)

    # Reset the policy and environments.
    policy_rollout_wrapper.reset()

    # For simulating latency, when calling the rollout wrapper we need to provide a timeout of 1 / fps minus a
    # small margin to allow time to actually send the action to the controller (for now this is hard-coded as
    # 1 ms).
    base_timeout = 1 / fps
    timeout_margin = 1e-3
    if do_simulate_latency:
        assert timeout_margin < base_timeout, "FPS is too high for simulating latency. See inline notes."

    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    is_dropped_cycle = False  # track whether the policy failed to return an action
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=not enable_progbar,
        leave=False,
    )
    while not np.all(done):
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Provide the policy with the observation and get the most current action sequence. If simulating
        # latency, don't provide a timeout on the first step. We will pretend the environment is static
        # to begin with.
        with torch.inference_mode():
            n_repeats = 100
            batch_size = observation["observation.state"].shape[0]
            observation_ = {
                k: einops.repeat(v, "b ... -> (n b) ...", n=n_repeats).clone() for k, v in observation.items()
            }
            observation_["observation.state"] += torch.randn_like(observation_["observation.state"]) * 1
            action_sequence = policy_rollout_wrapper.provide_observation_get_actions(
                observation_,
                observation_timestamp=step / fps,
                first_action_timestamp=step / fps,
                strict_observation_timestamps=step > 0,
                timeout=None if step == 0 or not do_simulate_latency else base_timeout - timeout_margin,
            )
            if action_sequence is not None:
                action_sequence = action_sequence.numpy()

        # Apply the action.
        if not do_simulate_latency:
            # When not simulating latency, the action is just the first one from the action sequence (which is
            # guaranteed to not be None).
            action = action_sequence[0]
        elif step == 0:
            # When simulating latency, on the first step we should just use the first action. We are
            # guaranteed that action_sequence is not None.
            action = action_sequence[0]
            # We also need to store the next action. If the next action is not available, we adopt the
            # strategy of repeating the current action.
            if len(action_sequence) > 1:
                next_action = action_sequence[1].copy()
            else:
                next_action = action.copy()
                is_dropped_cycle = True
        else:
            # When simulating latency, all steps after  the first must use the `next_action` from the previous
            # step.
            action = next_action.copy()
            if action_sequence is not None and len(action_sequence) > 1:
                next_action = action_sequence[1].copy()
            else:
                next_action = action.copy()
                is_dropped_cycle = True

        observation, reward, terminated, truncated, info = env.step(action[: env.num_envs])

        if render_callback is not None:
            render_callback(
                env,
                is_dropped_cycle,
                einops.rearrange(action_sequence, "s (n b) a -> s n b a", n=n_repeats, b=batch_size),
            )

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        if "final_info" in info:
            successes = [info["is_success"] if info is not None else False for info in info["final_info"]]
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        done = terminated | truncated | done

        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        is_dropped_cycle = False
        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        "action": torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret["observation"] = stacked_observations

    return ret


def _cv2_putText_by_top_left(text_top_left: tuple[int, int], **kwargs):  # noqa: N802
    """A wrapper around cv2.putText that lets one place it by the top-left coord."""
    text_size, _ = cv2.getTextSize(
        kwargs["text"], kwargs["fontFace"], kwargs["fontScale"], kwargs["thickness"]
    )
    text_bottom_left = (text_top_left[0], min(text_top_left[1] + text_size[1], kwargs["img"].shape[0]))
    cv2.putText(org=text_bottom_left, **kwargs)


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: torch.nn.Module,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    do_simulate_latency: bool = False,
    n_action_buffer: int = 0,
    enable_progbar: bool = False,
    enable_inner_progbar: bool = False,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
        do_simulate_latency: Whether to simulate latency between observation and action execution. See inline
            documentation in `rollout` for more details.
        n_action_buffer: Controls pre-emptive inference which is useful for rolling out in real-time (or
            when simulating latency). See PolicyRolloutWrapper for more details.
        enable_progbar: Enable progress bar over batches.
        enable_inner_progbar: Enable progress bar over steps in each batch.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    assert isinstance(policy, Policy)
    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(
        env: gym.vector.VectorEnv, is_dropped_cycle: bool = False, pusht_action_seq: np.ndarray | None = None
    ):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            img_batch = np.stack([env.envs[i].render() for i in range(n_to_render_now)])  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            img_batch = np.stack(env.call("render")[:n_to_render_now])
        if is_dropped_cycle:
            # put a red border around the frame and some text to make it obvious that an action was dropped
            # (see latency related documentation in `rollout`)
            border_size = max(1, min(img_batch.shape[1:3]) // 40)
            img_batch[:, :border_size] = [255, 0, 0]
            img_batch[:, :, :border_size] = [255, 0, 0]
            img_batch[:, -border_size:] = [255, 0, 0]
            img_batch[:, :, -border_size:] = [255, 0, 0]
            for img in img_batch:
                _cv2_putText_by_top_left(
                    text_top_left=(int(1.5 * border_size), int(1.5 * border_size)),
                    img=img,
                    text="Dropped cycle!",
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    thickness=1,
                    color=(255, 0, 0),
                )
        if pusht_action_seq is not None:
            for i, img in enumerate(img_batch):
                for j in range(pusht_action_seq.shape[1]):
                    actions = pusht_action_seq[:, j, i] / 512 * img.shape[:2]  # (horizon, 2)
                    actions = np.round(actions).astype(int)
                    for k, action in enumerate(actions[::-1]):
                        cv2.circle(
                            img,
                            tuple(action),
                            radius=2,
                            color=(
                                int(255 * k / len(actions)),
                                0,
                                int(255 * (len(actions) - k) / len(actions)),
                            ),
                            thickness=1,
                        )

        ep_frames.append(img_batch)

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    progbar = trange(n_batches, desc="Stepping through eval batches", disable=not enable_progbar)
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env,
            policy,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            do_simulate_latency=do_simulate_latency,
            n_action_buffer=n_action_buffer,
            enable_progbar=enable_inner_progbar,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(
                    0 if episode_data is None else (episode_data["episode_data_index"]["to"][-1].item())
                ),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are not correctly compiling the data.
                assert (
                    episode_data["hf_dataset"]["episode_index"][-1] + 1
                    == this_episode_data["hf_dataset"]["episode_index"][0]
                )
                assert (
                    episode_data["hf_dataset"]["index"][-1] + 1 == this_episode_data["hf_dataset"]["index"][0]
                )
                assert torch.equal(
                    episode_data["episode_data_index"]["to"][-1],
                    this_episode_data["episode_data_index"]["from"][0],
                )
                # Concatenate the episode data.
                episode_data = {
                    "hf_dataset": concatenate_datasets(
                        [episode_data["hf_dataset"], this_episode_data["hf_dataset"]]
                    ),
                    "episode_data_index": {
                        k: torch.cat(
                            [
                                episode_data["episode_data_index"][k],
                                this_episode_data["episode_data_index"][k],
                            ]
                        )
                        for k in ["from", "to"]
                    },
                }

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    total_frames = 0
    data_index_from = start_data_index
    for ep_ix in range(rollout_data["action"].shape[0]):
        num_frames = done_indices[ep_ix].item() + 1  # + 1 to include the first done frame
        total_frames += num_frames

        # TODO(rcadene): We need to add a missing last frame which is the observation
        # of a done state. it is critical to have this frame for tdmpc to predict a "done observation/state"
        ep_dict = {
            "action": rollout_data["action"][ep_ix, :num_frames],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * num_frames),
            "frame_index": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            "next.done": rollout_data["done"][ep_ix, :num_frames],
            "next.reward": rollout_data["reward"][ep_ix, :num_frames].type(torch.float32),
        }
        for key in rollout_data["observation"]:
            ep_dict[key] = rollout_data["observation"][key][ep_ix][:num_frames]
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(data_index_from)
        episode_data_index["to"].append(data_index_from + num_frames)

        data_index_from += num_frames

    data_dict = {}
    for key in ep_dicts[0]:
        if "image" not in key:
            data_dict[key] = torch.cat([x[key] for x in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for img in ep_dict[key]:
                    # sanity check that images are channel first
                    c, h, w = img.shape
                    assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

                    # sanity check that images are float32 in range [0,1]
                    assert img.dtype == torch.float32, f"expect torch.float32, but instead {img.dtype=}"
                    assert img.max() <= 1, f"expect pixels lower than 1, but instead {img.max()=}"
                    assert img.min() >= 0, f"expect pixels greater than 1, but instead {img.min()=}"

                    # from float32 in range [0,1] to uint8 in range [0,255]
                    img *= 255
                    img = img.type(torch.uint8)

                    # convert to channel last and numpy as expected by PIL
                    img = PILImage.fromarray(img.permute(1, 2, 0).numpy())

                    data_dict[key].append(img)

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)
    episode_data_index["from"] = torch.tensor(episode_data_index["from"])
    episode_data_index["to"] = torch.tensor(episode_data_index["to"])

    # TODO(rcadene): clean this
    features = {}
    for key in rollout_data["observation"]:
        if "image" in key:
            features[key] = Image()
        else:
            features[key] = Sequence(length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None))
    features.update(
        {
            "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
            "episode_index": Value(dtype="int64", id=None),
            "frame_index": Value(dtype="int64", id=None),
            "timestamp": Value(dtype="float32", id=None),
            "next.reward": Value(dtype="float32", id=None),
            "next.done": Value(dtype="bool", id=None),
            #'next.success': Value(dtype='bool', id=None),
            "index": Value(dtype="int64", id=None),
        }
    )
    features = Features(features)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    hf_dataset.set_transform(hf_transform_to_torch)
    return {
        "hf_dataset": hf_dataset,
        "episode_data_index": episode_data_index,
    }


def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    env = make_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)

    assert isinstance(policy, nn.Module)
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext():
        info = eval_policy(
            env,
            policy,
            hydra_cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(out_dir) / "videos",
            start_seed=hydra_cfg.seed,
            do_simulate_latency=hydra_cfg.eval.do_simulate_latency,
            n_action_buffer=hydra_cfg.eval.n_action_buffer,
            enable_progbar=True,
            enable_inner_progbar=True,
        )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)
    else:
        try:
            pretrained_policy_path = Path(
                snapshot_download(args.pretrained_policy_name_or_path, revision=args.revision)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            if isinstance(e, HFValidationError):
                error_message = (
                    "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
                )
            else:
                error_message = (
                    "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
                )

            logging.warning(f"{error_message} Treating it as a local directory.")
            pretrained_policy_path = Path(args.pretrained_policy_name_or_path)
        if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=args.out_dir,
            config_overrides=args.overrides,
        )
