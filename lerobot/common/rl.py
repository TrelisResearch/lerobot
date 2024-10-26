"""
For the LeRobot hackathon.
This file contains two types of functions
- a function to reset the arm position between rollouts
- a function to calculate the reward
"""

import cv2
import numpy as np
import torch

from lerobot.common.kinematics import RobotKinematics
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.vision import segment_hsv

# These are for the boundaries of the workspace. If the robot goes out of bounds, the episode is terminated
# and there is a negative reward. You might need to tweak these for your setup. Use `teleop_with_goals.py` to
# check rewards.
GRIPPER_TIP_Z_BOUNDS = (0.008, 0.065)
GRIPPER_TIP_X_BOUNDS = (0.10, 0.30)
GRIPPER_TIP_Y_BOUNDS = (-0.2, 0.2)
GRIPPER_TIP_BOUNDS = np.row_stack([GRIPPER_TIP_X_BOUNDS, GRIPPER_TIP_Y_BOUNDS, GRIPPER_TIP_Z_BOUNDS])


def is_in_bounds(gripper_tip_pos, buffer: float | np.ndarray = 0):
    if not isinstance(buffer, np.ndarray):
        buffer = np.zeros_like(GRIPPER_TIP_BOUNDS) + buffer
    for i, bounds in enumerate(GRIPPER_TIP_BOUNDS):
        assert (bounds[1] - bounds[0]) > buffer[i].sum()
        if gripper_tip_pos[i] < bounds[0] + buffer[i][0] or gripper_tip_pos[i] > bounds[1] - buffer[i][1]:
            return False
    return True


def calc_smoothness_reward(
    action: np.ndarray,
    prior_action: np.ndarray | None = None,
    first_order_coeff: float = -1.0,
    second_order_coeff: float = -1.0,
):
    """Gives a reward based on how "smooth" the robot's movements are.
    Args:
        action: The action (relative joint angle target) that was last executed.
        prior_action: The action executed prior to that (optional).
        first_order_coeff: Set this to a negative value to penalize the magnitude of the `action`. This might
            be considered (very loosely) as a proxy to penalizing for acceleration.
        second_order_coeff: Set this to a negative value to penalize the difference between the `action` and
            prior action. This might be considered (very loosely) as a proxy to penalizing for jerk.
    """
    reward = first_order_coeff * np.linalg.norm(action)
    if prior_action is not None:
        reward += second_order_coeff * np.linalg.norm(action - prior_action)
    return reward


def calc_reward_cube_push(
    img,
    goal_mask,
    current_joint_pos,
    distance_reward_coeff: float = 1 / 24,
    action: np.ndarray | None = None,
    prior_action: np.ndarray | None = None,
    first_order_smoothness_coeff: float = -0,
    second_order_smoothness_coeff: float = -0.02,
    oob_reward: float = -5.0,
    occlusion_limit=55,
    occlusion_reward=-3.0,
) -> tuple[float, bool, bool, dict]:
    """Reward function for the push cube task.

    Reward looks like this:
    1. The closer the cube is to the goal region, the higher the reward. The distance is measured in pixels
        and the `distance_reward_coeff` is a multiplier for that quantity. I chose it so that the reward would
        be -2 when the cube is on the opposing goal region. When the cube is just touching the goal region,
        the reward contribution here is 0.
    2. Once the cube is in contact with the goal region, we start using the intersection area as the reward
        signal. It ranges between 0 and 1.
    3. We have a smoothness penalty as described above in the function `calc_smoothness_reward`. I decided to
        only use the second order penalty.
    4. If the robot arm goes out of bounds (as per the logic in `is_in_bounds`) we give a reward of
        `oob_reward`.
    5. If the cube is occluded (the segmentation algorithm can't find the color of the cube in the image), we
        add `occlusion_reward`. The `occlusion_limit` sets the minimum size of the segmented cube area (in
        pixels) that we need to say that the cube is NOT occluded.
    """
    # Segment out the cube. Also annotate the image with the segmentation contour while we are at it, for
    # visualization purposes.
    obj_mask, annotated_img = segment_hsv(img)

    # Check if the segmented cube is large enough (in units of pixels) for us to consider it a successful
    # segmentation. If so, proceed to calculate the distance/intersection reward. Otherwise, add the occlusion
    # reward (really, a penalty).
    if np.count_nonzero(obj_mask) >= occlusion_limit:
        intersection_area = np.count_nonzero(np.bitwise_and(obj_mask, goal_mask))

        success = False
        if intersection_area <= occlusion_limit:
            # Find the minimum distance between the object and the goal.
            goal_contour = cv2.findContours(
                goal_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )[0]
            obj_contour = cv2.findContours(
                obj_mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )[0]
            obj_points = np.vstack(obj_contour).squeeze()  # shape (N, 2)
            goal_points = np.vstack(goal_contour).squeeze()  # shape (M, 2)
            distances = np.linalg.norm(obj_points[:, None] - goal_points[None, :], axis=-1)  # shape (N, M, 2)
            reward = -np.min(distances) * distance_reward_coeff
        elif intersection_area > 0:
            reward = intersection_area / np.count_nonzero(obj_mask)
            success = reward == 1
    else:
        success = False
        reward = occlusion_reward

    do_terminate = False

    # Check if the gripper went OOB.
    gripper_tip_pos = RobotKinematics.fk_gripper_tip(current_joint_pos)[:3, -1]

    if not is_in_bounds(gripper_tip_pos):
        do_terminate = True
        reward += oob_reward

    # Reward for success condition.
    if success:
        do_terminate = True
        reward += 5

    # Smoothness reward.
    if action is not None:
        reward += calc_smoothness_reward(
            action, prior_action, first_order_smoothness_coeff, second_order_smoothness_coeff
        )

    # Lose 1 for each step to encourage faster completion.
    reward -= 1

    info = {"annotated_img": annotated_img}

    return reward, success, do_terminate, info


def calc_reward_joint_goal(
    current_joint_pos,
    action: np.ndarray | None = None,
    prior_action: np.ndarray | None = None,
    first_order_smoothness_coeff: float = -1.0,
    second_order_smoothness_coeff: float = -1.0,
):
    """Ignore for the Hackathon

    This was for a different RL task I did. The goal was just to move the arm to a given target position.
    I did it to make sure my training loop was working, and also to find good values for the smoothness
    rewards.
    """
    # Whole arm
    goal = np.array([87, 82, 91, 65, 3, 30])
    curr = current_joint_pos
    reward = -np.abs(goal - curr).mean() / 10
    success = np.abs(goal - curr).max() <= 3

    do_terminate = False

    gripper_tip_pos = RobotKinematics.fk_gripper_tip(current_joint_pos)[:3, -1]
    if not is_in_bounds(gripper_tip_pos):
        reward -= 5
        do_terminate = True

    if success:
        do_terminate = True
        reward += 1

    if action is not None:
        reward += calc_smoothness_reward(
            action, prior_action, first_order_smoothness_coeff, second_order_smoothness_coeff
        )

    return reward, success, do_terminate


def _go_to_pos(robot, pos, tol=None):
    if tol is None:
        tol = np.array([3, 3, 3, 10, 3, 3])
    while True:
        robot.send_action(pos)
        current_pos = robot.follower_arms["main"].read("Present_Position")
        busy_wait(1 / 30)
        if np.all(np.abs(current_pos - pos.numpy()) < tol):
            break


def reset_for_joint_pos(robot: ManipulatorRobot):
    """Ignore for the Hackathon

    This was for a different RL task I did. The goal was just to move the arm to a given target position.
    I did it to make sure my training loop was working, and also to find good values for the smoothness
    rewards.
    """
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    reset_pos = robot.follower_arms["main"].read("Present_Position")
    while True:
        # For reference: goal = [87, 82, 91, 65, 3, 30]
        reset_pos[0] = np.random.uniform(70, 110)
        reset_pos[1] = np.random.uniform(70, 90)
        reset_pos[2] = np.random.uniform(80, 110)
        reset_pos[3] = np.random.uniform(45, 90)
        reset_pos[4] = np.random.uniform(-50, 50)
        reset_pos[5] = np.random.uniform(0, 90)
        if is_in_bounds(RobotKinematics.fk_gripper_tip(reset_pos)[:3, -1], buffer=0.02):
            break
    reset_pos = torch.from_numpy(reset_pos)
    _go_to_pos(robot, reset_pos)


def reset_for_cube_push(robot: ManipulatorRobot, right=True):
    """Reset the arm at the start of an episode.

    Reset to the right with right=True, or left with right=False.

    I've hard coded the reset position.
    I've also hard coded some intermediate positions to move to. This just prevents the arm from hitting
    things or pushing the cube out of bounds in between resets. It's like a very poor man's path planning lol.

    You can run `python lerobot/common/rl` to test the rest. Check the code in `if __name__ == "__main__":`.
    """
    robot.follower_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
    staging_pos = torch.tensor([0.21972656, 82.74902, 56.601562, 68.115234, 1.5820312, 1.3097577]).float()
    while True:
        reset_pos = torch.tensor(
            [
                # np.random.uniform(125, 135) if right else np.random.uniform(45, 55),
                np.random.uniform(34, 36) if right else np.random.uniform(-32, -22),
                # np.random.uniform(54, 58),
                np.random.uniform(63, 65),
                # np.random.uniform(50, 52),
                np.random.uniform(73, 77),
                # np.random.uniform(78, 98),
                np.random.uniform(53, 73),
                # np.random.uniform(-41, -31) if right else np.random.uniform(31, 41),
                np.random.uniform(-36, -26) if right else np.random.uniform(10, 20),
                # np.random.uniform(0, 20),
                np.random.uniform(0, 20),
            ]
        ).float()
        if is_in_bounds(
            RobotKinematics.fk_gripper_tip(reset_pos.numpy())[:3, -1],
            buffer=np.array([[0.02, 0.02], [0.02, 0.02], [0.02, 0.01]]),
        ):
            break
    intermediate_pos = torch.from_numpy(robot.follower_arms["main"].read("Present_Position"))
    intermediate_pos[1] = staging_pos[1]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 7, 5, 10, 5, 5]))
    intermediate_pos[1:] = staging_pos[1:]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 7, 5, 10, 5, 5]))
    if right and staging_pos[0] > intermediate_pos[0]:  # noqa: SIM114
        _go_to_pos(robot, staging_pos, tol=np.array([5, 7, 5, 10, 5, 5]))
    elif (not right) and staging_pos[0] < intermediate_pos[0]:
        _go_to_pos(robot, staging_pos, tol=np.array([5, 7, 5, 10, 5, 5]))
    intermediate_pos = staging_pos.clone()
    intermediate_pos[0] = reset_pos[0]
    _go_to_pos(robot, intermediate_pos, tol=np.array([5, 7, 5, 10, 5, 5]))
    _go_to_pos(robot, reset_pos, tol=np.array([3, 7, 3, 3, 3, 3]))


if __name__ == "__main__":
    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.utils.utils import init_hydra_config

    robot = make_robot(init_hydra_config("lerobot/configs/robot/moss.yaml"))
    robot.connect()
    reset_for_cube_push(robot, right=True)
    reset_for_cube_push(robot, right=False)
    robot.disconnect()
