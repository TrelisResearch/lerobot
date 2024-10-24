# robot: KochRobot = make_robot("lerobot/configs/robot/koch_.yaml")
# robot.follower_arms.
import threading
import time

import inputs
import numpy as np

from lerobot.common.kinematics import RobotKinematics

EE_SENSITIVITY = 5e-5
JOINT_SENSITIVITY = 0.04


def read_gamepad(gamepad_last_state):
    while True:
        events = inputs.get_gamepad()
        for event in events:
            if event.code == "ABS_X":  # Left stick X-axis
                gamepad_last_state[0] = event.state
            if event.code == "ABS_Y":  # Left stick Y-axis
                gamepad_last_state[1] = event.state
            if event.code == "ABS_RX":  # Right stick X-axis
                gamepad_last_state[2] = event.state
            if event.code == "ABS_RY":  # Right stick Y-axis
                gamepad_last_state[3] = event.state
            if event.code == "ABS_RZ":  # right trigger
                gamepad_last_state[4] = event.state
            if event.code == "ABS_Z":  # left trigger
                gamepad_last_state[5] = event.state
            if event.code == "BTN_TL":
                gamepad_last_state[6] = event.state
            if event.code == "BTN_TR":
                gamepad_last_state[7] = event.state
            if event.code == "BTN_SOUTH" and event.state == 1:
                gamepad_last_state[8] = 1

        time.sleep(0.0001)


class PS5Controller:
    gamepad_last_state = [128, 128, 128, 128, 0, 0, 0, 0, 0]
    gamepad_read_thread = threading.Thread(target=read_gamepad, args=(gamepad_last_state,))

    def __init__(self):
        try:
            inputs.devices.gamepads[0]
        except IndexError as e:
            raise inputs.UnpluggedError("No gamepad found.") from e
        self._kill = False
        self._flag_raised = False
        self._lock = threading.Lock()
        if not self.gamepad_read_thread.is_alive():
            self.gamepad_read_thread.start()

    def read(self, reference_joint_state):
        time.sleep(0.0001)  # give some time to read the gamepad
        gamepad_last_state = self.gamepad_last_state.copy()
        left_right = gamepad_last_state[0] - 128
        forward_backward = gamepad_last_state[1] - 128
        up = gamepad_last_state[4]
        down = gamepad_last_state[5]
        wrist_flex = gamepad_last_state[3] - 128
        wrist_twist = gamepad_last_state[2] - 128
        if abs(left_right) <= 10:
            left_right = 0
        if abs(forward_backward) <= 10:
            forward_backward = 0
        if abs(wrist_flex) <= 10:
            wrist_flex = 0
        if abs(wrist_twist) <= 10:
            wrist_twist = 0
        if abs(up) <= 5:
            up = 0
        if abs(down) <= 5:
            down = 0

        if (
            left_right == 0
            and forward_backward == 0
            and wrist_flex == 0
            and wrist_twist == 0
            and up == 0
            and down == 0
            and not gamepad_last_state[6]
            and not gamepad_last_state[7]
        ):
            return None

        current_ee_pose = RobotKinematics.fk_gripper(reference_joint_state)

        # Compute IK
        if left_right != 0 or forward_backward != 0 or up != 0 or down != 0:
            desired_ee_pose = (
                np.array(
                    [
                        [1, 0, 0, -left_right * EE_SENSITIVITY],
                        [0, 1, 0, forward_backward * EE_SENSITIVITY],
                        [0, 0, 1, (up - down) * EE_SENSITIVITY / 4],
                        [0, 0, 0, 1],
                    ]
                )
                @ current_ee_pose
            )
            target_joint_state = RobotKinematics.ik(
                reference_joint_state, desired_ee_pose, position_only=left_right != 0
            )
            if target_joint_state is None:
                raise AssertionError
        else:
            target_joint_state = reference_joint_state.copy()

        if wrist_flex != 0:
            target_joint_state[3] += wrist_flex * JOINT_SENSITIVITY

        if wrist_twist != 0:
            target_joint_state[4] += wrist_twist * JOINT_SENSITIVITY

        if gamepad_last_state[6]:
            target_joint_state[5] -= 3

        if gamepad_last_state[7]:
            target_joint_state[5] += 3

        print(target_joint_state)

        return target_joint_state

    def check_flag(self) -> bool:
        if self.gamepad_last_state[-1] > 0:
            self.gamepad_last_state[-1] = 0
            return True
        else:
            return False
