import argparse
import logging
import time

import cv2

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.digital_twin import DigitalTwin
from lerobot.common.utils.utils import init_hydra_config

parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("-v", "--visualize", nargs="+", choices=["camera", "twin"], default=[])
args = parser.parse_args()


robot: ManipulatorRobot = make_robot(init_hydra_config("lerobot/configs/robot/moss.yaml"))

robot.connect()

if "twin" in args.visualize:
    digital_twin = DigitalTwin()


while True:
    start = time.perf_counter()
    obs_dict, _ = robot.teleop_step(record_data=True)
    follower_pos = robot.follower_arms["main"].read("Present_Position")
    print(follower_pos)
    # print(follower_pos)
    if "camera" in args.visualize:
        # Try different possible keys for camera images
        camera_image = None
        possible_keys = [
            "observation.images.main",
            "observation.images.laptop",  # Based on your config file
            "images.laptop",
            "laptop"
        ]
        
        for key in possible_keys:
            if key in obs_dict:
                camera_image = obs_dict[key]
                break
        
        if camera_image is not None:
            cv2.imshow(
                "window",
                cv2.resize(
                    cv2.cvtColor(camera_image.numpy(), cv2.COLOR_RGB2BGR),
                    (0, 0),
                    fx=8,
                    fy=8,
                ),
            )
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        else:
            logging.warning("No camera image found in observation dictionary")
            print("Available keys:", obs_dict.keys())
    if "twin" in args.visualize:
        digital_twin.set_twin_pose(follower_pos)
        if digital_twin.quit_signal_is_set():
            break
    elapsed = time.perf_counter() - start
    if elapsed > 1 / args.fps:
        logging.warning(f"Loop iteration went overtime: {elapsed=}.")
    else:
        busy_wait((1 / args.fps) - elapsed)
