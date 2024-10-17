from pathlib import Path

from hydra.utils import instantiate

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.vision import GoalSetter

cfg = init_hydra_config("lerobot/configs/robot/koch_.yaml")
assert len(cfg["cameras"]) == 1
camera: OpenCVCamera = instantiate(cfg["cameras"][list(cfg["cameras"])[0]])
camera.connect()

for position in ["left", "right", "center"]:
    print(f"Draw goal region for {position}")
    save_goal_mask_path = Path(f"outputs/goal_mask_{position}.npy")
    goal_setter = GoalSetter()
    img = camera.read()
    goal_setter.set_image(img, resize_factor=8)
    k = goal_setter.run()
    goal_setter.save_goal_mask(save_goal_mask_path)
