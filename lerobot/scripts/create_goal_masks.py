"""
Draw goal masks and in-bounds mask.

How to use:
1. Put the cube on the bottom-left-most corner of the left goal region.
2. Run `python lerobot/scripts/create_goal_masks.py`
3. Draw the left goal mask on the opencv window that pops up (click-drag).
4. Move the cube to the bottom-right-most corner of the right goal region.
5. With the opencv window focussed, press any key.
6. Draw the right goal mask.
7. With the opencv window focussed, press any key.
8. Draw the mask for the whole in-bounds region.
9. With the opencv window focussed, press any key.
"""

from pathlib import Path
import cv2

from hydra.utils import instantiate

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.vision import GoalSetter

cfg = init_hydra_config("lerobot/configs/robot/moss.yaml")
assert len(cfg["cameras"]) == 1
camera: OpenCVCamera = instantiate(cfg["cameras"][list(cfg["cameras"])[0]])
camera.connect()

for position in ["left", "right", "center"]:
    print(f"Draw goal region for {position}")
    save_goal_mask_path = Path(f"outputs/goal_mask_{position}.npy")
    goal_setter = GoalSetter()
    for _ in range(10):
        img = camera.async_read()
    goal_setter.set_image(img, resize_factor=8)
    goal_setter.run()
    goal_setter.save_goal_mask(save_goal_mask_path)

# Add these lines to clean up
cv2.destroyAllWindows()
camera.disconnect()
