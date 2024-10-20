from pathlib import Path

import cv2
import numpy as np


def segment_hsv(img) -> tuple[np.ndarray, np.ndarray]:
    # Preprocess.
    img_orig = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Segment with HSV bounds.
    mask = (
        (
            (hsv[..., 0] >= 195 / 2)
            & (hsv[..., 0] <= 260 / 2)
            & (hsv[..., 1] >= 240)
            & (hsv[..., 2] >= 100)
            & (hsv[..., 2] <= 240)
        )
        * 255
    ).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask, img_orig

    # Keep the largest area.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.argmax(areas)
    mask *= 0
    mask[labels == keep + 1] = 255

    # Keep the convex hull.
    hull = cv2.convexHull(cv2.findNonZero(mask))
    mask = cv2.drawContours(mask, [hull], 0, 255, -1)

    annotated_image = cv2.drawContours(img_orig, [hull], 0, 255, 1)

    return (mask > 0).astype(bool), annotated_image


class GoalSetter:
    def __init__(self):
        self._window_name = "Draw goal region"
        self._stroke_radius = 20
        self._left_mouse_button_depressed = False
        self._mask = None
        self._resize_factor = 1.0

    @classmethod
    def from_mask_file(cls, fp: Path | str):
        obj = cls()
        obj._mask = np.load(fp)
        return obj

    def _draw_callback(self, event: int, x: int, y: int, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._left_mouse_button_depressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._left_mouse_button_depressed = False
        if self._left_mouse_button_depressed:
            cv2.circle(
                self._mask,
                (int(round(x / self._resize_factor)), int(round(y / self._resize_factor))),
                max(1, int(round(self._stroke_radius / self._resize_factor))),
                1,
                -1,
            )
            self._imshow()

    def _stroke_size_down_key_callback(self):
        """Decrease the stroke size for all drawing tools by a factor of 2."""
        self._stroke_radius = max(1, self._stroke_radius // 2)

    def _stroke_size_up_key_callback(self):
        """Increase the stroke size for all drawing tools by a factor of 2."""
        self._stroke_radius = min(50, self._stroke_radius * 2)

    def _imshow(self):
        if self._resize_factor != 1.0:
            mask = cv2.resize(
                self._mask,
                (0, 0),
                fx=self._resize_factor,
                fy=self._resize_factor,
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            mask = self._mask
        self._img_rgb[np.where(mask)] = np.array([255, 255, 255])
        cv2.imshow(self._window_name, cv2.cvtColor(self._img_rgb, cv2.COLOR_BGR2RGB))

    def set_image(self, img_rgb, resize_factor: float = 1.0):
        self._mask = np.zeros(shape=img_rgb.shape[:2], dtype=np.uint8)
        self._img_rgb = cv2.resize(img_rgb, (0, 0), fx=resize_factor, fy=resize_factor)
        self._resize_factor = resize_factor

    def _quit_key_callback(self):
        self._quit = True

    def get_goal_mask(self) -> np.ndarray:
        return (self._mask > 0).astype(bool)

    def save_goal_mask(self, fp: Path | str):
        np.save(str(fp), self._mask)

    def run(self):
        self._quit = False
        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._draw_callback)
        self._imshow()
        while not self._quit:
            k = cv2.waitKey(0)
            if k == ord("q"):
                self._quit_key_callback()
            elif k == ord("-"):
                self._stroke_size_down_key_callback()
            elif k in [ord("+"), ord("=")]:
                self._stroke_size_up_key_callback()
            self._imshow()
        return k
