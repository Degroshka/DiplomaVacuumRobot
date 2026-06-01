"""
Device I/O helpers for the Webots RGB-D navigation prototype.

The first refactoring step keeps Webots device ownership in the main controller
and moves only frame decoding here. This avoids changing navigation behaviour.
"""

import cv2
import numpy as np


def decode_camera_frame(raw_image, height, width):
    """Decode a Webots BGRA camera buffer into an OpenCV BGR image."""
    if raw_image is None:
        return None
    img = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def decode_depth_image(range_image, height, width):
    """Decode a Webots RangeFinder image into a float32 depth array."""
    if range_image is None:
        return None
    return np.array(range_image, dtype=np.float32).reshape((height, width))
