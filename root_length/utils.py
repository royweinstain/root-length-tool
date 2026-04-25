"""Shared utilities for the root length pipeline."""

import cv2
import numpy as np


def detect_polarity(image):
    """Detect whether the image has light or dark background.

    Returns:
        "light" if background is bright (roots are dark on white/light agar)
        "dark" if background is dark (roots are light on dark background)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Simple heuristic: if overall image is mostly dark, it's dark background
    # Typical threshold: images with dark bg have mean < 80
    if mean_brightness < 80:
        return "dark"
    return "light"
