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


# Cardinal growth directions (body -> tip), as (dy, dx) unit vectors in image
# coordinates (y increases downward).
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# HSV range for green plant bodies (cotyledons/leaves). Broader than the
# mark-detection green range so it also catches darker / yellow-green leaves.
_GREEN_LOWER = np.array([30, 35, 35])
_GREEN_UPPER = np.array([95, 255, 255])


def direction_to_vector(direction):
    """Return the (dy, dx) unit vector pointing from plant body toward tip."""
    return DIRECTION_VECTORS.get(direction, DIRECTION_VECTORS["down"])


def detect_growth_direction(plate_image, plate_mask=None, root_mask=None,
                            min_green_area=50, rim_fraction=0.12):
    """Detect the direction roots grow (plant body -> tip) for one plate.

    Roots are not always vertical: depending on how the plate was photographed,
    seedlings may be planted along any rim and grow toward the opposite side.
    This returns one of "up"/"down"/"left"/"right" describing growth direction,
    so the rest of the pipeline can order marks and find tips correctly without
    assuming the plant body is at the top.

    Strategy (green is usually but not always present):
      1. Primary cue — green cotyledons/leaves mark the plant body. The base
         rim is the side the green mass sits on relative to the plate center;
         growth points from there toward the opposite side. Using the offset
         from the plate *center* (rather than the root-mass centroid) is stable
         even when roots are short or under-segmented.
      2. Fallback — no usable green: the seedlings are planted densely along one
         rim, so the rim band with the most root pixels is the base; growth
         points away from it.

    Args:
        plate_image: BGR image of a single plate.
        plate_mask: Binary mask of the plate area.
        root_mask: Binary root mask (used for the geometry fallback and to
                   restrict green detection to plant material).
        min_green_area: Minimum green pixel count to trust the green cue.
        rim_fraction: Width of the near-rim band (fraction of plate size) used
                      by the geometry fallback.

    Returns:
        One of "up", "down", "left", "right" (defaults to "down").
    """
    h, w = plate_image.shape[:2]

    # Plate center = centroid of the plate mask (falls back to image center).
    if plate_mask is not None and np.count_nonzero(plate_mask) > 0:
        mp = np.argwhere(plate_mask > 0)
        cy, cx = mp.mean(axis=0)
    else:
        cy, cx = h / 2.0, w / 2.0

    # --- Primary cue: green plant bodies ---
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    if plate_mask is not None:
        green = cv2.bitwise_and(green, plate_mask)
    if root_mask is not None:
        green = cv2.bitwise_and(green, root_mask)

    if np.count_nonzero(green) >= min_green_area:
        gy, gx = np.argwhere(green > 0).mean(axis=0)
        # Vector from the plant body (green) toward the plate center => growth.
        return _snap_to_cardinal(cy - gy, cx - gx)

    # --- Fallback: densest near-rim band of root pixels is the base ---
    if root_mask is not None and np.count_nonzero(root_mask) > 0:
        band = max(1, int(rim_fraction * min(h, w)))
        densities = {
            "left": int(root_mask[:, :band].sum()),
            "right": int(root_mask[:, w - band:].sum()),
            "up": int(root_mask[:band, :].sum()),
            "down": int(root_mask[h - band:, :].sum()),
        }
        base = max(densities, key=densities.get)
        opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
        return opposite[base]

    return "down"


def _snap_to_cardinal(vy, vx):
    """Snap a (vy, vx) vector to the nearest cardinal growth direction."""
    if abs(vx) >= abs(vy):
        return "right" if vx > 0 else "left"
    return "down" if vy > 0 else "up"
