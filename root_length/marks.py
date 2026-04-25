"""Detect colored pen marks on roots."""

import cv2
import numpy as np


# HSV ranges for mark colors
COLOR_RANGES = {
    "red": [
        {"lower": np.array([0, 70, 50]), "upper": np.array([10, 255, 255])},
        {"lower": np.array([170, 70, 50]), "upper": np.array([180, 255, 255])},
    ],
    "blue": [
        {"lower": np.array([108, 80, 50]), "upper": np.array([130, 255, 255])},
    ],
    "green": [
        {"lower": np.array([35, 80, 50]), "upper": np.array([75, 255, 255])},
    ],
}


def detect_marks(plate_image, color="red", plate_mask=None, root_mask=None,
                 min_mark_area=10, max_mark_area=500):
    """Detect colored pen marks on the plate.

    For colored marks (red/blue/green), uses HSV color filtering.
    Marks are expected to be small, roughly horizontal lines crossing roots.

    Args:
        plate_image: BGR image of a single plate.
        color: Mark color to detect ("red", "blue", "green").
        plate_mask: Optional mask to restrict detection to plate area.
        root_mask: Optional root mask — marks should be near/on roots.
        min_mark_area: Minimum area for a mark region.
        max_mark_area: Maximum area for a mark (to reject large colored objects like leaves).

    Returns:
        mark_mask: Binary mask where marks are white (255).
    """
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    ranges = COLOR_RANGES.get(color)
    if ranges is None:
        return np.zeros(plate_image.shape[:2], dtype=np.uint8)

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask = cv2.inRange(hsv, r["lower"], r["upper"])
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Apply plate mask
    if plate_mask is not None:
        combined_mask = cv2.bitwise_and(combined_mask, plate_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Filter contours by size and shape
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(combined_mask)

    # If we have a root mask, expand it to create a "near roots" region
    near_roots = None
    if root_mask is not None:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        near_roots = cv2.dilate(root_mask, dilate_kernel)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_mark_area or area > max_mark_area:
            continue

        # Check if mark is near a root (if root mask provided)
        if near_roots is not None:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                mx = int(M["m10"] / M["m00"])
                my = int(M["m01"] / M["m00"])
                if 0 <= my < near_roots.shape[0] and 0 <= mx < near_roots.shape[1]:
                    if near_roots[my, mx] == 0:
                        continue  # not near any root

        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask


def auto_detect_mark_color(plate_image, plate_mask=None):
    """Try to auto-detect which color marks are present.

    Tests each color and returns the one with the most detected area.

    Returns:
        Best color name (str), or None if no marks detected.
    """
    best_color = None
    best_area = 0

    for color in ["red", "blue", "green"]:
        mask = detect_marks(plate_image, color=color, plate_mask=plate_mask)
        area = np.count_nonzero(mask)
        if area > best_area:
            best_area = area
            best_color = color

    return best_color if best_area > 50 else None
