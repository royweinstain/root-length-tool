"""Detect ruler and compute pixels-per-mm scale."""

import cv2
import numpy as np


def detect_ruler(image, search_fraction=0.25, detection_scale=2000, polarity="light"):
    """Detect a ruler in the image and compute pixel-to-mm scale.

    Searches all four edges for a rectangular ruler with tick marks.
    Supports rulers at any edge, any orientation, and both image polarities.

    Args:
        image: BGR image (full photo).
        search_fraction: Fraction of image to search along each edge.
        detection_scale: Resize longest edge to this for faster processing.
        polarity: "light" = dark ruler on light bg, "dark" = light ruler on dark bg.

    Returns:
        pixels_per_mm (float) in original image coordinates, or None if not found.
    """
    h, w = image.shape[:2]
    scale = detection_scale / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    margin_h = int(sh * search_fraction)
    margin_w = int(sw * search_fraction)

    candidates = [
        ("top", gray[:margin_h, :], "horizontal"),
        ("bottom", gray[sh - margin_h:, :], "horizontal"),
        ("left", gray[:, :margin_w], "vertical"),
        ("right", gray[:, sw - margin_w:], "vertical"),
    ]

    for position, region, orientation in candidates:
        result = _detect_ruler_in_region(region, orientation, polarity)
        if result is not None:
            return result / scale

    return None


def _detect_ruler_in_region(region, orientation, polarity="light"):
    """Find the ruler body and measure tick spacing.

    Returns pixels_per_mm at detection scale, or None.
    """
    # Threshold to find the ruler body
    # For light bg: ruler is dark -> BINARY_INV
    # For dark bg: ruler is light -> BINARY
    if polarity == "dark":
        _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(region, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # The ruler is the largest elongated region
    ruler_cnt = max(contours, key=cv2.contourArea)
    rx, ry, rw, rh = cv2.boundingRect(ruler_cnt)

    aspect = max(rw, rh) / max(min(rw, rh), 1)
    if aspect < 5:
        return None

    ruler_crop = region[ry:ry + rh, rx:rx + rw]

    if orientation == "horizontal":
        tick_strip = ruler_crop[int(rh * 0.6):, :]
        projection = np.mean(tick_strip, axis=0)
    else:
        tick_strip = ruler_crop[:, int(rw * 0.6):]
        projection = np.mean(tick_strip, axis=1)

    # Try both valley and peak detection for ticks
    # (tick marks can be lighter or darker than the ruler body)
    result = _measure_mm_from_ticks(projection, mode="valleys")
    if result is None:
        result = _measure_mm_from_ticks(projection, mode="peaks")
    return result


def _measure_mm_from_ticks(projection, mode="valleys"):
    """Measure mm spacing from a 1D projection of tick marks.

    Args:
        projection: 1D array of mean pixel values along the ruler.
        mode: "valleys" (ticks are darker than ruler body) or
              "peaks" (ticks are lighter than ruler body).

    Returns pixels_per_mm or None.
    """
    if len(projection) < 20:
        return None

    smoothed = np.convolve(projection, np.ones(3) / 3, mode="same")

    # Find extrema based on mode
    positions = []
    for i in range(2, len(smoothed) - 2):
        if mode == "valleys":
            is_extremum = smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]
        else:
            is_extremum = smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]

        if is_extremum:
            left_ref = smoothed[max(0, i - 8):i]
            right_ref = smoothed[i + 1:min(len(smoothed), i + 9)]
            if len(left_ref) == 0 or len(right_ref) == 0:
                continue
            if mode == "valleys":
                prominence = max(left_ref.max(), right_ref.max()) - smoothed[i]
            else:
                prominence = smoothed[i] - min(left_ref.min(), right_ref.min())
            if prominence > 4:
                positions.append(i)

    if len(positions) < 5:
        return None

    spacings = np.diff(positions).astype(float)
    median_sp = np.median(spacings)

    mm_candidates = spacings[(spacings > median_sp * 0.5) &
                             (spacings < median_sp * 1.5)]

    if len(mm_candidates) < 3:
        return None

    pixels_per_mm = np.mean(mm_candidates)

    if pixels_per_mm < 1 or pixels_per_mm > 50:
        return None

    return float(pixels_per_mm)
