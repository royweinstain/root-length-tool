"""Detect and crop individual plates (circular or square) from a full image."""

import cv2
import numpy as np


def detect_plates(image, min_radius_fraction=0.08, max_radius_fraction=0.18,
                   detection_scale=2000):
    """Detect plates in the image — tries circular first, then square.

    Args:
        image: BGR image (full photo with multiple plates).
        min_radius_fraction: Minimum plate radius as fraction of image diagonal.
        max_radius_fraction: Maximum plate radius as fraction of image diagonal.
        detection_scale: Resize longest edge to this for faster detection.

    Returns:
        List of plate dicts: {"center": (x, y), "radius": int, "shape": "circle"|"square",
                              "bbox": (x1, y1, x2, y2)}
        sorted top-to-bottom then left-to-right.

        For backwards compatibility, also works as list of (x_center, y_center, radius) tuples.
    """
    h, w = image.shape[:2]
    scale = detection_scale / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    diagonal = np.sqrt(sh**2 + sw**2)
    min_r = int(diagonal * min_radius_fraction)
    max_r = int(diagonal * max_radius_fraction)

    # Try circular detection first
    circles = _detect_circular_plates(gray, min_r, max_r)

    # If few/no circles found, try square detection
    squares = _detect_square_plates(gray, min_r, max_r)

    # Use whichever found more plates, or combine if both found some
    if len(circles) >= len(squares):
        plates = circles
        shape = "circle"
    else:
        plates = squares
        shape = "square"

    # Map coordinates back to original image space
    result = []
    for p in plates:
        cx = int(p[0] / scale)
        cy = int(p[1] / scale)
        r = int(p[2] / scale)
        result.append((cx, cy, r, shape))

    # Sort top-to-bottom, then left-to-right
    if result:
        median_r = np.median([r for _, _, r, _ in result])
        result.sort(key=lambda c: (c[1] // int(median_r), c[0]))

    # Return as (cx, cy, r) tuples for backwards compatibility
    # Store shape info in a module-level cache for crop_plate to use
    global _last_detection_shape
    _last_detection_shape = shape

    return [(cx, cy, r) for cx, cy, r, _ in result]


# Module-level cache for plate shape
_last_detection_shape = "circle"


def _detect_circular_plates(gray, min_r, max_r):
    """Detect circular plates via Hough circles."""
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r * 2,
        param1=100,
        param2=40,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    filtered = _filter_overlapping(circles)

    validated = []
    for (cx, cy, r) in filtered:
        if _is_valid_plate(gray, cx, cy, r):
            validated.append((cx, cy, r))

    return validated


def _detect_square_plates(gray, min_r, max_r):
    """Detect square/rectangular plates via contour detection.

    Returns plates as (center_x, center_y, half_size) tuples,
    where half_size is analogous to radius (half the plate dimension).
    """
    h, w = gray.shape

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_side = min_r * 2
    max_side = max_r * 2
    min_area = min_side ** 2 * 0.5
    max_area = max_side ** 2 * 2.0

    plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        # Square/rectangle should have 4 vertices
        if len(approx) < 4 or len(approx) > 6:
            continue

        # Check if roughly rectangular using bounding rect
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Aspect ratio should be near 1 for square plates (allow some tolerance)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 2.0:
            continue

        # Size check
        half_size = max(bw, bh) // 2
        if half_size < min_r or half_size > max_r:
            continue

        # Fill ratio — the contour should fill most of its bounding box
        fill_ratio = area / (bw * bh)
        if fill_ratio < 0.5:
            continue

        # Validate: check for contrast between interior and exterior
        cx = x + bw // 2
        cy = y + bh // 2
        if _is_valid_plate(gray, cx, cy, half_size):
            plates.append((cx, cy, half_size))

    # Filter overlapping detections
    if plates:
        plates = _filter_overlapping(np.array(plates))

    return plates


def _is_valid_plate(gray, cx, cy, radius, border_margin=0.05):
    """Check if a detected region is a real plate.

    Works for both polarities and both shapes by checking for rim/edge
    contrast and structure.
    """
    h, w = gray.shape

    margin = int(radius * 0.3)
    if cx - margin < 0 or cy - margin < 0 or cx + margin >= w or cy + margin >= h:
        return False

    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    # Sample at the rim
    rim_vals = []
    for a in angles:
        for frac in [0.95, 1.0, 1.05]:
            rx = int(cx + frac * radius * np.cos(a))
            ry = int(cy + frac * radius * np.sin(a))
            if 0 <= ry < h and 0 <= rx < w:
                rim_vals.append(gray[ry, rx])

    # Sample well inside
    inner_vals = []
    for a in angles:
        ix = int(cx + 0.5 * radius * np.cos(a))
        iy = int(cy + 0.5 * radius * np.sin(a))
        if 0 <= iy < h and 0 <= ix < w:
            inner_vals.append(gray[iy, ix])

    if not rim_vals or not inner_vals:
        return False

    rim_mean = np.mean(rim_vals)
    inner_mean = np.mean(inner_vals)
    rim_std = np.std(rim_vals)

    if abs(rim_mean - inner_mean) > 8 or rim_std > 15:
        return True

    return False


def _filter_overlapping(circles, overlap_threshold=0.5):
    """Remove duplicate/overlapping detections."""
    if len(circles) == 0:
        return []

    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []

    for circle in sorted_circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        is_duplicate = False
        for kx, ky, kr in kept:
            dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            if dist < (r + kr) * overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append((x, y, r))

    return kept


def crop_plate(image, center_x, center_y, radius, padding_fraction=0.02):
    """Crop a single plate from the image.

    Creates a circular or rectangular mask based on the last detection shape.

    Args:
        image: Full BGR image.
        center_x, center_y, radius: Plate parameters (radius = half-size for square).
        padding_fraction: Extra padding as fraction of radius.

    Returns:
        Cropped image of the plate region, and a binary mask of the plate area.
    """
    h, w = image.shape[:2]
    padding = int(radius * padding_fraction)
    r = radius + padding

    x1 = max(0, center_x - r)
    y1 = max(0, center_y - r)
    x2 = min(w, center_x + r)
    y2 = min(h, center_y + r)

    cropped = image[y1:y2, x1:x2].copy()

    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cx = center_x - x1
    cy = center_y - y1

    if _last_detection_shape == "square":
        # Rectangular mask with slight inset
        inset = int(radius * 0.02)
        mx1 = max(0, cx - radius + inset)
        my1 = max(0, cy - radius + inset)
        mx2 = min(cropped.shape[1], cx + radius - inset)
        my2 = min(cropped.shape[0], cy + radius - inset)
        mask[my1:my2, mx1:mx2] = 255
    else:
        # Circular mask
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    return cropped, mask
