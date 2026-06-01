"""Root segmentation and skeletonization."""

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

from .utils import _GREEN_LOWER, _GREEN_UPPER


def segment_roots(plate_image, plate_mask=None, polarity="light",
                   block_size=51, c_offset=10,
                   min_root_area=200, rim_shrink=0.08, top_crop_fraction=0.2,
                   min_elongation=3.0):
    """Segment roots from a single plate image.

    Uses adaptive thresholding to handle lighting variation.
    Filters out non-root objects (text, rim) by shape and position.

    Args:
        plate_image: BGR image of a single plate.
        plate_mask: Optional binary mask of the plate area (to ignore outside).
        polarity: "light" = dark roots on light background,
                  "dark" = light roots on dark background.
        block_size: Block size for adaptive thresholding (must be odd).
        c_offset: Constant subtracted from mean in adaptive thresholding.
        min_root_area: Minimum area in pixels for a root component.
        rim_shrink: Fraction to shrink the plate mask inward (removes rim artifacts).
        top_crop_fraction: Fraction of plate top to exclude (where text labels live).
        min_elongation: Minimum aspect ratio (height/width) to keep a component.
                        Roots are tall and thin; text and noise are not.

    Returns:
        binary_mask: Binary mask where roots are white (255).
        labels: Labeled image where each root has a unique integer ID.
        num_roots: Number of detected roots.
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Thresholding depends on polarity
    if polarity == "light":
        # Enhance local contrast to help detect faint roots
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # Dark roots on light background → adaptive threshold (handles uneven lighting)
        binary = cv2.adaptiveThreshold(
            gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, c_offset
        )
    else:
        # Light roots on dark background → global threshold works well
        # because the contrast is already high (bright roots on black agar)
        if plate_mask is not None:
            plate_pixels = gray[plate_mask > 0]
            # Roots are the brightest ~10% of the plate area
            thresh_val = np.percentile(plate_pixels, 90)
        else:
            thresh_val = 30
        # Ensure a minimum threshold to avoid picking up noise
        thresh_val = max(thresh_val, 25)
        _, binary = cv2.threshold(gray, int(thresh_val), 255, cv2.THRESH_BINARY)

    # Build a refined mask: shrink plate mask inward to exclude rim
    if plate_mask is not None:
        shrink_px = int(min(h, w) * rim_shrink)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (shrink_px * 2, shrink_px * 2))
        refined_mask = cv2.erode(plate_mask, kernel_erode)

        # Also mask out the top portion where text labels typically are
        top_cutoff = int(h * top_crop_fraction)
        refined_mask[:top_cutoff, :] = 0

        binary = cv2.bitwise_and(binary, refined_mask)

    # Morphological cleanup
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Remove small noise
    binary_bool = binary.astype(bool)
    binary_bool = remove_small_objects(binary_bool, max_size=min_root_area)
    binary = (binary_bool.astype(np.uint8)) * 255

    # Filter by elongation: keep only components that are tall and thin (root-like)
    num_labels, labels_raw = cv2.connectedComponents(binary)
    filtered = np.zeros_like(binary)

    root_id = 0
    labels = np.zeros_like(labels_raw)
    for label_id in range(1, num_labels):
        component = (labels_raw == label_id).astype(np.uint8)
        ys, xs = np.where(component)
        if len(ys) == 0:
            continue

        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1

        # Elongation: ratio of longest to shortest dimension
        # Roots can grow vertically OR horizontally depending on plate orientation
        elongation = max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1)

        # Reject grid lines: components spanning >80% of plate width or height
        # with very thin profile are grid artifacts, not roots
        if (bbox_w > w * 0.8 and bbox_h < h * 0.03) or \
           (bbox_h > h * 0.8 and bbox_w < w * 0.03):
            continue

        if elongation >= min_elongation or max(bbox_h, bbox_w) > max(h, w) * 0.15:
            root_id += 1
            filtered[component > 0] = 255
            labels[component > 0] = root_id

    num_roots = root_id
    return filtered, labels, num_roots


def separate_by_plant_body(root_mask, plate_image, plate_mask=None,
                           min_green_area=300, green_dilate=7, min_root_area=200):
    """Separate plants fused only through their overlapping green leaves.

    Seedlings planted in a tight row often have cotyledons/leaves that touch,
    which makes connected-component labeling merge several plants into one
    'root'. Because the leaves are green and the roots are not, removing the
    green plant-body mass before labeling separates the individual roots while
    leaving each root strand intact.

    This is a no-op when little/no green is present (e.g. dark-background images
    or plates without visible leaves), so it is safe to always apply. It does
    NOT resolve roots that physically cross out in the open — those stay merged
    and are left to the manual Split tool.

    Args:
        root_mask: Binary root mask (0/255), typically still including leaves.
        plate_image: BGR image of the plate (for green detection).
        plate_mask: Optional plate-area mask to restrict green detection.
        min_green_area: Minimum green pixel count before separation kicks in.
        green_dilate: Dilation (px) applied to the green mass so thin leaf
                      bridges between neighbours are fully cut.
        min_root_area: Leaf/noise fragments at or below this size are dropped.

    Returns:
        (root_mask, labels, num_roots) after separation.
    """
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    if plate_mask is not None:
        green = cv2.bitwise_and(green, plate_mask)

    if np.count_nonzero(green) >= min_green_area:
        if green_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (green_dilate, green_dilate))
            green = cv2.dilate(green, k)
        root_mask = cv2.bitwise_and(root_mask, cv2.bitwise_not(green))
        # Drop small leaf/noise fragments left behind by the cut.
        cleaned = remove_small_objects(root_mask > 0, max_size=min_root_area)
        root_mask = (cleaned.astype(np.uint8)) * 255

    num_labels, labels = cv2.connectedComponents(root_mask)
    return root_mask, labels, num_labels - 1


def skeletonize_roots(binary_mask):
    """Reduce root mask to 1-pixel-wide skeleton.

    Args:
        binary_mask: Binary mask (0/255) of root regions.

    Returns:
        skeleton: Binary skeleton image (True/False).
    """
    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    return skeleton


def prune_skeleton(skeleton, min_branch_length=10):
    """Remove short branches (spurs) from the skeleton.

    Args:
        skeleton: Binary skeleton (bool array).
        min_branch_length: Branches shorter than this are removed.

    Returns:
        Pruned skeleton (bool array).
    """
    # Find endpoints (pixels with only 1 neighbor)
    skel = skeleton.astype(np.uint8)

    # Iteratively remove short branches
    for _ in range(min_branch_length):
        endpoints = _find_endpoints(skel)
        if not np.any(endpoints):
            break
        # Remove endpoints (but not if they're also junction points)
        junctions = _find_junctions(skel)
        removable = endpoints & (~junctions)
        skel[removable] = 0

    return skel.astype(bool)


def _find_endpoints(skeleton):
    """Find endpoint pixels (exactly 1 neighbor) in skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    return (skeleton > 0) & (neighbor_count == 1)


def _find_junctions(skeleton):
    """Find junction pixels (3+ neighbors) in skeleton."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)
    return (skeleton > 0) & (neighbor_count >= 3)
