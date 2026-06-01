"""Main processing pipeline — ties all modules together."""

import cv2
import numpy as np

from .plates import detect_plates, crop_plate
from .ruler import detect_ruler
from .roots import (segment_roots, skeletonize_roots, prune_skeleton,
                    separate_by_plant_body)
from .marks import detect_marks, auto_detect_mark_color
from .measure import (find_mark_positions_on_skeleton, measure_segments,
                       base_endpoint, assign_marks_to_roots)
from .utils import detect_polarity, detect_growth_direction


def process_image(image_path, mark_color=None, pixels_per_mm=None,
                   processing_size=1200, growth_direction=None):
    """Process a full image: detect plates, roots, marks, and measure.

    Args:
        image_path: Path to the image file.
        mark_color: Color of marks ("red", "blue", "green").
                    If None, auto-detect.
        pixels_per_mm: Manual scale override. If None, auto-detect from ruler.
        processing_size: Max dimension for each cropped plate during processing.
                         Larger = more accurate but slower.

    Returns:
        List of plate results.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Detect image polarity (light or dark background)
    polarity = detect_polarity(image)

    # Step 1: Detect ruler for scale (before cropping plates)
    if pixels_per_mm is None:
        pixels_per_mm = detect_ruler(image, polarity=polarity)
        if pixels_per_mm is None:
            print(f"WARNING: Could not detect ruler in {image_path}. "
                  "Measurements will be in pixels.")

    # Step 2: Detect plates
    plates = detect_plates(image)
    if not plates:
        print(f"WARNING: No plates detected in {image_path}. "
              "Treating entire image as a single plate.")
        h, w = image.shape[:2]
        plates = [(w // 2, h // 2, min(w, h) // 2)]

    # Step 3: Process each plate
    results = []
    for idx, (cx, cy, r) in enumerate(plates):
        plate_result = process_single_plate(
            image, cx, cy, r, idx,
            mark_color=mark_color,
            pixels_per_mm=pixels_per_mm,
            processing_size=processing_size,
            polarity=polarity,
            growth_direction=growth_direction,
        )
        results.append(plate_result)

    return results


def process_single_plate(image, cx, cy, radius, plate_index,
                         mark_color=None, pixels_per_mm=None,
                         processing_size=1200, polarity="light",
                         growth_direction=None):
    """Process a single plate from the image.

    growth_direction: "up"/"down"/"left"/"right" (plant body -> tip). If None,
                      auto-detected from green plant bodies / plate geometry.
    """

    # Crop plate at full resolution
    plate_img_full, plate_mask_full = crop_plate(image, cx, cy, radius)

    # Resize for processing (skeletonization is very slow on large images)
    full_h, full_w = plate_img_full.shape[:2]
    proc_scale = min(1.0, processing_size / max(full_h, full_w))

    if proc_scale < 1.0:
        plate_img = cv2.resize(plate_img_full, None, fx=proc_scale, fy=proc_scale)
        plate_mask = cv2.resize(plate_mask_full, None, fx=proc_scale, fy=proc_scale,
                                interpolation=cv2.INTER_NEAREST)
        adj_ppmm = pixels_per_mm * proc_scale if pixels_per_mm else None
    else:
        plate_img = plate_img_full
        plate_mask = plate_mask_full
        adj_ppmm = pixels_per_mm

    # Segment roots (polarity-aware) — this mask still includes green leaves.
    root_mask, labels, num_roots = segment_roots(
        plate_img, plate_mask, polarity=polarity
    )

    # Determine growth direction (plant body -> tip) so marks are ordered and
    # tips found correctly regardless of plate orientation. Done BEFORE the
    # green plant bodies are removed below, since the green is the key cue.
    if growth_direction is None:
        growth_direction = detect_growth_direction(plate_img, plate_mask, root_mask)

    # Separate plants fused only through overlapping leaves (no-op without green).
    root_mask, labels, num_roots = separate_by_plant_body(
        root_mask, plate_img, plate_mask
    )

    # Skeletonize
    skeleton = skeletonize_roots(root_mask)
    skeleton = prune_skeleton(skeleton, min_branch_length=8)

    # Detect marks
    if mark_color is None:
        mark_color = auto_detect_mark_color(plate_img, plate_mask)

    if mark_color is not None:
        mark_mask = detect_marks(plate_img, color=mark_color, plate_mask=plate_mask,
                                  root_mask=root_mask)
    else:
        mark_mask = np.zeros(plate_img.shape[:2], dtype=np.uint8)

    # Find mark positions on skeleton
    mark_positions = find_mark_positions_on_skeleton(
        skeleton, mark_mask, labels, num_roots, direction=growth_direction
    )

    # Measure segments (include root tip as final endpoint)
    measurements = measure_segments(
        mark_positions, skeleton, labels, pixels_per_mm=adj_ppmm,
        include_tip=False, direction=growth_direction
    )

    return {
        "plate_index": plate_index,
        "plate_center": (cx, cy),
        "plate_radius": radius,
        "processing_scale": proc_scale,
        "polarity": polarity,
        "growth_direction": growth_direction,
        "roots": measurements,
        "mark_color_used": mark_color,
        "pixels_per_mm": pixels_per_mm,
        "num_roots_detected": num_roots,
        "plate_image": plate_img,
        "plate_mask": plate_mask,
        "skeleton": skeleton,
        "mark_mask": mark_mask,
        "root_mask": root_mask,
        "labels": labels,
    }


def create_debug_overlay(plate_result, highlight_root_id=None):
    """Create a visual overlay showing detected roots, skeleton, and marks.

    Args:
        plate_result: Result dict from process_single_plate.
        highlight_root_id: If set, that root is emphasized (bright halo + bold
                           skeleton + boxed number) so it can be picked out when
                           splitting/merging. Ignored if it isn't a current root.

    Returns:
        BGR image with colored overlays.
    """
    img = plate_result["plate_image"].copy()
    skeleton = plate_result["skeleton"]
    mark_mask = plate_result["mark_mask"]
    root_mask = plate_result["root_mask"]

    # Draw each root in a distinct color
    labels = plate_result["labels"]
    num_roots = plate_result["num_roots_detected"]
    # Distinct colors (BGR) that are easy to tell apart
    root_colors = [
        (180, 100, 0),    # teal
        (0, 180, 0),      # green
        (0, 100, 200),    # orange
        (200, 0, 100),    # purple
        (0, 200, 200),    # yellow
        (200, 150, 0),    # cyan
        (50, 50, 200),    # red-ish
        (100, 200, 50),   # lime
        (200, 50, 150),   # pink
        (50, 200, 150),   # sea green
        (150, 50, 200),   # magenta
        (100, 150, 200),  # sand
    ]
    root_overlay = np.zeros_like(img)
    for root_id in range(1, num_roots + 1):
        color = root_colors[(root_id - 1) % len(root_colors)]
        root_overlay[labels == root_id] = color
    img = cv2.addWeighted(img, 1.0, root_overlay, 0.35, 0)

    # Draw skeleton in white for visibility on colored roots
    img[skeleton > 0] = [255, 255, 255]

    # Emphasize a highlighted root so it is easy to identify on a busy plate.
    # Drawn before the marks so the marks stay visible on top of the halo.
    highlight_present = (
        highlight_root_id is not None and np.any(labels == highlight_root_id)
    )
    if highlight_present:
        hl_mask = (labels == highlight_root_id).astype(np.uint8) * 255
        # Bright halo ring around the root body
        halo = cv2.dilate(hl_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        ring = cv2.subtract(halo, hl_mask)
        img[ring > 0] = [0, 255, 255]  # yellow
        # Re-draw this root's skeleton thicker and bright yellow
        hl_skel = (skeleton & (labels == highlight_root_id)).astype(np.uint8)
        hl_skel = cv2.dilate(hl_skel, np.ones((3, 3), np.uint8))
        img[hl_skel > 0] = [0, 255, 255]

    # Draw marks, colored by their root association so it is visible which mark
    # belongs where: orange = unassigned (too far / problem), cyan = belongs to
    # the highlighted root, magenta = assigned to some other root.
    mark_labels, assignments = assign_marks_to_roots(skeleton, mark_mask, labels)
    for a in assignments:
        comp = mark_labels == a["mark_label"]
        if a["root_id"] <= 0:
            img[comp] = [0, 140, 255]      # orange — unassigned
        elif highlight_present and a["root_id"] == highlight_root_id:
            img[comp] = [255, 255, 0]      # cyan — highlighted root's mark
        else:
            img[comp] = [255, 0, 255]      # magenta — assigned elsewhere

    # Draw root ID numbers near the plant-body end of each root's skeleton
    labels = plate_result["labels"]
    num_roots = plate_result["num_roots_detected"]
    direction = plate_result.get("growth_direction", "down")
    for root_id in range(1, num_roots + 1):
        root_skel = skeleton & (labels == root_id)
        ys, xs = np.where(root_skel)
        if len(ys) == 0:
            continue
        # Place label at the plant-body end of the root (where measurement starts)
        anchor = base_endpoint(root_skel, direction)
        ly, lx = anchor if anchor is not None else (int(ys.min()), int(xs[np.argmin(ys)]))
        if highlight_present and root_id == highlight_root_id:
            # Boxed, larger label for the highlighted root
            text = str(root_id)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            ox, oy = lx + 5, ly - 5
            cv2.rectangle(img, (ox - 3, oy - th - 3), (ox + tw + 3, oy + 3),
                          (0, 0, 0), -1)
            cv2.putText(img, text, (ox, oy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(img, str(root_id), (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, str(root_id), (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    return img
