"""Test script to run the pipeline on sample images and save debug overlays."""

import cv2
import os
import sys
import numpy as np

from root_length.pipeline import process_image, create_debug_overlay
from root_length.plates import detect_plates
from root_length.ruler import detect_ruler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def test_plate_detection(image_path):
    """Test plate detection and save annotated image."""
    image = cv2.imread(image_path)
    plates = detect_plates(image)
    print(f"  Plates detected: {len(plates)}")

    annotated = image.copy()
    for i, (cx, cy, r) in enumerate(plates):
        cv2.circle(annotated, (cx, cy), r, (0, 255, 0), 3)
        cv2.putText(annotated, str(i), (cx - 10, cy - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        print(f"    Plate {i}: center=({cx},{cy}), radius={r}")

    return annotated, plates


def test_ruler_detection(image_path):
    """Test ruler detection."""
    image = cv2.imread(image_path)
    px_per_mm = detect_ruler(image)
    if px_per_mm:
        print(f"  Ruler: {px_per_mm:.1f} pixels/mm")
    else:
        print("  Ruler: NOT DETECTED")
    return px_per_mm


def test_full_pipeline(image_path, mark_color="black"):
    """Run the full pipeline and save debug overlays."""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\nProcessing: {os.path.basename(image_path)}")
    print("-" * 50)

    # Test individual steps
    print("Step 1: Ruler detection")
    px_per_mm = test_ruler_detection(image_path)

    print("Step 2: Plate detection")
    annotated, plates = test_plate_detection(image_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{basename}_plates.jpg"), annotated)
    print(f"  Saved: {basename}_plates.jpg")

    print("Step 3: Full pipeline")
    results = process_image(image_path, mark_color=mark_color)

    for plate_result in results:
        idx = plate_result["plate_index"]
        n_roots = plate_result["num_roots_detected"]
        color = plate_result["mark_color_used"]
        print(f"  Plate {idx}: {n_roots} roots detected, marks={color}")

        # Save debug overlay
        overlay = create_debug_overlay(plate_result)
        overlay_path = os.path.join(OUTPUT_DIR, f"{basename}_plate{idx}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)
        print(f"    Saved: {os.path.basename(overlay_path)}")

        # Print measurements
        for root_id, data in plate_result["roots"].items():
            unit = "mm" if px_per_mm else "px"
            segs = data["segments"]
            total = data["total_length_mm"]
            print(f"    Root {root_id}: {data['mark_count']} marks, "
                  f"segments={segs} {unit}, total={total} {unit}")

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("No images found in data/")
        return

    print(f"Found {len(images)} images in data/\n")

    for img_name in sorted(images):
        img_path = os.path.join(DATA_DIR, img_name)
        try:
            test_full_pipeline(img_path, mark_color="black")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
