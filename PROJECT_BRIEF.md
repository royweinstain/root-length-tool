# Root Length Measurement Tool — Project Brief & Implementation Plan

## Context

The lab manually measures primary root elongation in plant seedlings grown on petri dishes. Every 24 hours, colored pen marks are drawn across the roots. After the experiment (3-7 days), plates are photographed and the arc-length between consecutive marks along each root needs to be measured. This is currently done manually. Existing tools are too brittle for real lab conditions.

The goal is a **robust, semi-automated local tool** that handles real-world lab images with minimal user correction.

## Image conditions (agreed)

- **Multiple plates per image** supported (auto-detected and cropped)
- **Both image polarities**: light background (dark roots) and dark background (light roots)
- **Colored marks** (red, blue, or green) preferred for future images
- **Ruler included** in every image for real-world scale (mm/cm ruler, any edge, any polarity)
- Lighting may vary; roots may occasionally overlap; mark color may differ between experiments
- Roots may grow vertically or horizontally depending on plate orientation
- If students don't mark the root tip, measurement extends to the skeleton endpoint automatically
- No API calls — everything runs locally

## Architecture

```
[Load image] → Detect polarity (light/dark) → Plate detection (Hough circles)
→ Crop each plate → Ruler detection (px/mm, both polarities)
→ Root segmentation (polarity-aware thresholding) → Skeletonization
→ Mark detection (HSV color) → Mark-skeleton intersection
→ Arc-length measurement (with optional root tip) → GUI review/correction → CSV export
```

## Current status (2026-03-31)

### Phase 1: Core pipeline — COMPLETE

| Component | Status | Notes |
|---|---|---|
| Polarity detection | DONE | Auto-detects light vs dark background |
| Plate detection (circular) | DONE | Hough circles + rim validation. Works on all test images, both polarities |
| Plate detection (square) | PARTIAL | Code ready, fails on low-contrast color images. Will need GUI manual selection fallback |
| Ruler detection | DONE | search_fraction=0.25, works on full and cropped images (~22-23 px/mm light imgs, ~72.6 px/mm dark img) |
| Root segmentation | DONE | Adaptive thresh (light), percentile thresh (dark). Filters rim, text, grid lines, non-root shapes |
| Skeletonization | DONE | Fast at 1200px processing size, prunes short branches |
| Root tip detection | DONE | Adds skeleton endpoint as final measurement point if no mark at tip |
| Elongation filter | DONE | Works for both vertical and horizontal root growth |
| Grid-line rejection | DONE | Rejects components spanning >80% of plate dimension with <3% thickness |
| Mark detection (colored) | DONE | Red and blue verified on real images (37/38 and 28/39 detected). Green under-detects — not used. Aspect ratio filter removed — marks can be any orientation |
| Mark detection (black) | NOT VIABLE | Can't reliably separate black marks from roots/leaves/text |
| Arc-length measurement | DONE | BFS path tracing between marks, outputs mm via ruler scale |
| Mark ordering | DONE | BFS distance from plant body (topmost endpoint) ensures correct top-to-bottom order on branching skeletons |
| Mark-skeleton matching | DONE | 15x15 dilation kernel bridges marks up to ~7px from skeleton |

#### Known limitations
- **Colored/teal backgrounds**: The light/dark polarity model doesn't cover colored backgrounds (e.g. teal agar in older color images). Root segmentation fails.
- **Square plates on dark backgrounds**: Auto-detection fails when plate-to-background contrast is very low.
- **Green marks**: Under-detects compared to red and blue. Not recommended for use.
- **Root merging**: Some roots may be merged into a single skeleton — use Split Root tool in GUI to separate.
- **Faint root tips**: Very low-contrast root tips (near plate edge) may not be segmented — use Extend Root tool in GUI to trace manually.

### Phase 2: GUI for review and correction (tkinter) — FUNCTIONAL

Implemented (2026-03-31):
- Open image, auto-process pipeline, display plate overlay (roots/skeleton/marks)
- Navigate between plates (Prev/Next)
- Left-click to add marks, right-click to remove marks, measurements auto-update
- Mark color selector (auto/red/blue/green) + Re-detect button
- Manual scale entry (px/mm) + Apply Scale button
- Measurements table (root, marks, total, segments) with dynamic units
- Export to CSV
- Status bar with scale/unit info
- **Split Root tool**: click two points to draw a cut line, splits one root component into two
- **Merge Roots tool**: click on two roots, auto-bridges the gap (up to 50px) and combines into one root
- **Extend Root tool**: click on a root then click points along a faint continuation, right-click/Enter to apply
- **Distinct root colors**: each root drawn in a unique color on the overlay so boundaries are clear
- Root ID numbers displayed on the overlay, matching the measurements table
- **Undo (Ctrl+Z)**: reverts last action (add/remove mark, split, merge, extend), up to 20 steps
- **Save/Load Session**: corrections saved to `_session.pkl` file, auto-loaded when reopening the same image
- **Restart**: deletes session file and re-runs pipeline from scratch

User is generating new test images for further validation (as of 2026-03-31).

Still needed:
- Plate metadata entry (plate ID, experiment label)

### Phase 3: Batch processing and CSV export — PARTIAL

- CSV export works for single image (all plates)
- Batch folder processing not yet implemented

### Phase 4 (future): Enhancements

- Improved root tracing for heavily overlapping roots
- Support for additional mark colors or multi-color marks per plate
- Automated experiment metadata extraction from plate labels

## Project structure

```
Root length/
├── data/                    # Sample images (7 images: originals + cropped per-plate)
├── output/                  # Debug overlays from test runs
├── root_length/
│   ├── __init__.py
│   ├── pipeline.py          # Main processing pipeline (orchestrates all modules)
│   ├── plates.py            # Plate detection (Hough circles) and cropping
│   ├── ruler.py             # Ruler detection and px/mm scale calibration
│   ├── roots.py             # Root segmentation (polarity-aware) and skeletonization
│   ├── marks.py             # Mark detection (HSV color-based)
│   ├── measure.py           # Arc-length measurement, root tip detection
│   ├── gui.py               # Tkinter GUI for review and correction
│   └── utils.py             # Polarity detection utility
├── run_gui.py               # GUI launcher
├── test_pipeline.py         # Test script for full pipeline
├── requirements.txt         # opencv-python, scikit-image, numpy, Pillow
└── PROJECT_BRIEF.md         # This file
```

## Dependencies

- `opencv-python`, `scikit-image`, `numpy`, `Pillow`, `tkinter` (built-in)

## Design decisions for robustness

- **Auto polarity detection** — handles both light and dark background images
- **HSV color space** for mark detection (lighting-invariant)
- **Polarity-aware thresholding** — adaptive for light bg, percentile-based for dark bg
- **Elongation filtering** — rejects text, noise; works for vertical and horizontal roots
- **Root tip fallback** — measures to skeleton endpoint if last mark is missing
- **Semi-automated** — auto-detect + human correction for edge cases
- **Modular** — each step independent, testable, replaceable

## Test images

- `jpeg20260206_16031186.jpg` — 12 round plates, light background, black marks, horizontal ruler at top
- `jpeg20260206_16130171.jpg` — 5 round plates, light background, black marks, horizontal ruler at top
- `23112025.tif` — 10 round plates, dark background, light roots, ruler on left side (14639x19800)
- `GA3C1-3 REG COL STABILITY 061213.tif` — 5 square plates, teal/dark background, colored pen marks (red/blue/green), NO ruler, 4393x6676. Older color image — exposes limitations of the polarity model on colored backgrounds.
- `jpeg20260331_11261205.tif` — 3 round plates (one per color: blue/green/red), light background, colored marks, ruler at top (9900x7319). Primary test image for colored mark validation.
- `jpeg20260331_11261205 red.tif` — Cropped single plate (red marks) from above, ruler at top (3668x7942)
- `jpeg20260331_11261205 blue.tif` — Cropped single plate (blue marks) from above, ruler at top (3977x8601)

## What's needed next

1. Address GUI issues from user testing (in progress)
2. Batch folder processing
3. Split/merge roots in GUI
4. Plate metadata entry

## Open design question: Normalization vs. dual-path

**Option A — Normalize to light polarity first:**
- Pro: One code path for root segmentation (adaptive threshold only), simpler to maintain
- Pro: Could handle colored backgrounds by converting to grayscale + inverting
- Con: Inversion distorts color info — mark detection must happen BEFORE normalization
- Con: The inversion itself may not produce ideal contrast for all backgrounds
- Con: Adds a preprocessing step that could introduce artifacts
- Con: Current dual-path already works for the two main polarities

**Option B — Keep dual-path (current approach):**
- Pro: Already working and tested on 4 images across 2 polarities
- Pro: Preserves original color info throughout pipeline
- Pro: Each path is tuned for its polarity
- Con: Colored backgrounds (teal) don't fit either path — would need a third path or GUI fallback
- Con: Two code paths to maintain and test

**Recommendation**: Keep dual-path for the two main cases (light/dark) since they work. Handle colored backgrounds as edge cases via the GUI (manual threshold adjustment). This avoids introducing normalization artifacts and keeps the working pipeline intact.
