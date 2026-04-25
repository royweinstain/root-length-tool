# Root Length Measurement Tool

A semi-automated desktop tool for measuring primary root elongation in plant seedlings grown on petri dishes. The pipeline automatically detects plates, the ruler, roots, and colored pen marks drawn at 24-hour intervals, then measures the arc-length between consecutive marks along each root. A built-in GUI lets you review and correct the results before exporting to CSV.

Everything runs locally — no internet connection or API keys required.

---

## Image requirements

For the automatic pipeline to work well, your photographs should follow these rules. The closer your images are to these guidelines, the less manual correction you will need.

### Required

- **A ruler must be visible in the image.** A standard mm/cm ruler placed along any edge of the photo. The tool uses it to convert pixel measurements to millimeters. Without a ruler, results will be in pixels only (you can also enter a scale manually in the GUI).
- **Mark the root tip on the final day.** If you do not mark the tip, the tool will fall back to the skeleton endpoint, which can be noisy at the very end of the root. A clear mark at the tip gives the most accurate final measurement.
- **Each plate must be fully visible** in the photo. Multiple plates per image is supported and will be auto-detected.

### Strongly recommended

- **Use red pen marks.** Red is the most reliably detected color. Blue works but is less robust. **Green is not recommended** — the tool consistently under-detects green marks.
- **Draw marks fully across the root**, perpendicular to the direction of growth. A short tick that does not cross the root may be missed.
- **Even, diffuse lighting.** Avoid harsh glare or strong shadows on the agar.
- **Plain backgrounds.** White or black backgrounds work best. Colored agar (e.g. teal) is not currently supported by automatic root segmentation.
- **Round (petri) plates** are auto-detected reliably. **Square plates** on dark backgrounds may need manual selection.
- Photograph plates from directly above so they appear circular, not elliptical.

### Image format

- TIFF (`.tif`) or JPEG (`.jpg`) input
- Both polarities are supported automatically: light background with dark roots, or dark background with light roots
- Roots may grow vertically or horizontally — the tool detects the orientation per plate

---

## Installation

### Windows (easiest)

1. Install [Python 3.10+](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.
2. Download or clone this repository.
3. Double-click **`Root Length Tool.bat`**. The first run installs dependencies automatically; subsequent runs just launch the GUI.

### Manual install (any OS)

```bash
git clone https://github.com/royweinstain/root-length-tool.git
cd root-length-tool
pip install -r requirements.txt
python run_gui.py
```

Dependencies: `opencv-python`, `scikit-image`, `numpy`, `Pillow`. Tkinter is bundled with Python.

---

## How to use

### 1. Open an image

Click **Open Image** and select a `.tif` or `.jpg`. The pipeline runs automatically and shows the first detected plate.

### 2. Review the overlay

Each plate is shown with:
- **Roots** drawn in distinct colors, each with a numeric ID matching the measurements table
- **Marks** drawn as small circles along the roots
- The detected ruler scale (px/mm) shown in the status bar

Use **< Prev Plate** and **Next Plate >** to step through all plates in the image.

### 3. Correct the results

The GUI provides several tools for fixing errors:

| Tool | What it does |
|---|---|
| **Left-click** | Add a mark at the cursor (snaps to the nearest skeleton point) |
| **Right-click** | Remove the nearest mark |
| **Mark color → Re-detect** | Force red/blue/green detection if auto-mode picked the wrong color |
| **Split Root** | Click two points to draw a cut line — splits one root component into two |
| **Merge Roots** | Click on two roots — auto-bridges the gap (up to ~50 px) and combines them |
| **Extend Root** | Click on a root, then click points along a faint continuation. Press **Enter** or right-click to apply |
| **Add Root** | Manually trace a root that was missed by auto-detection. Press **Enter** to apply |
| **Apply Scale** | Override the auto-detected px/mm scale |
| **Ctrl+Z** | Undo the last action (up to 20 steps) |

### 4. Save your work

- **Save Session** stores your corrections to a `_session.pkl` file next to the image. The next time you open that image, your corrections load automatically.
- **Restart** deletes the session file and re-runs the pipeline from scratch.
- **Export CSV** writes the measurements table to a CSV file you can open in Excel or any analysis tool.

The CSV contains, per root: total length, individual segment lengths, and the number of marks. Units are millimeters when a ruler scale is available, otherwise pixels.

---

## Known limitations

- **Colored agar** (e.g. teal): the polarity model assumes light or dark background. Colored backgrounds break automatic root segmentation — use the manual **Add Root** tool as a workaround.
- **Square plates on dark backgrounds**: low contrast may prevent auto-detection. Round plates are reliable in both polarities.
- **Green marks**: under-detected. Use red or blue.
- **Heavily overlapping roots**: may be merged into one skeleton — use **Split Root** to separate.
- **Faint root tips** near the plate edge: may not be segmented automatically — use **Extend Root** to trace them manually.

---

## Project structure

```
root-length-tool/
├── root_length/
│   ├── pipeline.py    # Orchestrates the full processing pipeline
│   ├── plates.py      # Plate detection (Hough circles) and cropping
│   ├── ruler.py       # Ruler detection and px/mm calibration
│   ├── roots.py       # Root segmentation and skeletonization
│   ├── marks.py       # HSV-based mark detection
│   ├── measure.py     # Arc-length measurement, root-tip detection
│   ├── gui.py         # Tkinter GUI for review and correction
│   └── utils.py       # Polarity detection
├── run_gui.py         # GUI launcher
├── test_pipeline.py   # Headless pipeline test
├── requirements.txt
├── Root Length Tool.bat   # One-click Windows launcher
└── PROJECT_BRIEF.md   # Detailed design notes
```

See `PROJECT_BRIEF.md` for the full architecture and design rationale.
