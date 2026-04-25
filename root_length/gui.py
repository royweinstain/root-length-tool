"""Tkinter GUI for reviewing and correcting root length measurements."""

import csv
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .pipeline import process_image, create_debug_overlay
from .measure import find_mark_positions_on_skeleton, measure_segments


class ToolTip:
    """Hover tooltip for tkinter widgets."""

    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", foreground="#333",
                         relief=tk.SOLID, borderwidth=1,
                         font=("TkDefaultFont", 9), padx=6, pady=4)
        label.pack()

    def _hide(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class RootLengthGUI:
    """Main GUI window for reviewing root length pipeline results."""

    # Radius (in display pixels) for click-to-add/remove marks
    CLICK_RADIUS = 12
    # Radius drawn on the mark mask when adding a mark
    MARK_DRAW_RADIUS = 5

    def __init__(self, root):
        self.root = root
        self.root.title("Root Length Measurement")

        # State
        self.image_path = None
        self.plate_results = []
        self.current_plate_idx = 0
        self.display_scale = 1.0  # scale from plate image coords to canvas coords
        self.photo_image = None  # keep reference to prevent GC
        self.undo_stack = []  # list of (plate_idx, snapshot) tuples
        self.MAX_UNDO = 20
        self.split_mode = False
        self.split_start = None  # first click point for split
        self.merge_mode = False
        self.merge_first_root = None  # first root ID clicked for merge
        self.extend_mode = False
        self.extend_root_id = None  # which root is being extended
        self.extend_points = []  # list of (y, x) points clicked so far
        self.add_root_mode = False
        self.add_root_points = []  # list of (y, x) points for new root

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Row 1: File, navigation, detection settings
        row1 = ttk.Frame(self.root)
        row1.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))

        btn = ttk.Button(row1, text="Open Image", command=self._open_image)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Open an image file and run the\nauto-detection pipeline.")

        self.prev_btn = ttk.Button(row1, text="< Prev Plate", command=self._prev_plate, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2)

        self.plate_label = ttk.Label(row1, text="No image loaded")
        self.plate_label.pack(side=tk.LEFT, padx=8)

        self.next_btn = ttk.Button(row1, text="Next Plate >", command=self._next_plate, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(row1, text="Mark color:").pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="auto")
        color_combo = ttk.Combobox(row1, textvariable=self.color_var,
                                   values=["auto", "red", "blue", "green"],
                                   state="readonly", width=8)
        color_combo.pack(side=tk.LEFT, padx=2)
        ToolTip(color_combo, "Select mark color to detect.\n'auto' tries to pick the best color.")

        btn = ttk.Button(row1, text="Re-detect", command=self._redetect_marks)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Re-run mark detection with the\nselected color. Resets all corrections.")

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(row1, text="Scale (px/mm):").pack(side=tk.LEFT)
        self.scale_var = tk.StringVar(value="auto")
        self.scale_entry = ttk.Entry(row1, textvariable=self.scale_var, width=8)
        self.scale_entry.pack(side=tk.LEFT, padx=2)
        btn = ttk.Button(row1, text="Apply Scale", command=self._apply_scale)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Apply a manual px/mm scale to all\nplates and recalculate measurements.")

        # Row 2: Editing tools and actions
        row2 = ttk.Frame(self.root)
        row2.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))

        self.split_btn = ttk.Button(row2, text="Split Root", command=self._toggle_split_mode)
        self.split_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(self.split_btn, "Split one root into two.\nClick two points to draw a cut line.")

        self.extend_btn = ttk.Button(row2, text="Extend Root", command=self._toggle_extend_mode)
        self.extend_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(self.extend_btn, "Extend an existing root along a faint tip.\nClick the root, then click points along the\nextension. Right-click or Enter to finish.")

        self.merge_btn = ttk.Button(row2, text="Merge Roots", command=self._toggle_merge_mode)
        self.merge_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(self.merge_btn, "Merge two roots into one.\nClick on the first root, then the second.\nGap must be under 50 px.")

        self.add_root_btn = ttk.Button(row2, text="Add Root", command=self._toggle_add_root_mode)
        self.add_root_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(self.add_root_btn, "Manually trace a root that was not detected.\nClick points along the root path.\nRight-click or Enter to finish.")

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        btn = ttk.Button(row2, text="Save Session", command=self._save_session)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Save all corrections to a session file.\nAuto-loaded when reopening the same image.")

        btn = ttk.Button(row2, text="Export CSV", command=self._export_csv)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Export measurements for all plates\nto a CSV file.")

        btn = ttk.Button(row2, text="Restart", command=self._restart_fresh)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Discard all corrections and re-run\nthe pipeline from scratch.")

        btn = ttk.Button(row2, text="\u2618", command=self._show_motivation, width=3)
        btn.pack(side=tk.LEFT, padx=2)
        ToolTip(btn, "Words of wisdom for the\nweary root measurer.")

        # Main area: canvas on left, measurements on right
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Canvas with scrollbars
        canvas_frame = ttk.Frame(main_frame)
        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("<Control-z>", self._undo)
        main_frame.add(canvas_frame, weight=3)

        # Right panel: measurements table
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)

        ttk.Label(right_frame, text="Measurements", font=("", 11, "bold")).pack(anchor=tk.W, padx=4, pady=(4, 0))

        # Instructions
        self.instructions = ttk.Label(right_frame, text="Left-click: add mark\nRight-click: remove nearest mark",
                                      foreground="gray")
        self.instructions.pack(anchor=tk.W, padx=4, pady=(0, 4))

        # Treeview for measurement data
        tree_frame = ttk.Frame(right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ("root", "marks", "total_mm", "segments")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=20)
        self.tree.heading("root", text="Root")
        self.tree.heading("marks", text="Marks")
        self.tree.heading("total_mm", text="Total (mm)")
        self.tree.heading("segments", text="Segments (mm)")
        self.tree.column("root", width=50, anchor=tk.CENTER)
        self.tree.column("marks", width=50, anchor=tk.CENTER)
        self.tree.column("total_mm", width=80, anchor=tk.CENTER)
        self.tree.column("segments", width=200, anchor=tk.W)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------------------------------------------------- Image loading
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        self.undo_stack = []
        self.current_plate_idx = 0

        # Try loading a saved session first
        self.status_var.set(f"Loading {os.path.basename(path)}...")
        self.root.update_idletasks()

        if self._load_session():
            self.status_var.set(f"Loaded saved session for {os.path.basename(path)}")
        else:
            # No session — run the pipeline
            self.status_var.set(f"Processing {os.path.basename(path)}...")
            self.root.update_idletasks()

            color = self.color_var.get()
            mark_color = None if color == "auto" else color

            try:
                self.plate_results = process_image(path, mark_color=mark_color, processing_size=1200)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image:\n{e}")
                self.status_var.set("Error")
                return

        # Populate scale field from ruler detection
        ppmm = self.plate_results[0].get("pixels_per_mm")
        if ppmm:
            self.scale_var.set(f"{ppmm:.1f}")
        else:
            self.scale_var.set("auto")

        self._show_current_plate()

    # -------------------------------------------------------- Plate display
    def _show_current_plate(self):
        if not self.plate_results:
            return

        pr = self.plate_results[self.current_plate_idx]
        n_plates = len(self.plate_results)
        color = pr["mark_color_used"] or "none"

        self.plate_label.config(
            text=f"Plate {self.current_plate_idx + 1} / {n_plates}  "
                 f"[{color}]  roots={pr['num_roots_detected']}"
        )

        # Navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_plate_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_plate_idx < n_plates - 1 else tk.DISABLED)

        # Draw overlay on canvas
        overlay = create_debug_overlay(pr)
        self._display_image(overlay)

        # Update measurements table
        self._update_measurements()

        ppmm = pr["pixels_per_mm"]
        unit = "mm" if ppmm else "px"
        self.status_var.set(
            f"{os.path.basename(self.image_path)} — Plate {self.current_plate_idx + 1}/{n_plates} — "
            f"Scale: {ppmm:.1f} px/mm — Unit: {unit}" if ppmm else
            f"{os.path.basename(self.image_path)} — Plate {self.current_plate_idx + 1}/{n_plates} — "
            f"No ruler detected (measurements in pixels)"
        )

    def _display_image(self, bgr_image):
        """Show a BGR OpenCV image on the canvas, scaled to fit."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Scale to fit canvas (leave some margin)
        self.root.update_idletasks()
        canvas_w = max(self.canvas.winfo_width(), 600)
        canvas_h = max(self.canvas.winfo_height(), 600)
        img_w, img_h = pil_img.size

        self.display_scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        if self.display_scale < 1.0:
            new_w = int(img_w * self.display_scale)
            new_h = int(img_h * self.display_scale)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=(0, 0, pil_img.width, pil_img.height))

    def _update_measurements(self):
        """Refresh the measurements treeview from the current plate result."""
        self.tree.delete(*self.tree.get_children())
        pr = self.plate_results[self.current_plate_idx]
        ppmm = pr.get("pixels_per_mm")
        unit = "mm" if ppmm else "px"

        # Update column headers with current unit
        self.tree.heading("total_mm", text=f"Total ({unit})")
        self.tree.heading("segments", text=f"Segments ({unit})")

        for rid in sorted(pr["roots"].keys()):
            data = pr["roots"][rid]
            marks = data["mark_count"]
            total = f"{data['total_length_mm']:.1f}"
            segs = ", ".join(f"{s:.1f}" for s in data["segments"])
            self.tree.insert("", tk.END, values=(
                rid, marks, total, segs
            ))

    # -------------------------------------------------------- Navigation
    def _prev_plate(self):
        if self.current_plate_idx > 0:
            self.current_plate_idx -= 1
            self._show_current_plate()

    def _next_plate(self):
        if self.current_plate_idx < len(self.plate_results) - 1:
            self.current_plate_idx += 1
            self._show_current_plate()

    # ------------------------------------------------ Scale
    def _apply_scale(self):
        """Apply a manual px/mm scale to all plates and re-measure."""
        if not self.plate_results:
            return

        val = self.scale_var.get().strip()
        if val.lower() == "auto" or val == "":
            messagebox.showinfo("Scale", "Enter a numeric px/mm value (e.g. 22.0).")
            return

        try:
            ppmm = float(val)
            if ppmm <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Scale", "Invalid scale. Enter a positive number (e.g. 22.0).")
            return

        # Update all plate results with the new scale and re-measure
        for pr in self.plate_results:
            pr["pixels_per_mm"] = ppmm

            adj_ppmm = ppmm * pr["processing_scale"]

            mark_positions = find_mark_positions_on_skeleton(
                pr["skeleton"], pr["mark_mask"], pr["labels"], pr["num_roots_detected"]
            )
            measurements = measure_segments(
                mark_positions, pr["skeleton"], pr["labels"],
                pixels_per_mm=adj_ppmm, include_tip=False
            )
            pr["roots"] = measurements

        self.status_var.set(f"Applied scale: {ppmm:.1f} px/mm")
        self._show_current_plate()

    # ------------------------------------------------ Undo
    def _save_snapshot(self):
        """Save the current plate state to the undo stack."""
        if not self.plate_results:
            return
        pr = self.plate_results[self.current_plate_idx]
        snapshot = {
            "root_mask": pr["root_mask"].copy(),
            "labels": pr["labels"].copy(),
            "skeleton": pr["skeleton"].copy(),
            "mark_mask": pr["mark_mask"].copy(),
            "num_roots_detected": pr["num_roots_detected"],
            "roots": {k: dict(v) for k, v in pr["roots"].items()},
        }
        self.undo_stack.append((self.current_plate_idx, snapshot))
        if len(self.undo_stack) > self.MAX_UNDO:
            self.undo_stack.pop(0)

    def _undo(self, event=None):
        """Restore the previous state (Ctrl+Z)."""
        if not self.undo_stack:
            self.status_var.set("Nothing to undo")
            return

        plate_idx, snapshot = self.undo_stack.pop()
        pr = self.plate_results[plate_idx]
        pr["root_mask"] = snapshot["root_mask"]
        pr["labels"] = snapshot["labels"]
        pr["skeleton"] = snapshot["skeleton"]
        pr["mark_mask"] = snapshot["mark_mask"]
        pr["num_roots_detected"] = snapshot["num_roots_detected"]
        pr["roots"] = snapshot["roots"]

        self.current_plate_idx = plate_idx
        self.status_var.set("Undone")
        self._show_current_plate()

    # ------------------------------------------------ Mark editing
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas click coordinates to plate image pixel coordinates."""
        img_x = int(canvas_x / self.display_scale)
        img_y = int(canvas_y / self.display_scale)
        return img_y, img_x  # return as (row, col)

    def _on_left_click(self, event):
        """Left-click: add mark (normal mode) or set split points (split mode)."""
        if not self.plate_results:
            return

        y, x = self._canvas_to_image_coords(event.x, event.y)
        pr = self.plate_results[self.current_plate_idx]
        h, w = pr["mark_mask"].shape

        if y < 0 or y >= h or x < 0 or x >= w:
            return

        if self.split_mode:
            self._handle_split_click(y, x)
        elif self.merge_mode:
            self._handle_merge_click(y, x)
        elif self.extend_mode:
            self._handle_extend_click(y, x)
        elif self.add_root_mode:
            self._handle_add_root_click(y, x)
        else:
            self._add_mark(y, x)

    def _on_right_click(self, event):
        """Right-click: remove mark (normal mode) or finish extend (extend mode)."""
        if not self.plate_results:
            return

        if self.extend_mode and self.extend_root_id is not None:
            self._finish_extend()
            return

        if self.add_root_mode and len(self.add_root_points) >= 2:
            self._finish_add_root()
            return

        y, x = self._canvas_to_image_coords(event.x, event.y)
        pr = self.plate_results[self.current_plate_idx]
        h, w = pr["mark_mask"].shape

        if y < 0 or y >= h or x < 0 or x >= w:
            return

        self._remove_mark(y, x)

    def _add_mark(self, y, x):
        """Add a mark at the given image coordinates and re-measure."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        mark_mask = pr["mark_mask"]
        cv2.circle(mark_mask, (x, y), self.MARK_DRAW_RADIUS, 255, -1)
        self.status_var.set(f"Added mark at ({x}, {y})")
        self._remeasure()

    def _remove_mark(self, y, x):
        """Remove the mark nearest to (y, x) and re-measure."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        mark_mask = pr["mark_mask"]

        # Find the connected component at this location and erase it
        r = self.CLICK_RADIUS
        h, w = mark_mask.shape

        # Flood-fill approach: find contours and erase the one containing/nearest to click
        contours, _ = cv2.findContours(mark_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_cnt = None
        best_dist = float("inf")

        for cnt in contours:
            # Check distance from click to contour centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.sqrt((cy - y)**2 + (cx - x)**2)
            if dist < best_dist and dist < self.CLICK_RADIUS * 2:
                best_dist = dist
                best_cnt = cnt

        if best_cnt is not None:
            cv2.drawContours(mark_mask, [best_cnt], -1, 0, -1)
            # Also clear a small buffer around it
            cv2.drawContours(mark_mask, [best_cnt], -1, 0, 3)
            self.status_var.set(f"Removed mark near ({x}, {y})")
            self._remeasure()
        else:
            self.status_var.set(f"No mark found near ({x}, {y})")

    def _remeasure(self):
        """Re-run mark position finding and measurement on the current plate."""
        pr = self.plate_results[self.current_plate_idx]

        mark_positions = find_mark_positions_on_skeleton(
            pr["skeleton"], pr["mark_mask"], pr["labels"], pr["num_roots_detected"]
        )

        adj_ppmm = pr["pixels_per_mm"]
        if adj_ppmm and pr["processing_scale"] < 1.0:
            adj_ppmm = adj_ppmm * pr["processing_scale"]

        measurements = measure_segments(
            mark_positions, pr["skeleton"], pr["labels"],
            pixels_per_mm=adj_ppmm, include_tip=False
        )
        pr["roots"] = measurements

        self._show_current_plate()

    # ------------------------------------------------ Split root
    def _toggle_split_mode(self):
        """Toggle split mode on/off."""
        # Cancel other modes
        if self.merge_mode:
            self._toggle_merge_mode()
        if self.extend_mode:
            self._toggle_extend_mode()
        if self.add_root_mode:
            self._toggle_add_root_mode()

        self.split_mode = not self.split_mode
        self.split_start = None
        if self.split_mode:
            self.split_btn.config(text="Cancel Split")
            self.canvas.config(cursor="tcross")
            self.status_var.set("SPLIT MODE: click the first point of the cut line")
        else:
            self.split_btn.config(text="Split Root")
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Split mode cancelled")

    def _handle_split_click(self, y, x):
        """Handle a click in split mode."""
        if self.split_start is None:
            # First click — store start point and show feedback
            self.split_start = (y, x)
            self.status_var.set(f"SPLIT MODE: first point set at ({x}, {y}). Click the second point to cut.")
            # Draw a small marker on the canvas
            cx = int(x * self.display_scale)
            cy = int(y * self.display_scale)
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                    fill="red", outline="white", tags="split_marker")
        else:
            # Second click — perform the split
            start = self.split_start
            end = (y, x)
            self.canvas.delete("split_marker")
            self._perform_split(start, end)
            # Exit split mode
            self.split_mode = False
            self.split_start = None
            self.split_btn.config(text="Split Root")
            self.canvas.config(cursor="crosshair")

    def _perform_split(self, start, end):
        """Cut the root mask along a line and re-process."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        root_mask = pr["root_mask"]
        mark_mask = pr["mark_mask"]
        plate_img = pr["plate_image"]
        plate_mask = pr["plate_mask"]

        # Draw a black line on the root mask to cut the component
        pt1 = (start[1], start[0])  # (x, y)
        pt2 = (end[1], end[0])
        cut_thickness = 5
        cv2.line(root_mask, pt1, pt2, 0, cut_thickness)

        # Re-label connected components
        num_labels, new_labels = cv2.connectedComponents(root_mask)
        num_roots = num_labels - 1  # exclude background (label 0)

        # Re-skeletonize from the updated root mask
        from .roots import skeletonize_roots, prune_skeleton
        skeleton = skeletonize_roots(root_mask)
        skeleton = prune_skeleton(skeleton, min_branch_length=8)

        # Re-detect marks on new skeleton
        mark_positions = find_mark_positions_on_skeleton(
            skeleton, mark_mask, new_labels, num_roots
        )

        # Re-measure
        adj_ppmm = pr["pixels_per_mm"]
        if adj_ppmm and pr["processing_scale"] < 1.0:
            adj_ppmm = adj_ppmm * pr["processing_scale"]

        measurements = measure_segments(
            mark_positions, skeleton, new_labels,
            pixels_per_mm=adj_ppmm, include_tip=False
        )

        # Update plate result
        pr["labels"] = new_labels
        pr["num_roots_detected"] = num_roots
        pr["skeleton"] = skeleton
        pr["roots"] = measurements

        self.status_var.set(f"Split complete — now {num_roots} roots detected")
        self._show_current_plate()

    # ------------------------------------------------ Merge roots
    MAX_MERGE_GAP = 50  # max pixel gap to auto-bridge

    def _toggle_merge_mode(self):
        """Toggle merge mode on/off."""
        # Cancel other modes
        if self.split_mode:
            self._toggle_split_mode()
        if self.extend_mode:
            self._toggle_extend_mode()
        if self.add_root_mode:
            self._toggle_add_root_mode()

        self.merge_mode = not self.merge_mode
        self.merge_first_root = None
        if self.merge_mode:
            self.merge_btn.config(text="Cancel Merge")
            self.canvas.config(cursor="tcross")
            self.status_var.set("MERGE MODE: click on the first root to merge")
        else:
            self.merge_btn.config(text="Merge Roots")
            self.canvas.config(cursor="crosshair")
            self.status_var.set("Merge mode cancelled")

    def _handle_merge_click(self, y, x):
        """Handle a click in merge mode — identify which root was clicked."""
        pr = self.plate_results[self.current_plate_idx]
        labels = pr["labels"]

        # Find the root ID at or near the click point
        root_id = self._find_root_at(labels, y, x, search_radius=15)
        if root_id is None:
            self.status_var.set("No root found at click location. Click on a root.")
            return

        if self.merge_first_root is None:
            self.merge_first_root = root_id
            self.status_var.set(f"MERGE MODE: selected root {root_id}. Now click on the second root.")
        else:
            if root_id == self.merge_first_root:
                self.status_var.set(f"Same root ({root_id}). Click on a different root.")
                return
            self._perform_merge(self.merge_first_root, root_id)
            # Exit merge mode
            self.merge_mode = False
            self.merge_first_root = None
            self.merge_btn.config(text="Merge Roots")
            self.canvas.config(cursor="crosshair")

    def _find_root_at(self, labels, y, x, search_radius=15):
        """Find the root ID at or near (y, x)."""
        h, w = labels.shape
        # Check the exact pixel first
        if 0 <= y < h and 0 <= x < w and labels[y, x] > 0:
            return int(labels[y, x])
        # Search in expanding radius
        for r in range(1, search_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                        return int(labels[ny, nx])
        return None

    def _perform_merge(self, root_a, root_b):
        """Merge two roots: bridge the gap and re-process."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        root_mask = pr["root_mask"]
        skeleton = pr["skeleton"]
        labels = pr["labels"]
        mark_mask = pr["mark_mask"]

        # Find closest points between the two root masks
        mask_a = (labels == root_a).astype(np.uint8)
        mask_b = (labels == root_b).astype(np.uint8)
        ys_a, xs_a = np.where(mask_a > 0)
        ys_b, xs_b = np.where(mask_b > 0)

        if len(ys_a) == 0 or len(ys_b) == 0:
            self.status_var.set("Could not find root pixels for merge")
            return

        # Find closest pair between the two masks
        pts_a = np.column_stack((ys_a, xs_a))
        pts_b = np.column_stack((ys_b, xs_b))
        # Subsample for speed if needed
        if len(pts_a) > 1000:
            pts_a = pts_a[np.random.choice(len(pts_a), 1000, replace=False)]
        if len(pts_b) > 1000:
            pts_b = pts_b[np.random.choice(len(pts_b), 1000, replace=False)]

        best_dist = float("inf")
        best_pa = best_pb = None
        for pa in pts_a:
            dists = np.sqrt((pts_b[:, 0] - pa[0])**2 + (pts_b[:, 1] - pa[1])**2)
            min_idx = np.argmin(dists)
            if dists[min_idx] < best_dist:
                best_dist = dists[min_idx]
                best_pa = pa
                best_pb = pts_b[min_idx]

        if best_dist > self.MAX_MERGE_GAP:
            self.status_var.set(
                f"Gap between roots {root_a} and {root_b} is {best_dist:.0f} px "
                f"(max allowed: {self.MAX_MERGE_GAP}). Roots too far apart to merge."
            )
            return

        # Bridge the gap: draw a line on the root mask connecting the closest points
        pt1 = (int(best_pa[1]), int(best_pa[0]))  # (x, y)
        pt2 = (int(best_pb[1]), int(best_pb[0]))
        cv2.line(root_mask, pt1, pt2, 255, 3)

        # Re-label connected components
        num_labels, new_labels = cv2.connectedComponents(root_mask)
        num_roots = num_labels - 1

        # Re-skeletonize
        from .roots import skeletonize_roots, prune_skeleton
        new_skeleton = skeletonize_roots(root_mask)
        new_skeleton = prune_skeleton(new_skeleton, min_branch_length=8)

        # Re-detect mark positions and measure
        mark_positions = find_mark_positions_on_skeleton(
            new_skeleton, mark_mask, new_labels, num_roots
        )

        adj_ppmm = pr["pixels_per_mm"]
        if adj_ppmm and pr["processing_scale"] < 1.0:
            adj_ppmm = adj_ppmm * pr["processing_scale"]

        measurements = measure_segments(
            mark_positions, new_skeleton, new_labels,
            pixels_per_mm=adj_ppmm, include_tip=False
        )

        # Update plate result
        pr["labels"] = new_labels
        pr["num_roots_detected"] = num_roots
        pr["skeleton"] = new_skeleton
        pr["roots"] = measurements

        self.status_var.set(
            f"Merged roots {root_a} and {root_b} (gap: {best_dist:.0f} px) — "
            f"now {num_roots} roots detected"
        )
        self._show_current_plate()

    # ------------------------------------------------ Extend root
    EXTEND_LINE_THICKNESS = 5  # thickness of the drawn extension on root mask

    def _toggle_extend_mode(self):
        """Toggle extend mode on/off."""
        # Cancel other modes
        if self.split_mode:
            self._toggle_split_mode()
        if self.merge_mode:
            self._toggle_merge_mode()
        if self.add_root_mode:
            self._toggle_add_root_mode()

        self.extend_mode = not self.extend_mode
        self.extend_root_id = None
        self.extend_points = []
        if self.extend_mode:
            self.extend_btn.config(text="Cancel Extend")
            self.canvas.config(cursor="tcross")
            self.status_var.set("EXTEND MODE: click on the root you want to extend")
        else:
            self.extend_btn.config(text="Extend Root")
            self.canvas.config(cursor="crosshair")
            self.canvas.delete("extend_marker")
            self.status_var.set("Extend mode cancelled")

    def _handle_extend_click(self, y, x):
        """Handle clicks in extend mode."""
        pr = self.plate_results[self.current_plate_idx]

        if self.extend_root_id is None:
            # First click: identify which root to extend
            root_id = self._find_root_at(pr["labels"], y, x, search_radius=15)
            if root_id is None:
                self.status_var.set("No root found at click location. Click on a root.")
                return
            self.extend_root_id = root_id
            self.extend_points = [(y, x)]
            # Draw marker
            cx = int(x * self.display_scale)
            cy = int(y * self.display_scale)
            self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                    fill="lime", outline="white", tags="extend_marker")
            self.status_var.set(
                f"EXTEND MODE: extending root {root_id}. "
                f"Click points along the root path. Right-click or press Enter to finish."
            )
            # Bind Enter key to finish
            self.root.bind("<Return>", self._finish_extend)
        else:
            # Subsequent clicks: add points along the extension
            self.extend_points.append((y, x))
            # Draw marker and line to previous point
            cx = int(x * self.display_scale)
            cy = int(y * self.display_scale)
            self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                    fill="lime", outline="white", tags="extend_marker")
            # Draw line from previous point
            prev_y, prev_x = self.extend_points[-2]
            pcx = int(prev_x * self.display_scale)
            pcy = int(prev_y * self.display_scale)
            self.canvas.create_line(pcx, pcy, cx, cy, fill="lime", width=2, tags="extend_marker")
            self.status_var.set(
                f"EXTEND MODE: {len(self.extend_points)} points. "
                f"Click more points or right-click/Enter to finish."
            )

    def _finish_extend(self, event=None):
        """Finish the extend operation — apply the extension."""
        self.root.unbind("<Return>")
        self.canvas.delete("extend_marker")

        if len(self.extend_points) < 2:
            self.status_var.set("Need at least 2 points to extend. Cancelled.")
            self.extend_mode = False
            self.extend_root_id = None
            self.extend_points = []
            self.extend_btn.config(text="Extend Root")
            self.canvas.config(cursor="crosshair")
            return

        self._perform_extend()

        # Exit extend mode
        self.extend_mode = False
        self.extend_root_id = None
        self.extend_points = []
        self.extend_btn.config(text="Extend Root")
        self.canvas.config(cursor="crosshair")

    def _perform_extend(self):
        """Draw the extension path onto the root mask and re-process."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        root_mask = pr["root_mask"]
        mark_mask = pr["mark_mask"]

        # Draw lines between consecutive extend points onto the root mask
        for i in range(len(self.extend_points) - 1):
            pt1 = (self.extend_points[i][1], self.extend_points[i][0])  # (x, y)
            pt2 = (self.extend_points[i + 1][1], self.extend_points[i + 1][0])
            cv2.line(root_mask, pt1, pt2, 255, self.EXTEND_LINE_THICKNESS)

        # Re-label connected components
        num_labels, new_labels = cv2.connectedComponents(root_mask)
        num_roots = num_labels - 1

        # Re-skeletonize
        from .roots import skeletonize_roots, prune_skeleton
        new_skeleton = skeletonize_roots(root_mask)
        new_skeleton = prune_skeleton(new_skeleton, min_branch_length=8)

        # Re-detect mark positions and measure
        mark_positions = find_mark_positions_on_skeleton(
            new_skeleton, mark_mask, new_labels, num_roots
        )

        adj_ppmm = pr["pixels_per_mm"]
        if adj_ppmm and pr["processing_scale"] < 1.0:
            adj_ppmm = adj_ppmm * pr["processing_scale"]

        measurements = measure_segments(
            mark_positions, new_skeleton, new_labels,
            pixels_per_mm=adj_ppmm, include_tip=False
        )

        # Update plate result
        pr["labels"] = new_labels
        pr["num_roots_detected"] = num_roots
        pr["skeleton"] = new_skeleton
        pr["roots"] = measurements

        self.status_var.set(
            f"Extended root {self.extend_root_id} with {len(self.extend_points)} points — "
            f"now {num_roots} roots detected"
        )
        self._show_current_plate()

    # ------------------------------------------------ Add root
    ADD_ROOT_LINE_THICKNESS = 5

    def _toggle_add_root_mode(self):
        """Toggle add-root mode on/off."""
        # Cancel other modes
        if self.split_mode:
            self._toggle_split_mode()
        if self.merge_mode:
            self._toggle_merge_mode()
        if self.extend_mode:
            self._toggle_extend_mode()

        self.add_root_mode = not self.add_root_mode
        self.add_root_points = []
        if self.add_root_mode:
            self.add_root_btn.config(text="Cancel Add Root")
            self.canvas.config(cursor="tcross")
            self.status_var.set("ADD ROOT: click points along the root path. Right-click or Enter to finish.")
            self.root.bind("<Return>", self._finish_add_root)
        else:
            self.add_root_btn.config(text="Add Root")
            self.canvas.config(cursor="crosshair")
            self.canvas.delete("add_root_marker")
            self.root.unbind("<Return>")
            self.status_var.set("Add root cancelled")

    def _handle_add_root_click(self, y, x):
        """Handle clicks in add-root mode — collect points along the root."""
        self.add_root_points.append((y, x))

        # Draw marker
        cx = int(x * self.display_scale)
        cy = int(y * self.display_scale)
        self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3,
                                fill="yellow", outline="white", tags="add_root_marker")

        # Draw line from previous point
        if len(self.add_root_points) >= 2:
            prev_y, prev_x = self.add_root_points[-2]
            pcx = int(prev_x * self.display_scale)
            pcy = int(prev_y * self.display_scale)
            self.canvas.create_line(pcx, pcy, cx, cy, fill="yellow", width=2,
                                    tags="add_root_marker")

        self.status_var.set(
            f"ADD ROOT: {len(self.add_root_points)} points. "
            f"Click more points or right-click/Enter to finish."
        )

    def _finish_add_root(self, event=None):
        """Finish add-root — draw path onto root mask as a new root."""
        self.root.unbind("<Return>")
        self.canvas.delete("add_root_marker")

        if len(self.add_root_points) < 2:
            self.status_var.set("Need at least 2 points to add a root. Cancelled.")
            self.add_root_mode = False
            self.add_root_points = []
            self.add_root_btn.config(text="Add Root")
            self.canvas.config(cursor="crosshair")
            return

        self._perform_add_root()

        # Exit add-root mode
        self.add_root_mode = False
        self.add_root_points = []
        self.add_root_btn.config(text="Add Root")
        self.canvas.config(cursor="crosshair")

    def _perform_add_root(self):
        """Draw the new root path onto the root mask and re-process."""
        self._save_snapshot()
        pr = self.plate_results[self.current_plate_idx]
        root_mask = pr["root_mask"]
        mark_mask = pr["mark_mask"]

        # Draw lines between consecutive points onto the root mask
        for i in range(len(self.add_root_points) - 1):
            pt1 = (self.add_root_points[i][1], self.add_root_points[i][0])  # (x, y)
            pt2 = (self.add_root_points[i + 1][1], self.add_root_points[i + 1][0])
            cv2.line(root_mask, pt1, pt2, 255, self.ADD_ROOT_LINE_THICKNESS)

        # Re-label connected components
        num_labels, new_labels = cv2.connectedComponents(root_mask)
        num_roots = num_labels - 1

        # Re-skeletonize
        from .roots import skeletonize_roots, prune_skeleton
        new_skeleton = skeletonize_roots(root_mask)
        new_skeleton = prune_skeleton(new_skeleton, min_branch_length=8)

        # Re-detect mark positions and measure
        mark_positions = find_mark_positions_on_skeleton(
            new_skeleton, mark_mask, new_labels, num_roots
        )

        adj_ppmm = pr["pixels_per_mm"]
        if adj_ppmm and pr["processing_scale"] < 1.0:
            adj_ppmm = adj_ppmm * pr["processing_scale"]

        measurements = measure_segments(
            mark_positions, new_skeleton, new_labels,
            pixels_per_mm=adj_ppmm, include_tip=False
        )

        # Update plate result
        pr["labels"] = new_labels
        pr["num_roots_detected"] = num_roots
        pr["skeleton"] = new_skeleton
        pr["roots"] = measurements

        self.status_var.set(
            f"Added new root with {len(self.add_root_points)} points — "
            f"now {num_roots} roots detected"
        )
        self._show_current_plate()

    # ------------------------------------------------ Motivation
    _MOTIVATIONS = [
        # Plant puns
        "I'm rooting for you!",
        "You really know how to get to the root of things.",
        "This work is un-be-leaf-able.",
        "Alright, let's get to the root of the problem.",
        "You're doing a grape job. Wait, wrong plant.",
        "This is riveting. Absolutely riveting. Another root.",
        "I think we've really branched out today.",
        "Petal to the metal!",
        "You're a budding scientist!",
        "Lettuce continue.",
        "Time to turnip the productivity.",
        "You had me at 'skeletonize'.",
        "Phloem me maybe?",
        "What did the root say to the soil? I dig you.",
        # Lab encouragement
        "Only a few hundred more roots to go!",
        "Your PI would be so proud right now.",
        "This root is particularly beautiful. Just look at it.",
        "Remember: every root you measure is a root someone doesn't have to measure by hand.",
        "Fun fact: you've been staring at roots longer than most roots have been alive.",
        "Somewhere, a plant grew this root just for you to measure it.",
        "Have you considered that the roots are also measuring you?",
        "If each root takes 30 seconds, you'll be done by... let's not do that math.",
        "Pro tip: the roots don't get shorter if you squint.",
        "You're basically a root whisperer at this point.",
        "The skeleton is looking good. The root skeleton, I mean.",
        "Plot twist: the faint root was inside you all along.",
        "Another day, another dataset. You've got this.",
        "Science isn't about 'why'. It's about 'how many roots'.",
        "This plate sparks joy.",
    ]

    def _show_motivation(self):
        msg = random.choice(self._MOTIVATIONS)
        self.status_var.set(msg)

    # ----------------------------------------------- Re-detect marks
    def _redetect_marks(self):
        """Re-run mark detection with the currently selected color."""
        if not self.plate_results:
            return

        color = self.color_var.get()
        mark_color = None if color == "auto" else color

        self.status_var.set("Re-detecting marks...")
        self.root.update_idletasks()

        # Re-process the full image with the new color
        try:
            self.plate_results = process_image(
                self.image_path, mark_color=mark_color, processing_size=1200
            )
        except Exception as e:
            messagebox.showerror("Error", f"Re-detection failed:\n{e}")
            return

        self.current_plate_idx = min(self.current_plate_idx, len(self.plate_results) - 1)
        self._show_current_plate()

    # -------------------------------------------------------- Session save/load
    def _session_path(self):
        """Return the session file path for the current image."""
        if not self.image_path:
            return None
        base = os.path.splitext(self.image_path)[0]
        return base + "_session.pkl"

    def _save_session(self):
        """Save all plate results to a session file."""
        if not self.plate_results:
            messagebox.showinfo("Save", "No data to save.")
            return

        import pickle

        session_data = []
        for pr in self.plate_results:
            # Save only the data needed to restore state (not the plate image)
            plate_save = {
                "plate_index": pr["plate_index"],
                "plate_center": pr["plate_center"],
                "plate_radius": pr["plate_radius"],
                "processing_scale": pr["processing_scale"],
                "polarity": pr["polarity"],
                "mark_color_used": pr["mark_color_used"],
                "pixels_per_mm": pr["pixels_per_mm"],
                "num_roots_detected": pr["num_roots_detected"],
                "root_mask": pr["root_mask"],
                "labels": pr["labels"],
                "skeleton": pr["skeleton"],
                "mark_mask": pr["mark_mask"],
                "roots": pr["roots"],
            }
            session_data.append(plate_save)

        path = self._session_path()
        try:
            with open(path, "wb") as f:
                pickle.dump(session_data, f)
            self.status_var.set(f"Session saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session:\n{e}")

    def _load_session(self):
        """Try to load a saved session for the current image.

        Returns True if a session was loaded, False otherwise.
        """
        import pickle

        path = self._session_path()
        if not path or not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                session_data = pickle.load(f)
        except Exception:
            return False

        # Re-read the image to get plate images (not stored in session)
        image = cv2.imread(self.image_path)
        if image is None:
            return False

        from .plates import crop_plate

        restored = []
        for saved in session_data:
            cx, cy = saved["plate_center"]
            radius = saved["plate_radius"]
            plate_img_full, plate_mask_full = crop_plate(image, cx, cy, radius)

            # Resize to processing size (same scale as when saved)
            proc_scale = saved["processing_scale"]
            if proc_scale < 1.0:
                plate_img = cv2.resize(plate_img_full, None, fx=proc_scale, fy=proc_scale)
                plate_mask = cv2.resize(plate_mask_full, None, fx=proc_scale, fy=proc_scale,
                                        interpolation=cv2.INTER_NEAREST)
            else:
                plate_img = plate_img_full
                plate_mask = plate_mask_full

            pr = {
                "plate_index": saved["plate_index"],
                "plate_center": saved["plate_center"],
                "plate_radius": saved["plate_radius"],
                "processing_scale": saved["processing_scale"],
                "polarity": saved["polarity"],
                "mark_color_used": saved["mark_color_used"],
                "pixels_per_mm": saved["pixels_per_mm"],
                "num_roots_detected": saved["num_roots_detected"],
                "root_mask": saved["root_mask"],
                "labels": saved["labels"],
                "skeleton": saved["skeleton"],
                "mark_mask": saved["mark_mask"],
                "roots": saved["roots"],
                "plate_image": plate_img,
                "plate_mask": plate_mask,
            }
            restored.append(pr)

        self.plate_results = restored
        return True

    def _restart_fresh(self):
        """Delete saved session and re-run the pipeline from scratch."""
        if not self.image_path:
            return

        answer = messagebox.askyesno(
            "Restart",
            "This will discard all corrections and re-process the image from scratch.\n\nContinue?"
        )
        if not answer:
            return

        # Delete session file if it exists
        path = self._session_path()
        if path and os.path.exists(path):
            os.remove(path)

        # Re-run pipeline
        self.undo_stack = []
        self.status_var.set(f"Re-processing {os.path.basename(self.image_path)}...")
        self.root.update_idletasks()

        color = self.color_var.get()
        mark_color = None if color == "auto" else color

        try:
            self.plate_results = process_image(self.image_path, mark_color=mark_color, processing_size=1200)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
            self.status_var.set("Error")
            return

        self.current_plate_idx = 0
        ppmm = self.plate_results[0].get("pixels_per_mm")
        if ppmm:
            self.scale_var.set(f"{ppmm:.1f}")
        else:
            self.scale_var.set("auto")

        self._show_current_plate()
        self.status_var.set("Restarted fresh — all corrections cleared")

    # -------------------------------------------------------- CSV export
    def _export_csv(self):
        if not self.plate_results:
            messagebox.showinfo("Export", "No data to export.")
            return

        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_measurements.csv"
        path = filedialog.asksaveasfilename(
            title="Export measurements",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        # Find max number of segments across all plates/roots
        max_segs = 0
        for pr in self.plate_results:
            for data in pr["roots"].values():
                max_segs = max(max_segs, len(data["segments"]))

        ppmm = self.plate_results[0].get("pixels_per_mm")
        unit = "mm" if ppmm else "px"

        header = ["image", "plate", "root", "marks",
                  f"total_{unit}"]
        for i in range(max_segs):
            header.append(f"seg_{i+1}_{unit}")

        rows = []
        img_name = os.path.basename(self.image_path)
        for pr in self.plate_results:
            plate_idx = pr["plate_index"] + 1
            for rid in sorted(pr["roots"].keys()):
                data = pr["roots"][rid]
                row = [img_name, plate_idx, rid,
                       data["mark_count"], data["total_length_mm"]]
                for s in data["segments"]:
                    row.append(s)
                # Pad with empty cells if fewer segments
                while len(row) < len(header):
                    row.append("")
                rows.append(row)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        self.status_var.set(f"Exported {len(rows)} rows to {os.path.basename(path)}")


def main():
    root = tk.Tk()
    root.geometry("1400x900")
    app = RootLengthGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
