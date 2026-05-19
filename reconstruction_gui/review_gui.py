#!/usr/bin/env python3
"""
Mask Review GUI
===============
CustomTkinter-based review interface for batch mask inspection.
Thumbnail grid with sort/filter, preview panel, and OpenCV editor launch.

Usage:
    python review_gui.py ./output/masks --images-dir ./output/frames
"""

import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import argparse
import logging
import threading

from review_status import ReviewStatusManager, MaskStatus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Review status → border color (binary status model)
REVIEW_COLORS = {
    "reviewed":   "#22c55e",  # green
    "unreviewed": "#6b7280",  # gray
}

# Legacy aliases — kept so any out-of-tree consumer doesn't break.
# Internally, all border-color lookups now go through REVIEW_COLORS.
QUALITY_COLORS = {"": "#6b7280"}
STATUS_COLORS = REVIEW_COLORS

# Per-layer overlay tints in BGR channel deltas — applied to masked pixels.
# Order is the assignment order; index 0 → first layer (typically `base`).
# 12 distinct hues — beyond this the palette recycles. Beyond ~12, hue
# discrimination collapses for most viewers, so 12 is a deliberate cap.
LAYER_COLORS = [
    (0, 0, 100),       # red
    (100, 0, 0),       # blue
    (0, 100, 0),       # green
    (0, 100, 100),     # yellow
    (100, 100, 0),     # cyan
    (100, 0, 100),     # magenta
    (0, 60, 120),      # orange (R-heavy + bit of G)
    (100, 0, 60),      # purple (B + bit of R)
    (40, 110, 40),     # lime  (G with hint of B/R, distinct from green)
    (90, 60, 110),     # pink  (R-heavy + B/G)
    (90, 80, 30),      # teal  (B+G low, no R)
    (40, 60, 80),      # brown (low all-channel mix)
]

THUMB_SIZE = 150


def load_overlay_thumbnail(image_path: Path, mask_path: Path, size: int = THUMB_SIZE,
                           pad_to_square: bool = True) -> Optional[Image.Image]:
    """Create a thumbnail with mask overlay."""
    img = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None

    # Resize to thumbnail
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    # Red overlay on masked-out regions (black in mask = foreground to remove)
    overlay = img.copy()
    overlay[mask < 128, 2] = np.clip(overlay[mask < 128, 2].astype(int) + 100, 0, 255).astype(np.uint8)

    # Convert BGR→RGB→PIL
    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    if not pad_to_square:
        return pil_img

    # Pad to square
    result = Image.new("RGB", (size, size), (30, 30, 30))
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    result.paste(pil_img, (x_off, y_off))
    return result


def load_multi_layer_thumbnail(
    image_path: Path,
    layer_masks: List[Tuple[Path, Tuple[int, int, int]]],
    size: int = THUMB_SIZE,
    pad_to_square: bool = True,
) -> Optional[Image.Image]:
    """Composite multiple mask layers as colored overlays on a single thumbnail.

    Args:
        image_path: Source image path.
        layer_masks: List of (mask_path, (b_delta, g_delta, r_delta)) tuples.
            Each layer's BGR deltas are added to masked pixels (mask < 128).
        size: Target longest-side size for the thumbnail.
        pad_to_square: If True, pad to a square canvas.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    overlay = cv2.resize(img, (new_w, new_h))

    for mask_path, (bd, gd, rd) in layer_masks:
        if mask_path is None or not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (new_h, new_w):
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        masked = mask < 128  # disk convention: black = masked region
        # Apply BGR-channel deltas, clipped to [0, 255]
        overlay[masked, 0] = np.clip(overlay[masked, 0].astype(int) + bd, 0, 255).astype(np.uint8)
        overlay[masked, 1] = np.clip(overlay[masked, 1].astype(int) + gd, 0, 255).astype(np.uint8)
        overlay[masked, 2] = np.clip(overlay[masked, 2].astype(int) + rd, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    if not pad_to_square:
        return pil_img

    result = Image.new("RGB", (size, size), (30, 30, 30))
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    result.paste(pil_img, (x_off, y_off))
    return result


def compute_mask_area_percent(mask_path: Path) -> float:
    """Compute mask coverage as a percentage of total image area."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0.0
    total = mask.shape[0] * mask.shape[1]
    return (np.sum(mask < 128) / total * 100) if total > 0 else 0.0


class ThumbnailWidget(ctk.CTkFrame):
    """A single thumbnail tile in the grid."""

    def __init__(self, parent, stem: str, thumb_image: Image.Image, status: MaskStatus,
                 on_click=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.stem = stem
        self.on_click = on_click

        # Border color by binary review status
        border_color = REVIEW_COLORS.get(status.status, "#6b7280")
        self.configure(border_width=3, border_color=border_color, corner_radius=6)

        # Thumbnail image
        self._ctk_image = ctk.CTkImage(light_image=thumb_image, size=(THUMB_SIZE, THUMB_SIZE))
        self.image_label = ctk.CTkLabel(self, image=self._ctk_image, text="")
        self.image_label.pack(padx=2, pady=(2, 0))

        # Info line
        conf_str = f"{status.confidence:.0%}" if status.confidence > 0 else "?"
        info_text = f"{stem[:20]}  {conf_str}"
        self.info_label = ctk.CTkLabel(self, text=info_text, font=("Consolas", 11),
                                       text_color="#d1d5db")
        self.info_label.pack(padx=2, pady=(0, 2))

        # Click binding
        for widget in (self, self.image_label, self.info_label):
            widget.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        if self.on_click:
            self.on_click(self.stem)


class ReviewApp(ctk.CTk):
    """Main review application window."""

    def __init__(self, masks_dir: Path, images_dir: Path):
        super().__init__()
        self.title("Mask Review")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.masks_dir = Path(masks_dir)
        self.images_dir = Path(images_dir)
        self.status_mgr = ReviewStatusManager(self.masks_dir)

        # Discover mask/image pairs
        self.pairs: List[Dict] = []  # [{stem, mask_path, image_path}, ...]
        self._discover_pairs()

        # Currently selected
        self.selected_stem: Optional[str] = None

        # Build UI
        self._build_ui()
        self._populate_grid()

    # ---- Discovery ----

    def _discover_pairs(self):
        """Find all mask/image pairs.

        Strategy 1: masks named ``mask_*.png`` / ``*_mask.png`` (pipeline output).
        Strategy 2: any image in masks_dir whose stem matches a file in images_dir
                     (handles the common case where masks share the same filename).
        """
        # Strategy 1: prefixed mask names
        mask_files = sorted(
            list(self.masks_dir.glob("mask_*.png")) +
            list(self.masks_dir.glob("*_mask.png"))
        )
        for mf in mask_files:
            stem = mf.stem.replace("mask_", "").replace("_mask", "")
            image_path = self._find_image(stem)
            if image_path:
                self.pairs.append({
                    "stem": stem,
                    "mask_path": mf,
                    "image_path": image_path,
                })

        # Strategy 2: same-stem matching (masks and images share filenames)
        if not self.pairs and self.masks_dir != self.images_dir:
            image_stems = {}
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
                for f in self.images_dir.glob(ext):
                    image_stems[f.stem] = f

            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for mf in sorted(self.masks_dir.glob(ext)):
                    if mf.stem in image_stems and mf.stem not in {p["stem"] for p in self.pairs}:
                        self.pairs.append({
                            "stem": mf.stem,
                            "mask_path": mf,
                            "image_path": image_stems[mf.stem],
                        })

        # Populate quality metadata
        for pair in self.pairs:
            ms = self.status_mgr.get(pair["stem"])
            if ms.area_percent == 0.0:
                ms.area_percent = compute_mask_area_percent(pair["mask_path"])

        self.status_mgr.save()
        logger.info(f"Found {len(self.pairs)} mask/image pairs")

    def _find_image(self, stem: str) -> Optional[Path]:
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    # ---- UI Construction ----

    def _build_ui(self):
        # Main horizontal split: left (grid), right (preview)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # -- Top bar: sort/filter --
        top_frame = ctk.CTkFrame(self, height=40)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))

        ctk.CTkLabel(top_frame, text="Sort:").pack(side="left", padx=(10, 2))
        self.sort_var = ctk.StringVar(value="Filename")
        sort_menu = ctk.CTkOptionMenu(
            top_frame, variable=self.sort_var,
            values=["Filename", "Confidence", "Quality", "Area %", "Modified"],
            command=self._on_sort_change, width=120
        )
        sort_menu.pack(side="left", padx=2)

        ctk.CTkLabel(top_frame, text="Filter:").pack(side="left", padx=(20, 2))
        self.filter_var = ctk.StringVar(value="All")
        filter_seg = ctk.CTkSegmentedButton(
            top_frame, values=["All", "Unreviewed", "Reviewed"],
            variable=self.filter_var, command=self._on_filter_change
        )
        filter_seg.pack(side="left", padx=2)

        # Summary label
        self.summary_label = ctk.CTkLabel(top_frame, text="", font=("Consolas", 11))
        self.summary_label.pack(side="right", padx=10)

        # -- Left: scrollable thumbnail grid --
        self.grid_frame = ctk.CTkScrollableFrame(self, label_text="Masks")
        self.grid_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        # Configure 4 columns in grid
        for c in range(4):
            self.grid_frame.columnconfigure(c, weight=1)

        # -- Right: preview + actions --
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Preview area (3 views side by side)
        preview_container = ctk.CTkFrame(right_frame)
        preview_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        preview_container.grid_columnconfigure(0, weight=1)
        preview_container.grid_columnconfigure(1, weight=1)
        preview_container.grid_columnconfigure(2, weight=1)
        preview_container.grid_rowconfigure(0, weight=1)

        self.preview_labels = []
        view_titles = ["Original", "Overlay", "Mask"]
        for i, title in enumerate(view_titles):
            frame = ctk.CTkFrame(preview_container)
            frame.grid(row=0, column=i, sticky="nsew", padx=2, pady=2)
            ctk.CTkLabel(frame, text=title, font=("Consolas", 11)).pack(pady=(2, 0))
            lbl = ctk.CTkLabel(frame, text="")
            lbl.pack(expand=True, fill="both", padx=2, pady=2)
            self.preview_labels.append(lbl)

        # Info panel
        info_frame = ctk.CTkFrame(right_frame)
        info_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)

        self.info_text = ctk.CTkLabel(info_frame, text="Select a mask", font=("Consolas", 12),
                                      anchor="w", justify="left")
        self.info_text.pack(fill="x", padx=10, pady=5)

        # Action buttons
        btn_frame = ctk.CTkFrame(right_frame)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.edit_btn = ctk.CTkButton(btn_frame, text="Edit (OpenCV)", command=self._on_edit,
                                      fg_color="#2563eb", width=120)
        self.edit_btn.pack(side="left", padx=5, pady=5)

        self.save_btn = ctk.CTkButton(btn_frame, text="Save", command=self._on_save,
                                      fg_color="#16a34a", width=90)
        self.save_btn.pack(side="left", padx=5, pady=5)

        self.delete_btn = ctk.CTkButton(btn_frame, text="Delete", command=self._on_delete,
                                        fg_color="#dc2626", width=90)
        self.delete_btn.pack(side="left", padx=5, pady=5)

        self.skip_btn = ctk.CTkButton(btn_frame, text="Skip", command=self._on_skip,
                                      fg_color="#6b7280", width=90)
        self.skip_btn.pack(side="left", padx=5, pady=5)

        self.propagate_btn = ctk.CTkButton(btn_frame, text="Propagate Edit",
                                           command=self._on_propagate,
                                           fg_color="#7c3aed", width=120)
        self.propagate_btn.pack(side="left", padx=5, pady=5)

    # ---- Grid Population ----

    def _get_sorted_filtered_pairs(self) -> List[Dict]:
        """Apply current sort and filter to pairs."""
        pairs = list(self.pairs)

        # Filter
        filt = self.filter_var.get()
        if filt == "Unreviewed":
            pairs = [p for p in pairs if self.status_mgr.get(p["stem"]).status == "unreviewed"]
        elif filt == "Reviewed":
            pairs = [p for p in pairs if self.status_mgr.get(p["stem"]).status == "reviewed"]

        # Sort
        sort_key = self.sort_var.get()
        if sort_key == "Confidence":
            pairs.sort(key=lambda p: self.status_mgr.get(p["stem"]).confidence)
        elif sort_key == "Quality":
            order = {"reject": 0, "poor": 1, "review": 2, "good": 3, "excellent": 4, "": -1}
            pairs.sort(key=lambda p: order.get(self.status_mgr.get(p["stem"]).quality, -1))
        elif sort_key == "Area %":
            pairs.sort(key=lambda p: self.status_mgr.get(p["stem"]).area_percent, reverse=True)
        elif sort_key == "Modified":
            order = {True: 0, False: 1, None: 2}
            pairs.sort(key=lambda p: order.get(self.status_mgr.get(p["stem"]).last_edit_modified, 2))
        else:  # Filename
            pairs.sort(key=lambda p: p["stem"])

        return pairs

    def _populate_grid(self):
        """Rebuild the thumbnail grid."""
        # Clear existing widgets
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        pairs = self._get_sorted_filtered_pairs()

        for idx, pair in enumerate(pairs):
            stem = pair["stem"]
            ms = self.status_mgr.get(stem)
            thumb = load_overlay_thumbnail(pair["image_path"], pair["mask_path"])
            if thumb is None:
                continue

            tw = ThumbnailWidget(
                self.grid_frame, stem, thumb, ms,
                on_click=self._on_thumbnail_click
            )
            row, col = divmod(idx, 4)
            tw.grid(row=row, column=col, padx=4, pady=4, sticky="n")

        # Update summary
        summary = self.status_mgr.get_summary()
        total = len(self.pairs)
        reviewed = summary.get("reviewed", 0)
        unreviewed = summary.get("unreviewed", 0)
        self.summary_label.configure(
            text=f"{total} masks | {reviewed} reviewed | {unreviewed} unreviewed"
        )

    def _on_sort_change(self, _=None):
        self._populate_grid()

    def _on_filter_change(self, _=None):
        self._populate_grid()

    # ---- Selection & Preview ----

    def _on_thumbnail_click(self, stem: str):
        self.selected_stem = stem
        self._update_preview()

    def _update_preview(self):
        if not self.selected_stem:
            return

        pair = next((p for p in self.pairs if p["stem"] == self.selected_stem), None)
        if not pair:
            return

        ms = self.status_mgr.get(self.selected_stem)

        # Load images for preview
        img = cv2.imread(str(pair["image_path"]))
        mask = cv2.imread(str(pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return

        # Compute preview size (fit in panel)
        preview_w = 180
        h, w = img.shape[:2]
        scale = preview_w / max(h, w)
        pw, ph = int(w * scale), int(h * scale)

        img_small = cv2.resize(img, (pw, ph))
        mask_small = cv2.resize(mask, (pw, ph))

        # Original
        orig_pil = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        # Overlay
        overlay = img_small.copy()
        overlay[mask_small < 128, 2] = np.clip(
            overlay[mask_small < 128, 2].astype(int) + 100, 0, 255
        ).astype(np.uint8)
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        # Mask only
        mask_rgb = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2RGB)
        mask_pil = Image.fromarray(mask_rgb)

        images = [orig_pil, overlay_pil, mask_pil]
        for i, pil_img in enumerate(images):
            ctk_img = ctk.CTkImage(light_image=pil_img, size=(pw, ph))
            self.preview_labels[i].configure(image=ctk_img, text="")
            self.preview_labels[i]._ctk_image = ctk_img  # prevent GC

        # Info text
        mod_str = ""
        if ms.last_edit_modified is True:
            mod_str = "  |  edited"
        elif ms.last_edit_modified is False:
            mod_str = "  |  unchanged"
        self.info_text.configure(
            text=f"File: {self.selected_stem}\n"
                 f"{ms.status}  |  Confidence: {ms.confidence:.0%}  |  "
                 f"Area: {ms.area_percent:.1f}%{mod_str}"
        )

    # ---- Actions ----

    def _on_edit(self):
        if not self.selected_stem:
            return
        pair = next((p for p in self.pairs if p["stem"] == self.selected_stem), None)
        if not pair:
            return

        # Launch OpenCV editor in a thread so CTk doesn't freeze
        def _run_editor():
            try:
                from review_masks import MaskReviewer
            except ImportError:
                # Try relative import
                import sys
                import os
                sys.path.insert(0, os.path.dirname(__file__))
                from review_masks import MaskReviewer

            reviewer = MaskReviewer()
            result = reviewer.review_mask(
                pair["image_path"], pair["mask_path"],
                window_name=f"Edit: {self.selected_stem}"
            )
            if result == "save":
                self.status_mgr.record_action(self.selected_stem, "reviewed", modified=True)
            # Refresh UI on main thread
            self.after(0, self._refresh_after_edit)

        thread = threading.Thread(target=_run_editor, daemon=True)
        thread.start()

    def _refresh_after_edit(self):
        """Refresh thumbnail and preview after editing."""
        self._populate_grid()
        self._update_preview()

    def _on_save(self):
        """Mark the current mask as reviewed without opening the editor."""
        if self.selected_stem:
            self.status_mgr.record_action(self.selected_stem, "reviewed", modified=False)
            self._advance_to_next()

    def _on_delete(self):
        """Delete the current mask from disk and remove from the review set."""
        if not self.selected_stem:
            return
        pair = next((p for p in self.pairs if p["stem"] == self.selected_stem), None)
        if pair:
            mask_path = Path(pair["mask_path"])
            if mask_path.exists():
                mask_path.unlink()
                logger.info(f"Deleted: {mask_path.name}")
        self.status_mgr.record_action(self.selected_stem, "deleted")
        self.pairs = [p for p in self.pairs if p["stem"] != self.selected_stem]
        self._advance_to_next()

    def _on_skip(self):
        """Advance to next without changing review status."""
        if self.selected_stem:
            self._advance_to_next()

    def _on_propagate(self):
        """Propagate the current mask edit to similar frames.

        Computes delta between original and edited mask, finds other pending
        masks, and applies the same delta (added/removed pixels) using
        bounding-box overlap as a similarity heuristic.
        """
        if not self.selected_stem:
            return

        pair = next((p for p in self.pairs if p["stem"] == self.selected_stem), None)
        if not pair:
            return

        ms = self.status_mgr.get(self.selected_stem)
        if not ms.last_edit_modified:
            logger.info("Propagate requires a mask edited in the OpenCV editor as source")
            return

        # Load current (edited) mask
        edited_mask = cv2.imread(str(pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if edited_mask is None:
            return

        # Find bounding box of the mask (region of interest)
        mask_binary = (edited_mask > 127).astype(np.uint8)
        coords = cv2.findNonZero(mask_binary)
        if coords is None:
            return
        bx, by, bw, bh = cv2.boundingRect(coords)

        # Apply to other pending masks that have overlapping mask regions
        applied_count = 0
        for other_pair in self.pairs:
            if other_pair["stem"] == self.selected_stem:
                continue
            other_ms = self.status_mgr.get(other_pair["stem"])
            if other_ms.status != "unreviewed":
                continue

            other_mask = cv2.imread(str(other_pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
            if other_mask is None or other_mask.shape != edited_mask.shape:
                continue

            # Check overlap: does the other mask have content in the same region?
            other_binary = (other_mask > 127).astype(np.uint8)
            other_coords = cv2.findNonZero(other_binary)
            if other_coords is None:
                continue
            ox, oy, ow, oh = cv2.boundingRect(other_coords)

            # Compute bounding box IoU
            ix1 = max(bx, ox)
            iy1 = max(by, oy)
            ix2 = min(bx + bw, ox + ow)
            iy2 = min(by + bh, oy + oh)
            if ix2 <= ix1 or iy2 <= iy1:
                continue  # No overlap
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            union_area = bw * bh + ow * oh - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou < 0.3:
                continue  # Not similar enough

            # Apply: replace the overlapping region with our edited version
            cv2.imwrite(str(other_pair["mask_path"]), edited_mask)
            self.status_mgr.record_action(other_pair["stem"], "reviewed", modified=True)
            applied_count += 1

        logger.info(f"Propagated edit to {applied_count} masks")
        self._populate_grid()

    def _advance_to_next(self):
        """Move selection to next unreviewed mask and refresh."""
        current_pairs = self._get_sorted_filtered_pairs()
        current_idx = next(
            (i for i, p in enumerate(current_pairs) if p["stem"] == self.selected_stem), -1
        )
        # Find next unreviewed
        for i in range(current_idx + 1, len(current_pairs)):
            ms = self.status_mgr.get(current_pairs[i]["stem"])
            if ms.status == "unreviewed":
                self.selected_stem = current_pairs[i]["stem"]
                self._populate_grid()
                self._update_preview()
                return
        # No more unreviewed — just refresh
        self._populate_grid()
        self._update_preview()


def main():
    parser = argparse.ArgumentParser(description="Mask Review GUI")
    parser.add_argument("masks_dir", type=Path, help="Directory containing masks")
    parser.add_argument("--images-dir", type=Path, help="Directory containing original images")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)

    # Auto-detect images dir
    if args.images_dir:
        images_dir = args.images_dir
    else:
        for candidate in [masks_dir.parent / "frames", masks_dir.parent.parent / "frames",
                          masks_dir.parent / "images"]:
            if candidate.exists():
                images_dir = candidate
                break
        else:
            logger.error("Could not find images directory. Use --images-dir.")
            return

    # Handle nested masks/ folder
    if (masks_dir / "masks").exists():
        masks_dir = masks_dir / "masks"

    app = ReviewApp(masks_dir, images_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
