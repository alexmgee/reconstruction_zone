#!/usr/bin/env python3
"""
Batch Mask Review Tool v2
=========================
Loops through images in the review folder and allows interactive editing.
Saves corrected masks back to the masks folder.

Features:
- Brush painting (add/remove)
- Click-to-segment (flood fill for connected regions)
- Help overlay with all controls
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

if sys.stderr is not None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
logger = logging.getLogger(__name__)


class MaskReviewer:
    def __init__(self):
        self.current_image = None
        self.current_mask = None
        self.original_mask = None
        self.history = []
        self.drawing = False
        self.brush_size = 20
        self.mode = 1  # 1: add, 0: remove
        self.show_mask = True
        self.show_help = True  # Initially visible, press 'h' to toggle
        self.view_mode = 0  # 0=overlay, 1=original, 2=mask-only
        self.ab_flicker = False  # A/B comparison mode
        self.ab_frame_counter = 0  # frame counter for flicker timing
        self.flood_tolerance = 15  # Color tolerance for flood fill (lower = more precise)
        self.brightness_mode = False  # Toggle with 'b' - uses brightness only (good for shadows)
        self.mouse_x = 0
        self.mouse_y = 0
        # Zoom and pan
        self.zoom_level = 1.0
        self.pan_x = 0  # Pan offset in original image coordinates
        self.pan_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        # Lasso selection
        self.lasso_points = []  # Points being drawn for lasso (stored in image coords)
        self.drawing_lasso = False
        # Window position persistence (prevents shift on save/next)
        self._last_window_pos = None

    def _display_to_image(self, dx, dy):
        """Convert display coordinates to image coordinates (unclamped).

        cv2.circle/fillPoly handle out-of-bounds centers natively,
        so no clamping needed for drawing operations.
        """
        ix = int(self.pan_x + dx / self.zoom_level)
        iy = int(self.pan_y + dy / self.zoom_level)
        return ix, iy

    def _display_to_image_clamped(self, dx, dy):
        """Convert display coordinates to image coordinates (clamped to bounds).

        Use for operations that index into image arrays (flood fill, etc.).
        """
        h, w = self.current_image.shape[:2]
        ix, iy = self._display_to_image(dx, dy)
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def _image_to_display(self, ix, iy):
        """Convert image coordinates to display coordinates."""
        dx = int((ix - self.pan_x) * self.zoom_level)
        dy = int((iy - self.pan_y) * self.zoom_level)
        return dx, dy

    def _clamp_pan(self):
        """Clamp pan values to valid image bounds."""
        if self.zoom_level <= 1.0:
            self.pan_x = 0.0
            self.pan_y = 0.0
            return
        h, w = self.current_image.shape[:2]
        max_pan_x = w - w / self.zoom_level
        max_pan_y = h - h / self.zoom_level
        self.pan_x = max(0.0, min(max_pan_x, self.pan_x))
        self.pan_y = max(0.0, min(max_pan_y, self.pan_y))

    def _mouse_callback(self, event, x, y, flags, param):
        # Ignore clicks in the status bar area (below the image)
        if y >= self.current_image.shape[0]:
            return
        # Ctrl + drag = lasso selection (draw shape, then Otsu threshold inside)
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing_lasso = True
                ix, iy = self._display_to_image(x, y)
                self.lasso_points = [(ix, iy)]
                self.mode = 1  # Add mode for lasso
                return
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.drawing_lasso = True
                ix, iy = self._display_to_image(x, y)
                self.lasso_points = [(ix, iy)]
                self.mode = 0  # Remove mode for lasso
                return
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing_lasso:
                ix, iy = self._display_to_image(x, y)
                self.lasso_points.append((ix, iy))
                self.mouse_x, self.mouse_y = x, y
                return
            elif (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP) and self.drawing_lasso:
                if len(self.lasso_points) > 2:
                    self._apply_lasso_selection()
                    self.history.append(self.current_mask.copy())
                self.drawing_lasso = False
                self.lasso_points = []
                return
        
        # Cancel lasso if ctrl released during draw
        if self.drawing_lasso and not (flags & cv2.EVENT_FLAG_CTRLKEY):
            self.drawing_lasso = False
            self.lasso_points = []
        
        # Shift + click = flood fill segment
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            ix, iy = self._display_to_image_clamped(x, y)
            self._flood_fill_add(ix, iy)
            self.history.append(self.current_mask.copy())
            return
        elif event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            ix, iy = self._display_to_image_clamped(x, y)
            self._flood_fill_remove(ix, iy)
            self.history.append(self.current_mask.copy())
            return
        
        # Middle click = pan
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            return
        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            return
        
        # Regular brush
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.mode = 1  # Add mode
            self._draw_at(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.mode = 0  # Remove mode
            self._draw_at(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x, self.mouse_y = x, y
            # Safety: clear stale drawing/panning flags if buttons aren't actually held
            # (fixes "stuck drawing" when mouse-up event is missed due to focus loss)
            if self.drawing and not (flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON)):
                self.history.append(self.current_mask.copy())
                self.drawing = False
            if self.panning and not (flags & cv2.EVENT_FLAG_MBUTTON):
                self.panning = False
            if self.panning:
                # Pan: move view by mouse delta
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                self.pan_x -= dx / self.zoom_level
                self.pan_y -= dy / self.zoom_level
                self.pan_start_x = x
                self.pan_start_y = y
                self._clamp_pan()
            elif self.drawing:
                self._draw_at(x, y)
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            if self.drawing:
                self.history.append(self.current_mask.copy())
            self.drawing = False

        # Scroll wheel = zoom centered on cursor
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Remember image point under cursor before zoom
            ix, iy = self._display_to_image(x, y)
            if flags > 0:  # Scroll up = zoom in
                self.zoom_level = min(8.0, self.zoom_level * 1.2)
            else:  # Scroll down = zoom out
                self.zoom_level = max(1.0, self.zoom_level / 1.2)
            # Recalculate pan so the same image point stays under cursor
            self.pan_x = ix - x / self.zoom_level
            self.pan_y = iy - y / self.zoom_level
            self._clamp_pan()
    
    def _draw_at(self, x, y):
        """Draw on mask at display position (x, y), transformed to image coords."""
        ix, iy = self._display_to_image(x, y)
        cv2.circle(self.current_mask, (ix, iy), self.brush_size, self.mode, -1)
    
    def _apply_lasso_selection(self):
        """Apply Otsu thresholding within the drawn lasso polygon."""
        if len(self.lasso_points) < 3:
            return
        
        h, w = self.current_image.shape[:2]
        
        # Create polygon mask from lasso points
        polygon = np.array(self.lasso_points, dtype=np.int32)
        lasso_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lasso_mask, [polygon], 1)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Extract pixels within the lasso region
        lasso_pixels = gray[lasso_mask > 0]
        
        if len(lasso_pixels) < 10:
            return
        
        # Use Otsu's method to find optimal threshold between dark (shadow) and light (surface)
        otsu_threshold, _ = cv2.threshold(lasso_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adjust threshold based on tolerance setting (lower tolerance = stricter, higher = more permissive)
        # At tolerance=50 (default-ish), use Otsu directly
        # At tolerance=0, use threshold - 20 (stricter)
        # At tolerance=100, use threshold + 20 (more permissive)
        adjustment = (self.flood_tolerance - 50) * 0.4
        threshold = otsu_threshold + adjustment
        
        # Create selection: pixels darker than threshold within the lasso
        dark_pixels = ((gray < threshold) & (lasso_mask > 0)).astype(np.uint8)
        
        # Morphological opening to remove small isolated spots (grass texture)
        kernel = np.ones((5, 5), np.uint8)
        dark_pixels = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest connected component (the main shadow body)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_pixels)
        
        if num_labels > 1:  # More than just background
            # Find the largest non-background component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            dark_pixels = (labels == largest_label).astype(np.uint8)
        
        # Apply to mask based on mode
        if self.mode == 1:  # Add
            self.current_mask = np.maximum(self.current_mask, dark_pixels)
        else:  # Remove
            self.current_mask = np.where(dark_pixels > 0, 0, self.current_mask).astype(np.uint8)
        
        logger.info(f"Lasso: Otsu={otsu_threshold:.0f}, adjusted={threshold:.0f}, selected {dark_pixels.sum()} pixels")
    
    def _flood_fill_add(self, x, y):
        """Flood fill to add a region. Uses brightness-only or LAB color based on mode."""
        h, w = self.current_image.shape[:2]
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        if self.brightness_mode:
            # RELATIVE darkness mode: compare each pixel to its local neighborhood
            # Shadow = darker than surroundings. Dark texture = similar to surroundings.
            
            # Compute local brightness using a blur (approximation of local context)
            # 51x51 kernel considers ~50px neighborhood
            local_median = cv2.blur(gray, (51, 51))
            
            # A pixel is "in shadow" if it's significantly darker than its local neighborhood
            darkness_ratio = gray.astype(np.float32) / (local_median.astype(np.float32) + 1)
            
            # Threshold: tolerance=30 means pixel must be 30% darker than surroundings
            # At tolerance=100, threshold=0 meaning any darker pixel qualifies
            threshold = 1.0 - (self.flood_tolerance / 100.0)
            shadow_mask = (darkness_ratio < threshold).astype(np.uint8)
            
            # Morphological closing to bridge small gaps (gravel texture, grass blades)
            # This connects nearby shadow regions that are separated by bright texture
            kernel_size = max(5, self.flood_tolerance // 5)  # Larger tolerance = more closing
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(shadow_mask)
            
            # Get the label at click point
            click_label = labels[y, x]
            
            if click_label > 0:
                filled_region = (labels == click_label).astype(np.uint8)
            else:
                filled_region = np.zeros((h, w), np.uint8)
        else:
            # LAB color mode: use edge detection + flood fill
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            flood_mask[1:-1, 1:-1] = (edges > 0).astype(np.uint8)
            
            lab_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            lo_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            hi_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            lab_copy = lab_image.copy()
            
            cv2.floodFill(
                lab_copy,
                flood_mask,
                (x, y),
                (255, 128, 128),
                lo_diff,
                hi_diff,
                cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )
            
            filled_region = flood_mask[1:-1, 1:-1]
            filled_region = np.where(edges > 0, 0, filled_region).astype(np.uint8)
        
        self.current_mask = np.maximum(self.current_mask, filled_region)
        
    def _flood_fill_remove(self, x, y):
        """Flood fill to remove a region. Uses brightness-only or LAB color based on mode."""
        h, w = self.current_image.shape[:2]
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        if self.brightness_mode:
            # RELATIVE darkness mode: compare each pixel to its local neighborhood
            local_median = cv2.blur(gray, (51, 51))
            darkness_ratio = gray.astype(np.float32) / (local_median.astype(np.float32) + 1)
            threshold = 1.0 - (self.flood_tolerance / 100.0)
            shadow_mask = (darkness_ratio < threshold).astype(np.uint8)
            
            # Morphological closing to bridge small gaps
            kernel_size = max(5, self.flood_tolerance // 5)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
            
            num_labels, labels = cv2.connectedComponents(shadow_mask)
            click_label = labels[y, x]
            
            if click_label > 0:
                filled_region = (labels == click_label).astype(np.uint8)
            else:
                filled_region = np.zeros((h, w), np.uint8)
        else:
            # LAB color mode: use edge detection + flood fill
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            flood_mask[1:-1, 1:-1] = (edges > 0).astype(np.uint8)
            
            lab_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            lo_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            hi_diff = (self.flood_tolerance, self.flood_tolerance // 2, self.flood_tolerance // 2)
            lab_copy = lab_image.copy()
            
            cv2.floodFill(
                lab_copy,
                flood_mask,
                (x, y),
                (255, 128, 128),
                lo_diff,
                hi_diff,
                cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
            )
            
            filled_region = flood_mask[1:-1, 1:-1]
            filled_region = np.where(edges > 0, 0, filled_region).astype(np.uint8)
        
        self.current_mask = np.where(filled_region > 0, 0, self.current_mask).astype(np.uint8)

    def _create_display(self):
        # A/B flicker: alternate between overlay and original every ~500ms (15 frames at ~30fps)
        if self.ab_flicker:
            self.ab_frame_counter += 1
            effective_mode = 0 if (self.ab_frame_counter // 15) % 2 == 0 else 1
        else:
            effective_mode = self.view_mode

        if effective_mode == 1:
            # Original image only
            display = self.current_image.copy()
        elif effective_mode == 2:
            # Mask only (white on black)
            display = cv2.cvtColor(self.current_mask * 255, cv2.COLOR_GRAY2BGR)
        else:
            # Overlay mode (default)
            display = self.current_image.copy()
            if self.show_mask:
                mask_overlay = np.zeros_like(display)
                mask_overlay[:, :, 2] = self.current_mask * 200  # Red channel
                display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)

                contours, _ = cv2.findContours(self.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

                # Highlight contour near cursor (yellow) for edge-aware editing
                if contours and not self.drawing:
                    cursor_img_x, cursor_img_y = self._display_to_image(self.mouse_x, self.mouse_y)
                    proximity = self.brush_size * 2
                    for cnt in contours:
                        dist = cv2.pointPolygonTest(cnt, (float(cursor_img_x), float(cursor_img_y)), True)
                        if abs(dist) < proximity:
                            cv2.drawContours(display, [cnt], -1, (0, 255, 255), 3)
                            break

        # Draw lasso path while drawing
        if self.drawing_lasso and len(self.lasso_points) > 1:
            # Draw the lasso path as a polyline
            pts = np.array(self.lasso_points, dtype=np.int32)
            lasso_color = (255, 255, 0) if self.mode == 1 else (0, 165, 255)  # Cyan for add, Orange for remove
            cv2.polylines(display, [pts], isClosed=False, color=lasso_color, thickness=2)
            # Draw dotted line back to start to show what will be closed
            if len(self.lasso_points) > 2:
                cv2.line(display, self.lasso_points[-1], self.lasso_points[0], lasso_color, 1, cv2.LINE_AA)
        
        # Apply zoom by cropping and resizing
        if self.zoom_level > 1.0:
            h, w = display.shape[:2]
            # Calculate visible area size
            view_w = int(w / self.zoom_level)
            view_h = int(h / self.zoom_level)

            # Use pan position as crop origin, clamped to bounds
            x1 = int(max(0, min(w - view_w, self.pan_x)))
            y1 = int(max(0, min(h - view_h, self.pan_y)))
            x2 = x1 + view_w
            y2 = y1 + view_h

            cropped = display[y1:y2, x1:x2]

            # Resize back to original size
            display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Help overlay
        if self.show_help:
            self._draw_help_overlay(display)

        # Draw brush cursor at actual mouse position
        cursor_color = (0, 255, 0) if self.mode == 1 else (0, 0, 255)
        brush_display_size = int(self.brush_size * self.zoom_level)
        cv2.circle(display, (self.mouse_x, self.mouse_y), brush_display_size, cursor_color, 2)
        cv2.circle(display, (self.mouse_x, self.mouse_y), 2, cursor_color, -1)
        
        # Status bar — appended below image so it doesn't cover the nadir
        zoom_pct = int(self.zoom_level * 100)
        fill_mode = "BRIGHTNESS" if self.brightness_mode else "COLOR"
        view_names = ["OVERLAY", "ORIGINAL", "MASK"]
        view_name = view_names[self.view_mode]
        mask_pixels = int(np.sum(self.current_mask > 0))
        total_pixels = self.current_mask.shape[0] * self.current_mask.shape[1]
        coverage = (mask_pixels / total_pixels * 100) if total_pixels > 0 else 0
        save_flash = getattr(self, '_save_flash', 0)
        if save_flash > 0:
            status = (f"SAVED!  |  Brush: {self.brush_size} | Tol: {self.flood_tolerance} | "
                      f"Zoom: {zoom_pct}% | Mask: {coverage:.1f}%")
            bar_color, text_color = (30, 90, 30), (100, 255, 100)
        else:
            status = (f"Brush: {self.brush_size} | Tol: {self.flood_tolerance} | Zoom: {zoom_pct}% | "
                      f"Fill: {fill_mode} | View: {view_name} | Mask: {coverage:.1f}% | 'h' = help")
            bar_color, text_color = (40, 40, 40), (255, 255, 255)
        bar = np.full((30, display.shape[1], 3), 0, dtype=np.uint8)
        bar[:] = bar_color
        cv2.putText(bar, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        display = np.vstack([display, bar])

        return display
    
    def _draw_help_overlay(self, display):
        """Draw a semi-transparent help panel."""
        h, w = display.shape[:2]
        
        # Help panel dimensions
        panel_w, panel_h = 340, 400
        panel_x, panel_y = 10, 10
        
        # Semi-transparent background
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, display, 0.15, 0, display)
        
        # Border
        cv2.rectangle(display, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(display, "CONTROLS", (panel_x + 110, panel_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Help text
        controls = [
            ("Left Click", "Brush ADD to mask"),
            ("Right Click", "Brush REMOVE from mask"),
            ("Ctrl + Drag", "LASSO select (Otsu threshold)"),
            ("Shift + Click", "Flood fill add/remove"),
            ("Middle Drag", "Pan view"),
            ("Scroll Wheel", "Zoom in/out"),
            ("", ""),
            ("s", "Save and go to next"),
            ("n", "Skip (don't save)"),
            ("q", "Quit review"),
            ("r", "Reset to original"),
            ("u", "Undo last change"),
            ("+/-", "Adjust brush size"),
            ("[/]", "Adjust flood tolerance"),
            ("0", "Reset zoom to 100%"),
            ("v", "Cycle view (overlay/original/mask)"),
            ("m", "Toggle overlay/original"),
            ("a", "A/B flicker comparison"),
            ("b", "Toggle fill mode (color/brightness)"),
            ("h", "Toggle this help"),
        ]
        
        y_offset = panel_y + 50
        for key, desc in controls:
            if key == "":
                y_offset += 5
                continue
            cv2.putText(display, key, (panel_x + 10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
            cv2.putText(display, desc, (panel_x + 100, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 18
    
    def _load_and_scale(self, image_path, mask_path):
        """Load image+mask, scale to fit screen, reset editing state.

        Returns (orig_h, orig_w) of the mask for save-time upscaling,
        or None if loading failed.
        """
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.panning = False
        self.drawing = False

        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Failed to load mask: {mask_path}")
            return None

        orig_h, orig_w = mask.shape[:2]

        # Mask convention: black (0) = foreground to remove, white (255) = keep
        # Internal: 1 = masked/remove, 0 = keep (for red overlay display)
        self.current_mask = (mask < 128).astype(np.uint8)
        self.original_mask = self.current_mask.copy()
        self.history = [self.current_mask.copy()]

        h, w = self.current_image.shape[:2]
        try:
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = 1920, 1080
        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.9) - 30
        self.scale = min(max_w / w, max_h / h, 1.0)
        if self.scale < 1.0:
            new_w, new_h = int(w * self.scale), int(h * self.scale)
            self.current_image = cv2.resize(self.current_image, (new_w, new_h))
            self.current_mask = cv2.resize(self.current_mask, (new_w, new_h))
            self.original_mask = cv2.resize(self.original_mask, (new_w, new_h))

        return orig_h, orig_w

    def _save_window_pos(self, window_name):
        """Save window position before destroying it."""
        try:
            import ctypes
            from ctypes import wintypes
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd:
                rect = wintypes.RECT()
                ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                self._last_window_pos = (rect.left, rect.top)
        except Exception:
            pass

    def review_mask(self, image_path: Path, mask_path: Path, review_path: Path = None, window_name: str = "Mask Review"):
        """
        Review and edit a single mask (one-shot mode).

        Returns:
            'save': saved changes
            'skip': skipped without saving
            'quit': user wants to quit
        """
        result = self._load_and_scale(image_path, mask_path)
        if result is None:
            return 'skip'
        orig_h, orig_w = result
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.current_image.shape[1], self.current_image.shape[0] + 30)
        if self._last_window_pos:
            cv2.moveWindow(window_name, *self._last_window_pos)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        logger.info(f"Reviewing: {image_path.name}")
        
        result = 'skip'
        while True:
            display = self._create_display()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90:  # Caps Lock / Shift: normalize A-Z → a-z
                key += 32

            if key == ord('s'):
                # Save and continue - resize back to original
                if self.scale < 1.0:
                    final_mask = cv2.resize(self.current_mask, (orig_w, orig_h))
                else:
                    final_mask = self.current_mask
                # Invert back: internal 1=remove → disk black=remove, white=keep
                cv2.imwrite(str(mask_path), (1 - final_mask) * 255)
                logger.info(f"Saved: {mask_path.name}")
                
                # Delete review image if it exists
                if review_path and review_path.exists():
                    review_path.unlink()
                    logger.info(f"Removed from review: {review_path.name}")
                
                result = 'save'
                break
            elif key == ord('n'):
                logger.info("Skipped")
                result = 'skip'
                break
            elif key == ord('q'):
                result = 'quit'
                break
            elif key == ord('r'):
                self.current_mask = self.original_mask.copy()
                self.history = [self.original_mask.copy()]
            elif key == ord('u') and len(self.history) > 1:
                self.history.pop()
                self.current_mask = self.history[-1].copy()
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(100, self.brush_size + 5)
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 5)
            elif key == ord(']'):
                self.flood_tolerance = min(100, self.flood_tolerance + 5)
            elif key == ord('['):
                self.flood_tolerance = max(5, self.flood_tolerance - 5)
            elif key == ord('v'):
                self.view_mode = (self.view_mode + 1) % 3  # overlay → original → mask-only
            elif key == ord('m'):
                # Toggle overlay ↔ original (backward compat)
                self.view_mode = 1 if self.view_mode == 0 else 0
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('a'):
                self.ab_flicker = not self.ab_flicker
                self.ab_frame_counter = 0
                logger.info(f"A/B flicker: {'ON' if self.ab_flicker else 'OFF'}")
            elif key == ord('b'):
                self.brightness_mode = not self.brightness_mode
                mode_name = "BRIGHTNESS" if self.brightness_mode else "COLOR"
                logger.info(f"Fill mode: {mode_name}")
            elif key == ord('0'):
                # Reset zoom and pan
                self.zoom_level = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0
        
        self._save_window_pos(window_name)
        cv2.destroyWindow(window_name)
        return result

    def edit_session(self, image_path, mask_path, load_queue, save_callback=None,
                     window_name="Mask Editor"):
        """Persistent editing session — window stays open until 'q'.

        Instead of one window per image (review_mask), this keeps a single
        window open. The GUI pushes new images via load_queue and the user
        navigates freely.

        Args:
            image_path: Initial image to load
            mask_path: Initial mask to load
            load_queue: threading.Queue — GUI puts (image_path, mask_path, stem) tuples
            save_callback: callable(mask_path) — called after each save (from editor thread)
            window_name: OpenCV window title
        """
        import queue as queue_mod

        dims = self._load_and_scale(image_path, mask_path)
        if dims is None:
            return
        orig_h, orig_w = dims
        self._current_mask_path = mask_path
        self._save_flash = 0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.current_image.shape[1],
                         self.current_image.shape[0] + 30)
        if self._last_window_pos:
            cv2.moveWindow(window_name, *self._last_window_pos)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        logger.info(f"Edit session started: {image_path.name}")

        while True:
            # Check for new image from GUI (non-blocking)
            try:
                new_img, new_mask, new_stem = load_queue.get_nowait()
                dims = self._load_and_scale(new_img, new_mask)
                if dims is not None:
                    orig_h, orig_w = dims
                    self._current_mask_path = new_mask
                    self._save_flash = 0
                    try:
                        cv2.setWindowTitle(window_name, f"Edit: {new_stem}")
                    except Exception:
                        pass
                    cv2.resizeWindow(window_name, self.current_image.shape[1],
                                     self.current_image.shape[0] + 30)
                    logger.info(f"Loaded: {new_stem}")
            except queue_mod.Empty:
                pass

            # Detect window closed via X button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            display = self._create_display()
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90:  # Caps Lock / Shift: normalize A-Z → a-z
                key += 32

            if key == ord('s'):
                if self.scale < 1.0:
                    final_mask = cv2.resize(self.current_mask, (orig_w, orig_h))
                else:
                    final_mask = self.current_mask
                # Invert back: internal 1=remove → disk black=remove, white=keep
                cv2.imwrite(str(self._current_mask_path), (1 - final_mask) * 255)
                logger.info(f"Saved: {Path(self._current_mask_path).name}")
                self._save_flash = 45
                if save_callback:
                    save_callback(self._current_mask_path)
                # Window stays open — don't break
            elif key == ord('q'):
                break
            elif key == ord('r'):
                self.current_mask = self.original_mask.copy()
                self.history = [self.original_mask.copy()]
            elif key == ord('u') and len(self.history) > 1:
                self.history.pop()
                self.current_mask = self.history[-1].copy()
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(100, self.brush_size + 5)
            elif key == ord('-'):
                self.brush_size = max(1, self.brush_size - 5)
            elif key == ord(']'):
                self.flood_tolerance = min(100, self.flood_tolerance + 5)
            elif key == ord('['):
                self.flood_tolerance = max(5, self.flood_tolerance - 5)
            elif key == ord('v'):
                self.view_mode = (self.view_mode + 1) % 3
            elif key == ord('m'):
                self.view_mode = 1 if self.view_mode == 0 else 0
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('a'):
                self.ab_flicker = not self.ab_flicker
                self.ab_frame_counter = 0
            elif key == ord('b'):
                self.brightness_mode = not self.brightness_mode
            elif key == ord('0'):
                self.zoom_level = 1.0
                self.pan_x = 0.0
                self.pan_y = 0.0

            if self._save_flash > 0:
                self._save_flash -= 1

        self._save_window_pos(window_name)
        cv2.destroyWindow(window_name)


def main():
    parser = argparse.ArgumentParser(description="Batch mask review tool with click-to-segment")
    parser.add_argument("masks_dir", type=Path, help="Directory containing masks to review")
    parser.add_argument("--images-dir", type=Path, help="Directory containing original images")
    parser.add_argument("--review-only", action="store_true", help="Only review masks flagged for review")
    
    args = parser.parse_args()
    
    masks_dir = Path(args.masks_dir)
    
    # Find masks folder
    if (masks_dir / "masks").exists():
        actual_masks_dir = masks_dir / "masks"
        review_dir = masks_dir / "review"
    else:
        actual_masks_dir = masks_dir
        review_dir = masks_dir.parent / "review"
    
    # Find original images
    if args.images_dir:
        images_dir = args.images_dir
    else:
        possible_paths = [
            masks_dir.parent / "frames",
            masks_dir.parent.parent / "frames",
        ]
        images_dir = None
        for p in possible_paths:
            if p.exists():
                images_dir = p
                break
        if images_dir is None:
            logger.error("Could not find images directory. Please specify with --images-dir")
            return
    
    logger.info(f"Masks dir: {actual_masks_dir}")
    logger.info(f"Images dir: {images_dir}")
    
    # Get list of masks to review
    if args.review_only and review_dir.exists():
        review_files = list(review_dir.glob("review_*.jpg")) + list(review_dir.glob("review_*.png"))
        mask_stems = [f.stem.replace("review_", "") for f in review_files]
    else:
        mask_files = list(actual_masks_dir.glob("mask_*.png")) + list(actual_masks_dir.glob("*_mask.png"))
        mask_stems = [f.stem.replace("mask_", "").replace("_mask", "") for f in mask_files]
    
    logger.info(f"Found {len(mask_stems)} masks to review")
    
    reviewer = MaskReviewer()
    stats = {'saved': 0, 'skipped': 0, 'total': len(mask_stems)}
    
    for i, stem in enumerate(sorted(mask_stems)):
        logger.info(f"\n[{i+1}/{len(mask_stems)}] {stem}")
        
        # Find image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        
        if image_path is None:
            logger.warning(f"Image not found for: {stem}")
            continue
        
        # Find mask
        mask_path = None
        for pattern in [f"mask_{stem}.png", f"{stem}_mask.png"]:
            candidate = actual_masks_dir / pattern
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            logger.warning(f"Mask not found for: {stem}")
            continue
        
        # Find review file (to delete after save)
        review_path = None
        if review_dir.exists():
            for ext in ['.jpg', '.png']:
                candidate = review_dir / f"review_{stem}{ext}"
                if candidate.exists():
                    review_path = candidate
                    break
        
        result = reviewer.review_mask(image_path, mask_path, review_path)
        
        if result == 'save':
            stats['saved'] += 1
        elif result == 'skip':
            stats['skipped'] += 1
        elif result == 'quit':
            break
    
    logger.info(f"\nReview complete: {stats['saved']} saved, {stats['skipped']} skipped out of {stats['total']}")


def edit_session_subprocess():
    """Standalone edit session launched as a subprocess by Masking Studio.

    Uses file-based IPC:
      --cmd-file: Studio writes lines like "image_path|mask_path|stem" to push new images
      --signal-file: Editor writes "saved|mask_path" after each save
    """
    parser = argparse.ArgumentParser(description="Mask editor subprocess session")
    parser.add_argument("image_path", type=Path)
    parser.add_argument("mask_path", type=Path)
    parser.add_argument("--cmd-file", type=Path, required=True,
                        help="File the studio writes commands to")
    parser.add_argument("--signal-file", type=Path, required=True,
                        help="File the editor writes save signals to")
    parser.add_argument("--window-name", default="Mask Editor")
    args = parser.parse_args()

    import queue as queue_mod

    reviewer = MaskReviewer()
    dims = reviewer._load_and_scale(args.image_path, args.mask_path)
    if dims is None:
        logger.error(f"Failed to load: {args.image_path}")
        return
    orig_h, orig_w = dims
    reviewer._current_mask_path = args.mask_path
    reviewer._save_flash = 0

    window_name = args.window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, reviewer.current_image.shape[1],
                     reviewer.current_image.shape[0] + 30)
    if reviewer._last_window_pos:
        cv2.moveWindow(window_name, *reviewer._last_window_pos)
    cv2.setMouseCallback(window_name, reviewer._mouse_callback)

    # Clear cmd file so we only read fresh commands
    args.cmd_file.write_text("", encoding="utf-8")

    logger.info(f"Edit session started: {args.image_path.name}")

    while True:
        # Poll cmd file for new image commands from studio
        try:
            content = args.cmd_file.read_text(encoding="utf-8").strip()
            if content:
                args.cmd_file.write_text("", encoding="utf-8")
                # Take the last command if multiple were written
                line = content.strip().splitlines()[-1]
                parts = line.split("|")
                if len(parts) >= 3:
                    new_img, new_mask, new_stem = Path(parts[0]), Path(parts[1]), parts[2]
                    d = reviewer._load_and_scale(new_img, new_mask)
                    if d is not None:
                        orig_h, orig_w = d
                        reviewer._current_mask_path = new_mask
                        reviewer._save_flash = 0
                        try:
                            cv2.setWindowTitle(window_name, f"Edit: {new_stem}")
                        except Exception:
                            pass
                        cv2.resizeWindow(window_name, reviewer.current_image.shape[1],
                                         reviewer.current_image.shape[0] + 30)
                        logger.info(f"Loaded: {new_stem}")
        except Exception:
            pass

        # Detect window closed
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

        display = reviewer._create_display()
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if 65 <= key <= 90:  # Caps Lock / Shift: normalize A-Z → a-z
            key += 32

        if key == ord('s'):
            if reviewer.scale < 1.0:
                final_mask = cv2.resize(reviewer.current_mask, (orig_w, orig_h))
            else:
                final_mask = reviewer.current_mask
            cv2.imwrite(str(reviewer._current_mask_path), (1 - final_mask) * 255)
            logger.info(f"Saved: {Path(reviewer._current_mask_path).name}")
            reviewer._save_flash = 45
            # Signal save to studio — distinguish modified vs unchanged
            changed = not np.array_equal(reviewer.current_mask, reviewer.original_mask)
            tag = "saved_modified" if changed else "saved_unchanged"
            try:
                with open(args.signal_file, "a", encoding="utf-8") as f:
                    f.write(f"{tag}|{reviewer._current_mask_path}\n")
            except Exception:
                pass
        elif key == ord('q'):
            break
        elif key == ord('r'):
            reviewer.current_mask = reviewer.original_mask.copy()
            reviewer.history = [reviewer.original_mask.copy()]
        elif key == ord('u') and len(reviewer.history) > 1:
            reviewer.history.pop()
            reviewer.current_mask = reviewer.history[-1].copy()
        elif key == ord('+') or key == ord('='):
            reviewer.brush_size = min(100, reviewer.brush_size + 5)
        elif key == ord('-'):
            reviewer.brush_size = max(1, reviewer.brush_size - 5)
        elif key == ord(']'):
            reviewer.flood_tolerance = min(100, reviewer.flood_tolerance + 5)
        elif key == ord('['):
            reviewer.flood_tolerance = max(5, reviewer.flood_tolerance - 5)
        elif key == ord('v'):
            reviewer.view_mode = (reviewer.view_mode + 1) % 3
        elif key == ord('m'):
            reviewer.view_mode = 1 if reviewer.view_mode == 0 else 0
        elif key == ord('h'):
            reviewer.show_help = not reviewer.show_help
        elif key == ord('a'):
            reviewer.ab_flicker = not reviewer.ab_flicker
            reviewer.ab_frame_counter = 0
        elif key == ord('b'):
            reviewer.brightness_mode = not reviewer.brightness_mode
        elif key == ord('0'):
            reviewer.zoom_level = 1.0
            reviewer.pan_x = 0.0
            reviewer.pan_y = 0.0

        if reviewer._save_flash > 0:
            reviewer._save_flash -= 1

    reviewer._save_window_pos(window_name)
    cv2.destroyWindow(window_name)
    # Signal closed
    try:
        with open(args.signal_file, "a", encoding="utf-8") as f:
            f.write("closed\n")
    except Exception:
        pass


if __name__ == "__main__":
    # Check if invoked in edit-session mode (subprocess from studio)
    if len(sys.argv) > 1 and "--cmd-file" in sys.argv:
        edit_session_subprocess()
    else:
        main()
