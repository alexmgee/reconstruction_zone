"""
Point Cloud Viewer — VTK 3D viewer for COLMAP sparse reconstructions.

Embeds a VTK render window in a tkinter frame via native HWND parenting
(SetParentInfo). No vtkRenderingTk.dll required.

Usage::

    from pointcloud_viewer import PointCloudViewer

    if PointCloudViewer.available():
        viewer = PointCloudViewer(parent_frame)
        viewer.load_model(model_data)  # dict from ColmapRunner.parse_model()
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy dependency check ────────────────────────────────────────────

_HAS_VTK = None


def _check_vtk() -> bool:
    global _HAS_VTK
    if _HAS_VTK is None:
        try:
            from vtkmodules.vtkRenderingCore import (  # noqa: F401
                vtkActor,
                vtkPolyDataMapper,
                vtkRenderer,
                vtkRenderWindow,
                vtkRenderWindowInteractor,
            )
            from vtkmodules.vtkInteractionStyle import (  # noqa: F401
                vtkInteractorStyleTrackballCamera,
            )
            import pyvista  # noqa: F401

            _HAS_VTK = True
        except ImportError:
            _HAS_VTK = False
    return _HAS_VTK


# ── Camera model constants ───────────────────────────────────────────

_PERSPECTIVE_MODELS = {
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL",
    "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
}
_SPHERICAL_MODELS = {"SPHERE"}


class PointCloudViewer:
    """VTK 3D viewer for COLMAP sparse reconstructions, embedded in tkinter.

    Embeds via vtkRenderWindow.SetParentInfo(hwnd) — does not require
    vtkRenderingTk.dll. Mouse interaction handled by VTK's own interactor
    pumped from tkinter's event loop.
    """

    COLOR_MODES = ["rgb", "reproj_error", "track_length", "depth", "elevation"]

    _BACKGROUND = (0.04, 0.04, 0.10)
    _CAMERA_COLOR = (0.96, 0.62, 0.04)  # #f59e0b
    _CAMERA_OPACITY = 0.7
    _PUMP_INTERVAL_MS = 16  # ~60fps

    @classmethod
    def available(cls) -> bool:
        """Check whether VTK and pyvista are importable."""
        return _check_vtk()

    def __init__(self, parent_frame: tk.Frame):
        if not self.available():
            raise RuntimeError(
                "PointCloudViewer requires pyvista and vtk. "
                "Install with: pip install pyvista"
            )

        from vtkmodules.vtkRenderingCore import (
            vtkRenderWindow,
            vtkRenderWindowInteractor,
        )
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        import pyvista as pv

        self._parent = parent_frame
        self._destroyed = False
        self._pump_id: Optional[str] = None
        self._model_data: Optional[Dict[str, Any]] = None
        self._point_actor = None
        self._camera_actors: List = []
        self._color_mode = "rgb"
        self._point_size = 3.0
        self._show_cameras = True
        self._applied_roll = 0.0  # tracks roll we've applied via set_roll()

        # Use pyvista to create the renderer (ensures proper OpenGL init)
        self._pv_plotter = pv.Plotter(off_screen=True)
        self._pv_plotter.set_background(color=list(self._BACKGROUND))
        self._renderer = self._pv_plotter.renderer

        # Transfer renderer from pyvista's window to our embedded window
        self._pv_plotter.ren_win.RemoveRenderer(self._renderer)

        self._render_window = vtkRenderWindow()
        self._render_window.AddRenderer(self._renderer)

        # Embed in tkinter frame via native HWND
        parent_frame.update_idletasks()
        hwnd = parent_frame.winfo_id()
        self._render_window.SetParentInfo(str(hwnd))

        w = parent_frame.winfo_width() or 800
        h = parent_frame.winfo_height() or 600
        self._render_window.SetSize(w, h)

        # Interactor: Initialize but NOT Start (tkinter drives the loop)
        self._interactor = vtkRenderWindowInteractor()
        self._interactor.SetRenderWindow(self._render_window)
        style = vtkInteractorStyleTrackballCamera()
        self._interactor.SetInteractorStyle(style)
        self._interactor.Initialize()

        # Navigation gizmo — clickable orientation cube in the top-right corner
        from vtkmodules.vtkInteractionWidgets import vtkCameraOrientationWidget
        self._orientation_widget = vtkCameraOrientationWidget()
        self._orientation_widget.SetParentRenderer(self._renderer)
        self._orientation_widget.On()

        # XYZ axes indicator — separate renderer overlay in the bottom-left.
        # Uses its own renderer + camera so it doesn't interfere with the
        # main viewport's mouse events or aspect ratio.
        from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
        from vtkmodules.vtkRenderingCore import vtkRenderer
        axes = vtkAxesActor()
        axes.SetShaftTypeToCylinder()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetCylinderRadius(0.03)
        axes.SetConeRadius(0.15)
        axes.SetXAxisLabelText("X")
        axes.SetYAxisLabelText("Y")
        axes.SetZAxisLabelText("Z")
        for getter in (axes.GetXAxisCaptionActor2D,
                       axes.GetYAxisCaptionActor2D,
                       axes.GetZAxisCaptionActor2D):
            getter().GetCaptionTextProperty().SetFontSize(12)
            getter().GetCaptionTextProperty().BoldOn()
            getter().GetCaptionTextProperty().ShadowOff()
        self._axes_renderer = vtkRenderer()
        self._axes_renderer.SetViewport(0.0, 0.0, 0.13, 0.13)
        self._axes_renderer.SetBackground(0.0, 0.0, 0.0)
        self._axes_renderer.SetBackgroundAlpha(0.0)
        self._axes_renderer.InteractiveOff()
        self._axes_renderer.AddActor(axes)
        self._axes_renderer.ResetCamera()
        self._render_window.AddRenderer(self._axes_renderer)
        self._axes_actor = axes

        # Resize handling
        parent_frame.bind("<Configure>", self._on_resize)

        # Start event pump
        self._start_pump()

        # Deferred first render
        parent_frame.after(200, self.render)

    # ── Event pump ───────────────────────────────────────────────────

    def _start_pump(self):
        """Start pumping VTK events from tkinter's mainloop."""
        if self._destroyed:
            return
        self._pump()

    def _pump(self):
        """Single pump cycle — process VTK events, schedule next."""
        if self._destroyed:
            return
        try:
            self._interactor.ProcessEvents()
            self._guard_gizmo_zoom()
            self._sync_axes_camera()
        except Exception:
            pass
        self._pump_id = self._parent.after(self._PUMP_INTERVAL_MS, self._pump)

    def _guard_gizmo_zoom(self):
        """Prevent the orientation gizmo from changing zoom level.

        Normal orbit preserves camera-focal distance.  Scroll-zoom changes
        it gradually.  The gizmo snap changes direction AND distance in one
        frame.  We detect the snap (large direction change) and restore the
        previous distance while keeping the new direction.
        """
        cam = self._renderer.GetActiveCamera()
        pos = np.array(cam.GetPosition())
        foc = np.array(cam.GetFocalPoint())
        dist = np.linalg.norm(pos - foc)
        if dist < 1e-12:
            return

        direction = (pos - foc) / dist

        if hasattr(self, "_guard_dir") and hasattr(self, "_guard_dist"):
            dot = float(np.clip(np.dot(direction, self._guard_dir), -1, 1))
            angle_change = np.degrees(np.arccos(dot))

            if angle_change > 20:
                # Gizmo snap detected — keep new direction, restore distance
                new_pos = foc + direction * self._guard_dist
                cam.SetPosition(new_pos[0], new_pos[1], new_pos[2])
                self._renderer.ResetCameraClippingRange()
                self._guard_dir = direction
                # _guard_dist intentionally NOT updated — preserve user's zoom
                return

        self._guard_dir = direction
        self._guard_dist = dist

    def _sync_axes_camera(self):
        """Copy main camera orientation to the axes overlay camera."""
        if not hasattr(self, "_axes_renderer"):
            return
        main_cam = self._renderer.GetActiveCamera()
        axes_cam = self._axes_renderer.GetActiveCamera()
        # Copy orientation only (position/focal stay fixed so arrows stay centred)
        axes_cam.SetPosition(main_cam.GetPosition())
        axes_cam.SetFocalPoint(main_cam.GetFocalPoint())
        axes_cam.SetViewUp(main_cam.GetViewUp())
        self._axes_renderer.ResetCamera()  # re-fit arrows to the small viewport

    def pause_pump(self):
        """Stop the event pump (call when hiding the viewer / tab switch away)."""
        if self._pump_id is not None:
            try:
                self._parent.after_cancel(self._pump_id)
            except Exception:
                pass
            self._pump_id = None

    def resume_pump(self):
        """Restart the event pump (call when showing the viewer / tab switch back)."""
        if self._destroyed:
            return
        if self._pump_id is None:
            self._start_pump()
        self.render()

    # ── Render ───────────────────────────────────────────────────────

    def render(self):
        """Force a render. No-op if destroyed."""
        if self._destroyed:
            return
        try:
            self._sync_axes_camera()
            self._render_window.Render()
        except Exception:
            pass

    def _on_resize(self, event):
        if self._destroyed:
            return
        if event.widget == self._parent and event.width > 1 and event.height > 1:
            self._render_window.SetSize(event.width, event.height)
            self.render()

    # ── Model loading ────────────────────────────────────────────────

    def load_model(self, model_data: Dict[str, Any]):
        """Load a parsed COLMAP sparse model (from ColmapRunner.parse_model()).

        Clears any existing model, builds point cloud + camera actors,
        resets the camera view.
        """
        self.clear_model()
        self._model_data = model_data

        points3D = model_data.get("points3D", {})
        images = model_data.get("images", {})
        cameras = model_data.get("cameras", {})

        if points3D:
            self._build_point_cloud(points3D)
        if images and cameras:
            self._build_cameras(images, cameras)

        self._reapply_upright()
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self.render()
        # Force additional renders to handle DWM/compositor timing on Windows
        self._parent.after(100, self.render)
        self._parent.after(300, self.render)

    def clear_model(self):
        """Remove all actors from the scene."""
        if self._point_actor is not None:
            self._renderer.RemoveActor(self._point_actor)
            self._point_actor = None
        for actor in self._camera_actors:
            self._renderer.RemoveActor(actor)
        self._camera_actors.clear()
        self._model_data = None
        self.render()

    def _reapply_upright(self):
        """Apply the current upright transform (if any) to all actors."""
        if not hasattr(self, "_upright_matrix"):
            return
        import vtk
        mat4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat4.SetElement(i, j, self._upright_matrix[i, j])
        t = vtk.vtkTransform()
        t.SetMatrix(mat4)
        if self._point_actor is not None:
            self._point_actor.SetUserTransform(t)
        for actor in self._camera_actors:
            actor.SetUserTransform(t)

    # ── Point cloud ──────────────────────────────────────────────────

    def _build_point_cloud(self, points3D: dict):
        """Build VTK point cloud actor from COLMAPPoint3D dict."""
        import vtk
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        from vtkmodules.util.numpy_support import numpy_to_vtk

        if self._point_actor is not None:
            self._renderer.RemoveActor(self._point_actor)

        point_list = list(points3D.values())
        n = len(point_list)
        if n == 0:
            return

        positions = np.array([p.xyz for p in point_list], dtype=np.float64)
        rgb = np.array([p.rgb for p in point_list], dtype=np.uint8)
        errors = np.array([p.error for p in point_list], dtype=np.float64)
        track_lengths = np.array([len(p.track) for p in point_list], dtype=np.float64)

        # Derived scalars
        centroid = positions.mean(axis=0)
        depth = np.linalg.norm(positions - centroid, axis=1)
        elevation = positions[:, 2]

        # Store scalars for color mode switching
        self._point_scalars = {
            "rgb": rgb,
            "reproj_error": errors,
            "track_length": track_lengths,
            "depth": depth,
            "elevation": elevation,
        }
        self._point_positions = positions
        self._point_count = n

        # Build VTK polydata
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(positions))

        vtk_verts = vtk.vtkCellArray()
        for i in range(n):
            vtk_verts.InsertNextCell(1)
            vtk_verts.InsertCellPoint(i)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetVerts(vtk_verts)

        # Apply current color mode
        self._apply_color_to_polydata(polydata)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self._point_size)
        actor.GetProperty().RenderPointsAsSpheresOn()

        self._renderer.AddActor(actor)
        self._point_actor = actor
        self._point_polydata = polydata

    def _apply_color_to_polydata(self, polydata):
        """Set the scalar colors on polydata based on current color mode."""
        from vtkmodules.util.numpy_support import numpy_to_vtk

        mode = self._color_mode
        scalars = self._point_scalars

        if mode == "rgb":
            colors = scalars["rgb"]
        else:
            # Map scalar to RGB via matplotlib colormap
            import matplotlib.cm as cm
            raw = scalars.get(mode, scalars["reproj_error"])
            vmin, vmax = raw.min(), raw.max()
            if vmax - vmin < 1e-10:
                norm = np.zeros_like(raw)
            else:
                norm = (raw - vmin) / (vmax - vmin)
            cmap = cm.get_cmap("viridis")
            colors = (cmap(norm)[:, :3] * 255).astype(np.uint8)

        vtk_colors = numpy_to_vtk(colors)
        vtk_colors.SetName("colors")
        polydata.GetPointData().SetScalars(vtk_colors)

    # ── Camera rendering ─────────────────────────────────────────────

    def _build_cameras(self, images: dict, cameras: dict):
        """Build camera visualization actors."""
        for actor in self._camera_actors:
            self._renderer.RemoveActor(actor)
        self._camera_actors.clear()

        if not self._show_cameras:
            return

        for img in images.values():
            cam = cameras.get(img.camera_id)
            if cam is None:
                continue

            center = img.get_camera_center()
            R = img.get_rotation()
            model = cam.model.upper() if hasattr(cam, "model") else "PINHOLE"

            if model in _SPHERICAL_MODELS:
                self._add_sphere_camera(center, R)
            else:
                self._add_perspective_frustum(center, R, cam)

    def _add_perspective_frustum(self, center, R, cam):
        """Add a perspective camera frustum wireframe."""
        import vtk
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        f = cam.params[0] if cam.params else 500
        scale = 0.3
        hw = cam.width / (2 * f) * scale
        hh = cam.height / (2 * f) * scale

        # Frustum corners in camera space (Z forward)
        corners_cam = np.array([
            [-hw, -hh, scale],
            [hw, -hh, scale],
            [hw, hh, scale],
            [-hw, hh, scale],
        ])

        # Transform to world space
        corners_world = (R.T @ corners_cam.T).T + center

        # Build line polydata
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        apex_id = points.InsertNextPoint(center)
        corner_ids = [points.InsertNextPoint(c) for c in corners_world]

        for cid in corner_ids:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, apex_id)
            line.GetPointIds().SetId(1, cid)
            lines.InsertNextCell(line)

        for i in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, corner_ids[i])
            line.GetPointIds().SetId(1, corner_ids[(i + 1) % 4])
            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self._CAMERA_COLOR)
        actor.GetProperty().SetLineWidth(1.5)
        actor.GetProperty().SetOpacity(self._CAMERA_OPACITY)

        self._renderer.AddActor(actor)
        self._camera_actors.append(actor)

    def _add_sphere_camera(self, center, R):
        """Add a spherical camera marker (center glyph + forward direction)."""
        import pyvista as pv
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        # Center sphere
        sphere = pv.Sphere(radius=0.15, center=center)
        mapper_s = vtkPolyDataMapper()
        mapper_s.SetInputData(sphere)
        actor_s = vtkActor()
        actor_s.SetMapper(mapper_s)
        actor_s.GetProperty().SetColor(*self._CAMERA_COLOR)
        actor_s.GetProperty().SetOpacity(self._CAMERA_OPACITY)
        self._renderer.AddActor(actor_s)
        self._camera_actors.append(actor_s)

        # Forward direction line (camera Z axis in world)
        forward = R.T @ np.array([0, 0, 1.0])
        tip = center + forward * 0.5
        line = pv.Line(center, tip)
        mapper_l = vtkPolyDataMapper()
        mapper_l.SetInputData(line)
        actor_l = vtkActor()
        actor_l.SetMapper(mapper_l)
        actor_l.GetProperty().SetColor(*self._CAMERA_COLOR)
        actor_l.GetProperty().SetLineWidth(2.0)
        actor_l.GetProperty().SetOpacity(self._CAMERA_OPACITY)
        self._renderer.AddActor(actor_l)
        self._camera_actors.append(actor_l)

    # ── Set Upright (bake camera orientation into model) ───────────

    def set_upright(self):
        """Bake the current camera orientation into the model transform.

        After this call the reconstruction is rotated so that the current
        view becomes the "front" view.  The camera is then reset to
        default so that the orientation gizmo's X/Y/Z snap-views
        correspond to meaningful directions (top, front, side).
        """
        import vtk

        if self._destroyed or self._model_data is None:
            return

        cam = self._renderer.GetActiveCamera()
        pos = np.array(cam.GetPosition())
        foc = np.array(cam.GetFocalPoint())
        up = np.array(cam.GetViewUp())
        # Save camera distance so the view stays the same size
        cam_dist = np.linalg.norm(pos - foc)

        fwd = foc - pos
        fwd_len = np.linalg.norm(fwd)
        if fwd_len < 1e-12:
            return
        fwd /= fwd_len

        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up_ortho = np.cross(right, fwd)
        up_ortho /= np.linalg.norm(up_ortho)

        # Rotation that maps current camera frame → world axes:
        #   right    → +X
        #   up_ortho → +Y
        #   -fwd     → +Z  (camera looks down -Z in default pose)
        R = np.eye(4)
        R[0, :3] = right
        R[1, :3] = up_ortho
        R[2, :3] = -fwd

        # Pivot = current focal point (the center of what the user is
        # looking at).  This keeps the model centred after rotation.
        pivot = foc.copy()

        # Build: translate to pivot → rotate → translate back
        T_neg = np.eye(4)
        T_neg[:3, 3] = -pivot
        T_pos = np.eye(4)
        T_pos[:3, 3] = pivot
        M = T_pos @ R @ T_neg

        # Compose with any existing model transform
        existing = np.eye(4)
        if hasattr(self, "_upright_matrix"):
            existing = self._upright_matrix
        composed = M @ existing
        self._upright_matrix = composed

        # Apply to VTK
        mat4 = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mat4.SetElement(i, j, composed[i, j])
        t = vtk.vtkTransform()
        t.SetMatrix(mat4)

        if self._point_actor is not None:
            self._point_actor.SetUserTransform(t)
        for actor in self._camera_actors:
            actor.SetUserTransform(t)

        # Place camera at the default front view (looking along -Z),
        # centred on the pivot at the same distance the user had.
        # This avoids ResetCamera() which produces unpredictable zoom.
        cam.SetFocalPoint(pivot[0], pivot[1], pivot[2])
        cam.SetPosition(pivot[0], pivot[1], pivot[2] + cam_dist)
        cam.SetViewUp(0, 1, 0)
        self._renderer.ResetCameraClippingRange()
        self._applied_roll = 0.0
        # Update gizmo guard so it tracks this zoom level
        self._guard_dist = cam_dist
        self._guard_dir = np.array([0.0, 0.0, 1.0])
        self.render()

    def reset_upright(self):
        """Remove the upright transform, restoring original COLMAP coords."""
        import vtk

        if self._destroyed:
            return
        if hasattr(self, "_upright_matrix"):
            del self._upright_matrix
        identity = vtk.vtkTransform()
        identity.Identity()
        if self._point_actor is not None:
            self._point_actor.SetUserTransform(identity)
        for actor in self._camera_actors:
            actor.SetUserTransform(identity)
        cam = self._renderer.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0, 0, 1)
        cam.SetViewUp(0, 1, 0)
        self._renderer.ResetCamera()
        self._applied_roll = 0.0
        self.render()

    # ── Upright persistence ────────────────────────────────────────

    _UPRIGHT_FILENAME = "upright_transform.json"

    def save_upright(self, model_dir: str):
        """Save the current upright matrix to model_dir/upright_transform.json."""
        import json
        from pathlib import Path
        if not hasattr(self, "_upright_matrix"):
            return
        path = Path(model_dir) / self._UPRIGHT_FILENAME
        data = {"matrix_4x4": self._upright_matrix.tolist()}
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved upright transform: %s", path)

    def load_upright(self, model_dir: str) -> bool:
        """Load an upright matrix from model_dir/upright_transform.json.

        Returns True if a transform was loaded and applied.
        """
        import json
        from pathlib import Path
        path = Path(model_dir) / self._UPRIGHT_FILENAME
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._upright_matrix = np.array(data["matrix_4x4"])
            self._reapply_upright()
            self.render()
            logger.info("Loaded upright transform: %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load upright transform: %s", e)
            return False

    # ── Camera roll ────────────────────────────────────────────────

    def set_roll(self, angle: float):
        """Set the camera roll to an absolute slider angle in degrees.

        Applies only the delta since the last set_roll() call so that
        orbiting between adjustments is preserved.
        """
        if self._destroyed:
            return
        delta = angle - self._applied_roll
        self._applied_roll = angle
        self._renderer.GetActiveCamera().Roll(delta)
        self.render()

    # ── Public controls ──────────────────────────────────────────────

    def set_color_mode(self, mode: str):
        """Switch point coloring mode."""
        if mode not in self.COLOR_MODES or self._model_data is None:
            return
        self._color_mode = mode
        if hasattr(self, "_point_polydata"):
            self._apply_color_to_polydata(self._point_polydata)
            self._point_polydata.Modified()
        self.render()

    def set_point_size(self, size: float):
        """Change point render size."""
        self._point_size = max(1.0, size)
        if self._point_actor is not None:
            self._point_actor.GetProperty().SetPointSize(self._point_size)
        self.render()

    def toggle_cameras(self, show: bool):
        """Show or hide camera actors."""
        self._show_cameras = show
        if self._model_data is not None:
            images = self._model_data.get("images", {})
            cameras = self._model_data.get("cameras", {})
            self._build_cameras(images, cameras)
            self._reapply_upright()
        self.render()

    def reset_camera(self):
        """Reset view to default front pose, including roll."""
        if self._destroyed:
            return
        cam = self._renderer.GetActiveCamera()
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0, 0, 1)
        cam.SetViewUp(0, 1, 0)
        self._renderer.ResetCamera()
        self._applied_roll = 0.0
        self.render()

    def screenshot(self, path: str):
        """Save current view as PNG."""
        if self._destroyed:
            return
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter

        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self._render_window)
        w2i.Update()

        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        logger.info("Screenshot saved: %s", path)

    # ── Cleanup ──────────────────────────────────────────────────────

    def destroy(self):
        """Clean up all VTK resources. All methods become no-ops after this."""
        if self._destroyed:
            return
        self._destroyed = True

        # 1. Cancel the event pump
        self.pause_pump()

        # 2. Terminate interactor
        try:
            self._interactor.TerminateApp()
        except Exception:
            pass

        # 3. Finalize render window
        try:
            self._render_window.Finalize()
        except Exception:
            pass

        # 4. Close pyvista plotter
        try:
            self._pv_plotter.close()
        except Exception:
            pass

        logger.info("PointCloudViewer destroyed")
