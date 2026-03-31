"""
Coverage Tab — Gap detection and bridge extraction.

Ported from panoex_gui.py into a scrollable tab for Reconstruction Zone.
All GUI state is stored on the *app* instance.  Bridge extraction shares
video/output state from the Extract tab (no duplicate entries).

Usage::

    from tabs.gaps_tab import build_gaps_tab
    build_gaps_tab(app, tab_frame)
"""

import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from widgets import CollapsibleSection, Tooltip

# ── prep360 core (optional) ────────────────────────────────────────────

try:
    from prep360.core import (
        GapDetector, GapReport, SpatialGap,
        BridgeExtractor, BridgeResult,
        FisheyeReframer, FISHEYE_PRESETS,
    )

    HAS_PREP360 = True
except ImportError:
    HAS_PREP360 = False


# ======================================================================
# Public entry point
# ======================================================================

def build_gaps_tab(app, parent):
    """Populate *parent* (the "Gaps" tab frame) with gap analysis and
    bridge extraction sections.  All widget references are stored on *app*.
    """
    scroll = ctk.CTkScrollableFrame(parent)
    scroll.pack(fill="both", expand=True)

    _build_gap_analysis_section(app, scroll)
    _build_bridge_section(app, scroll)

    app._last_gap_report = None


# ======================================================================
#  1. GAP ANALYSIS
# ======================================================================

def _build_gap_analysis_section(app, parent):
    sec = CollapsibleSection(parent, "Gap Analysis", expanded=True)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    c = sec.content

    # Source type radio buttons
    app.gaps_source_var = ctk.StringVar(value="colmap")
    src_radios = ctk.CTkFrame(c, fg_color="transparent")
    src_radios.pack(fill="x", pady=3, padx=12)
    radio_colmap = ctk.CTkRadioButton(src_radios, text="COLMAP directory",
                       variable=app.gaps_source_var, value="colmap",
                       command=lambda: _on_source_change(app))
    radio_colmap.pack(anchor="w", pady=2)
    Tooltip(radio_colmap, "Load camera positions from a COLMAP sparse reconstruction.\nExpects a directory with cameras.bin, images.bin, points3D.bin.")
    radio_meta = ctk.CTkRadioButton(src_radios, text="Metashape XML",
                       variable=app.gaps_source_var, value="metashape",
                       command=lambda: _on_source_change(app))
    radio_meta.pack(anchor="w", pady=2)
    Tooltip(radio_meta, "Load camera positions from a Metashape cameras.xml export.\nExport via: File \u2192 Export \u2192 Export Cameras.")
    radio_xmp = ctk.CTkRadioButton(src_radios, text="XMP sidecars",
                       variable=app.gaps_source_var, value="xmp",
                       command=lambda: _on_source_change(app))
    radio_xmp.pack(anchor="w", pady=2)
    Tooltip(radio_xmp, "Load camera positions from XMP sidecar files.\nReads GPS coordinates embedded in .xmp files alongside images.")

    # Source path picker
    src_frame = ctk.CTkFrame(c, fg_color="transparent")
    src_frame.pack(fill="x", pady=3, padx=6)
    app.gaps_source_label = ctk.CTkLabel(src_frame, text="COLMAP:", width=65, anchor="w")
    app.gaps_source_label.pack(side="left")
    app.gaps_source_entry = ctk.CTkEntry(src_frame,
                                          placeholder_text="Select folder or file...")
    app.gaps_source_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(app.gaps_source_entry, "Path to your camera position data.\nBrowse to select the appropriate file or folder.")
    ctk.CTkButton(src_frame, text="...", width=36,
                  command=lambda: _browse_source(app)).pack(side="left")

    # Source images (optional)
    img_frame = ctk.CTkFrame(c, fg_color="transparent")
    img_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(img_frame, text="Images:", width=65, anchor="w").pack(side="left")
    app.gaps_images_entry = ctk.CTkEntry(img_frame,
                                          placeholder_text="Source images (optional)...")
    app.gaps_images_entry.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(app.gaps_images_entry, "Optional: path to source images for visual gap reports.\nUsed to show which images are near detected gaps.")
    ctk.CTkButton(img_frame, text="...", width=36,
                  command=lambda: app._browse_folder_for(app.gaps_images_entry)
                  ).pack(side="left")

    # Analysis parameters (collapsible)
    params = CollapsibleSection(c, "Analysis Parameters", expanded=False)
    params.pack(fill="x", padx=2, pady=(4, 0))
    pc = params.content

    eps_frame = ctk.CTkFrame(pc, fg_color="transparent")
    eps_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(eps_frame, text="Cluster eps:", width=90, anchor="w").pack(side="left")
    app.gaps_eps_var = ctk.DoubleVar(value=5.0)
    app.gaps_eps_label = ctk.CTkLabel(eps_frame, text="5.0", width=35,
                                      font=("Consolas", 11))
    app.gaps_eps_label.pack(side="right")
    eps_slider = ctk.CTkSlider(eps_frame, from_=1.0, to=20.0, variable=app.gaps_eps_var,
                  command=lambda v: app.gaps_eps_label.configure(text=f"{float(v):.1f}"))
    eps_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(eps_slider, "DBSCAN clustering distance threshold.\nSmaller = tighter clusters, may split continuous coverage.\nLarger = looser clusters, may merge distinct areas.")

    sparse_frame = ctk.CTkFrame(pc, fg_color="transparent")
    sparse_frame.pack(fill="x", pady=3, padx=6)
    ctk.CTkLabel(sparse_frame, text="Sparse thresh:", width=90, anchor="w").pack(side="left")
    app.gaps_sparse_var = ctk.DoubleVar(value=2.0)
    app.gaps_sparse_label = ctk.CTkLabel(sparse_frame, text="2.0x", width=35,
                                         font=("Consolas", 11))
    app.gaps_sparse_label.pack(side="right")
    sparse_slider = ctk.CTkSlider(sparse_frame, from_=1.0, to=5.0, variable=app.gaps_sparse_var,
                  command=lambda v: app.gaps_sparse_label.configure(text=f"{float(v):.1f}x"))
    sparse_slider.pack(side="left", fill="x", expand=True, padx=(6, 4))
    Tooltip(sparse_slider, "How sparse a region must be to count as a gap.\nMultiplier of median point density.\nHigher = only flags very sparse areas.")

    # Analyze button
    app.gaps_analyze_btn = ctk.CTkButton(
        c, text="Analyze Gaps", command=lambda: _run_gap_analysis(app),
        fg_color="#1565C0", hover_color="#0D47A1",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.gaps_analyze_btn.pack(fill="x", pady=(6, 4), padx=6)
    Tooltip(app.gaps_analyze_btn, "Run gap analysis on loaded camera positions.\nIdentifies spatial gaps in your reconstruction coverage.")

    # Summary + detail textboxes
    app.gaps_summary_text = ctk.CTkTextbox(
        c, height=100, font=ctk.CTkFont(family="Consolas", size=11),
        fg_color="#1a1a1a", state="disabled",
    )
    app.gaps_summary_text.pack(fill="x", padx=6, pady=(4, 2))

    app.gaps_detail_text = ctk.CTkTextbox(
        c, height=120, font=ctk.CTkFont(family="Consolas", size=11),
        fg_color="#1a1a1a", state="disabled",
    )
    app.gaps_detail_text.pack(fill="x", padx=6, pady=(2, 4))


# ── gap helpers ───────────────────────────────────────────────────────

def _on_source_change(app):
    source = app.gaps_source_var.get()
    labels = {"colmap": "COLMAP:", "metashape": "XML:", "xmp": "XMP Dir:"}
    placeholders = {
        "colmap": "COLMAP output directory...",
        "metashape": "Metashape cameras.xml...",
        "xmp": "XMP sidecar directory...",
    }
    app.gaps_source_label.configure(text=labels[source])
    app.gaps_source_entry.delete(0, "end")
    app.gaps_source_entry.configure(placeholder_text=placeholders[source])


def _browse_source(app):
    source = app.gaps_source_var.get()
    if source == "metashape":
        path = filedialog.askopenfilename(
            title="Select Metashape cameras.xml",
            filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")],
        )
        if path:
            app.gaps_source_entry.delete(0, "end")
            app.gaps_source_entry.insert(0, path)
    else:
        app._browse_folder_for(app.gaps_source_entry)


# ── gap analysis worker ──────────────────────────────────────────────

def _run_gap_analysis(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    source_path = app.gaps_source_entry.get()
    if not source_path:
        app.log("Error: Please select a camera source")
        return
    if not Path(source_path).exists():
        app.log(f"Error: Path not found: {source_path}")
        return
    app.gaps_analyze_btn.configure(state="disabled")
    threading.Thread(target=_gap_analysis_worker, args=(app, source_path),
                     daemon=True).start()


def _gap_analysis_worker(app, source_path):
    try:
        source_type = app.gaps_source_var.get()
        images_dir = app.gaps_images_entry.get() or None

        detector = GapDetector(
            cluster_eps=app.gaps_eps_var.get(),
            sparse_threshold=app.gaps_sparse_var.get(),
        )

        app.log(f"Analyzing gaps ({source_type}: {Path(source_path).name})...")

        kwargs = {}
        if source_type == "colmap":
            kwargs["colmap_dir"] = source_path
        elif source_type == "metashape":
            kwargs["metashape_xml"] = source_path
        elif source_type == "xmp":
            kwargs["xmp_dir"] = source_path

        if images_dir:
            kwargs["source_images_dir"] = images_dir

        report = detector.analyze(**kwargs, log=app.log)
        app._last_gap_report = report

        summary = report.summary()
        app.log(f"\n{summary}")

        # Detail text
        detail_lines = []
        for i, gap in enumerate(report.gaps):
            detail_lines.append(f"Gap {i}: {gap.gap_type}")
            nearby = gap.cameras_before + gap.cameras_after
            detail_lines.append(f"  Cameras: {', '.join(nearby[:5])}")
            if len(nearby) > 5:
                detail_lines.append(f"  ... and {len(nearby) - 5} more")
            detail_lines.append(f"  Extent: {gap.extent:.2f}")
            detail_lines.append(f"  Severity: {gap.severity:.2f}")
            if gap.estimated_timestamps:
                t_start, t_end = gap.estimated_timestamps
                detail_lines.append(f"  Timestamps: {t_start:.1f}s - {t_end:.1f}s")
            detail_lines.append("")

        if not report.gaps:
            detail_lines.append(
                "No spatial gaps detected \u2014 reconstruction looks well-connected.")

        detail_text = "\n".join(detail_lines)

        app.after(0, lambda: app._set_textbox(app.gaps_summary_text, summary))
        app.after(0, lambda: app._set_textbox(app.gaps_detail_text, detail_text))

        # Auto-expand bridge section if gaps found
        if report.gaps and hasattr(app, "gaps_bridge_section"):
            app.after(0, lambda: app.gaps_bridge_section.expand())
            app.after(0, lambda: _refresh_bridge_info(app))

    except Exception as e:
        app.log(f"Gap analysis error: {e}")
    finally:
        app.after(0, lambda: app.gaps_analyze_btn.configure(state="normal"))


# ======================================================================
#  2. BRIDGE EXTRACTION
# ======================================================================

def _build_bridge_section(app, parent):
    sec = CollapsibleSection(parent, "Bridge Extraction", expanded=False)
    sec.pack(fill="x", pady=(0, 6), padx=4)
    app.gaps_bridge_section = sec
    c = sec.content

    # Shared-state info — reads video/output from Extract tab
    app.gaps_bridge_info = ctk.CTkLabel(
        c, text="Set video and output on the Extract tab",
        font=ctk.CTkFont(family="Consolas", size=10),
        text_color="#9ca3af", anchor="w", justify="left",
    )
    app.gaps_bridge_info.pack(fill="x", padx=12, pady=(4, 2))

    # Reframe toggle (Coverage-specific; reads fisheye config from Extract)
    app.gaps_reframe_var = ctk.BooleanVar(value=False)
    reframe_cb = ctk.CTkCheckBox(c, text="Reframe fisheye \u2192 perspective",
                    variable=app.gaps_reframe_var)
    reframe_cb.pack(pady=3, padx=12, anchor="w")
    Tooltip(reframe_cb, "Reframe extracted bridge frames from fisheye to perspective.\nUses the fisheye preset configured on the Extract tab.")

    # Buttons
    app.gaps_bridge_btn = ctk.CTkButton(
        c, text="Extract Bridges", command=lambda: _run_bridge(app),
        fg_color="#2E7D32", hover_color="#1B5E20",
        font=ctk.CTkFont(size=13, weight="bold"), height=38,
    )
    app.gaps_bridge_btn.pack(fill="x", pady=(6, 4), padx=6)
    Tooltip(app.gaps_bridge_btn, "Extract frames from the source video at gap timestamps.\nFills spatial gaps identified by the gap analysis.")

    app.gaps_stop_btn = ctk.CTkButton(
        c, text="Stop", command=app.stop_operation,
        fg_color="#C62828", hover_color="#8B0000", height=32,
    )
    # hidden initially


# ── bridge worker ─────────────────────────────────────────────────────

def _refresh_bridge_info(app):
    """Update the bridge info label from Extract tab state."""
    video = getattr(app, "current_video_path", None)
    output = app.extract_output_entry.get() if hasattr(app, "extract_output_entry") else ""
    parts = []
    if video:
        parts.append(f"Video: {Path(video).name}")
    if output:
        parts.append(f"Output: {Path(output).name}/bridges/")
    if parts:
        app.gaps_bridge_info.configure(text="\n".join(parts))
    else:
        app.gaps_bridge_info.configure(text="Set video and output on the Extract tab")


def _run_bridge(app):
    if not HAS_PREP360:
        app.log("Error: prep360 core not available")
        return
    if app._last_gap_report is None:
        app.log("Error: Run gap analysis first")
        return

    # Read video/output from Extract tab
    video_path = getattr(app, "current_video_path", None)
    base_output = app.extract_output_entry.get() if hasattr(app, "extract_output_entry") else ""

    if not video_path:
        app.log("Error: No video loaded — analyze a video on the Extract tab first")
        return
    if not Path(video_path).exists():
        app.log(f"Error: Video not found: {video_path}")
        return
    if not base_output:
        app.log("Error: No output folder — set it on the Extract tab")
        return

    output_dir = str(Path(base_output) / "bridges")
    app._start_operation(app.gaps_bridge_btn, app.gaps_stop_btn)
    threading.Thread(target=_bridge_worker, args=(app, video_path, output_dir),
                     daemon=True).start()


def _bridge_worker(app, video_path, output_dir):
    try:
        extractor = BridgeExtractor()

        reframe = app.gaps_reframe_var.get()
        reframer = None
        view_config = None

        if reframe:
            try:
                # Get fisheye calibration + preset from Source tab settings
                from tabs.source_tab import _get_fisheye_calibration, _get_preset_key
                calib = _get_fisheye_calibration(app)
                reframer = FisheyeReframer(calib)
                preset_key = _get_preset_key(app)
                view_config = FISHEYE_PRESETS.get(preset_key)
                app.log(f"Reframing enabled: {preset_key}")
            except Exception as e:
                app.log(f"Warning: Cannot reframe ({e}), extracting raw frames")
                reframe = False

        def progress(msg):
            if not app.cancel_flag.is_set():
                app.log(f"  {msg}")

        result = extractor.extract(
            gap_report=app._last_gap_report,
            video_path=video_path,
            output_dir=output_dir,
            reframe=reframe,
            reframer=reframer,
            view_config=view_config,
            progress_callback=progress,
            log=app.log,
        )

        summary = result.summary()
        app.log(f"\n{summary}")
        app.after(0, lambda: app._set_textbox(app.gaps_detail_text, summary))

        report_path = str(Path(output_dir) / "bridge_result.json")
        result.save(report_path)
        app.log(f"Report saved: {report_path}")

    except Exception as e:
        app.log(f"Bridge extraction error: {e}")
    finally:
        app.after(0, lambda: app._stop_operation(
            app.gaps_bridge_btn, app.gaps_stop_btn))
