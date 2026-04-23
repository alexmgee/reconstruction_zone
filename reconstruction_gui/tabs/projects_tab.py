"""
Projects Tab -- Central registry for photogrammetry projects.

Left column (tab content): "New Project" form / "Current Projects" list.
Right column (_proj_right_panel): project info panel + directory tree + thumbnail viewer.
All interactions are inline -- no popout dialogs.
"""

import os
import threading
import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from widgets import (
    Section, CollapsibleSection, Tooltip,
    LABEL_FIELD_WIDTH,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ======================================================================
#  Entry point
# ======================================================================

def build_projects_tab(app, parent):
    """Build the Projects tab UI. Called once during app init.

    Left column (tab content): "New Project" form / "Current Projects" list.
    Right column (_proj_right_panel): project info panel + directory tree + thumbnail viewer.
    """
    app._selected_project_id = None
    app._proj_edit_mode = False

    # Initialize store
    from project_store import ProjectStore
    store_path = app._prefs.get("tracker_store_path", "D:\\tracker.json")
    app._project_store = ProjectStore(store_path)

    # -- Left column: mode toggle + mode content frames --
    _build_left_column(app, parent)

    # -- Right column: info + viewer panel (lives in _main_frame column 1) --
    build_project_right_panel(app)

    # Initial list population (deferred so UI is built first)
    parent.after(100, lambda: _refresh_current_projects_list(app))


# ======================================================================
#  Left column: mode toggle + content
# ======================================================================

def _build_left_column(app, parent):
    """Left column: CTkSegmentedButton toggling New Project / Current Projects."""
    # Mode toggle at top
    app._proj_mode_var = tk.StringVar(value="New Project")
    toggle = ctk.CTkSegmentedButton(
        parent,
        values=["New Project", "Current Projects"],
        variable=app._proj_mode_var,
        command=lambda val: _switch_left_mode(app, val),
    )
    toggle.pack(fill="x", padx=6, pady=(6, 4))

    # Container for mode content (only one visible at a time)
    app._proj_new_frame = ctk.CTkFrame(parent, fg_color="transparent")
    app._proj_current_frame = ctk.CTkFrame(parent, fg_color="transparent")

    # Form state for new project (accumulates before Create)
    app._proj_form_locations = []   # [(label, path, derived_from)]
    app._proj_form_sources = []     # [(label, path, media_type)]
    app._proj_form_methods = []     # [(tool_name, path)]
    app._proj_form_notes = []       # [str]
    app._proj_form_tags = []        # [str]

    _build_new_project_mode(app)
    _build_current_projects_mode(app)

    # Show default mode
    app._proj_new_frame.pack(fill="both", expand=True)

    # Store path at bottom
    settings_frame = ctk.CTkFrame(parent, fg_color="transparent")
    settings_frame.pack(fill="x", padx=6, pady=(4, 6), side="bottom")
    ctk.CTkLabel(
        settings_frame, text="Store:", font=("Consolas", 9), text_color="#6b7280",
    ).pack(side="left")
    ctk.CTkLabel(
        settings_frame,
        text=str(app._project_store.store_path) if app._project_store else "Not set",
        font=("Consolas", 9), text_color="#6b7280",
    ).pack(side="left", padx=(4, 0))


def _switch_left_mode(app, mode):
    """Switch between New Project and Current Projects modes."""
    if mode == "New Project":
        app._proj_current_frame.pack_forget()
        app._proj_new_frame.pack(fill="both", expand=True)
    else:
        app._proj_new_frame.pack_forget()
        app._proj_current_frame.pack(fill="both", expand=True)
        _refresh_current_projects_list(app)


# ======================================================================
#  New Project mode
# ======================================================================

def _build_new_project_mode(app):
    """New Project form: layout matching Extract/Mask/Review tab hierarchy."""
    scroll = ctk.CTkScrollableFrame(app._proj_new_frame)
    scroll.pack(fill="both", expand=True)
    app._proj_new_scroll = scroll

    # ── Project Setup section ─────────────────────────────────────────
    setup_sec = Section(scroll, "Project Setup")
    setup_sec.pack(fill="x", pady=(0, 6), padx=4)
    c = setup_sec.content

    # Title
    ctk.CTkLabel(c, text="Title:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(4, 0))
    app._proj_new_title = ctk.CTkEntry(c, height=28)
    app._proj_new_title.pack(fill="x", padx=6, pady=(0, 4))
    app._proj_new_title.insert(0, datetime.now().strftime("%Y-%m-%d Untitled"))

    # Root Directory
    ctk.CTkLabel(c, text="Root:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(2, 0))
    dir_row = ctk.CTkFrame(c, fg_color="transparent")
    dir_row.pack(fill="x", padx=6, pady=(0, 4))
    app._proj_new_dir = ctk.CTkEntry(dir_row, placeholder_text="Select project root...", height=28)
    app._proj_new_dir.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _browse_root():
        p = filedialog.askdirectory(title="Select project root directory")
        if p:
            app._proj_new_dir.delete(0, "end")
            app._proj_new_dir.insert(0, p)

    ctk.CTkButton(dir_row, text="...", width=36, height=28, command=_browse_root).pack(side="right")

    # Location (working directories)
    ctk.CTkLabel(c, text="Location:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(2, 0))
    app._proj_form_loc_display = ctk.CTkFrame(c, fg_color="transparent")
    # Not packed — only packed by _rebuild_form_list when items exist
    loc_add_row = ctk.CTkFrame(c, fg_color="transparent")
    loc_add_row.pack(fill="x", padx=6, pady=(0, 4))
    loc_label_entry = ctk.CTkEntry(loc_add_row, placeholder_text="Label...", height=28)
    loc_label_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_location():
        label = loc_label_entry.get().strip()
        if not label:
            return
        path = filedialog.askdirectory(title="Select Working Directory")
        if not path:
            return
        app._proj_form_locations.append((label, path, ""))
        loc_label_entry.delete(0, "end")
        _rebuild_form_list(app, app._proj_form_loc_display, app._proj_form_locations, "loc")

    ctk.CTkButton(loc_add_row, text="...", width=36, height=28,
                  command=_add_location).pack(side="right")

    # Sources
    ctk.CTkLabel(c, text="Sources:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(2, 0))
    app._proj_form_src_display = ctk.CTkFrame(c, fg_color="transparent")
    src_add_row = ctk.CTkFrame(c, fg_color="transparent")
    src_add_row.pack(fill="x", padx=6, pady=(0, 4))
    src_label_entry = ctk.CTkEntry(src_add_row, placeholder_text="Label dataset...", height=28)
    src_label_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_source():
        label = src_label_entry.get().strip()
        if not label:
            return
        path = filedialog.askdirectory(title="Select Source Directory")
        if not path:
            return
        app._proj_form_sources.append((label, path, "images"))
        src_label_entry.delete(0, "end")
        _rebuild_form_list(app, app._proj_form_src_display, app._proj_form_sources, "src")

    ctk.CTkButton(src_add_row, text="...", width=36, height=28,
                  command=_add_source).pack(side="right")

    # Method
    ctk.CTkLabel(c, text="Method:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(2, 0))
    app._proj_form_method_display = ctk.CTkFrame(c, fg_color="transparent")
    method_btn_row = ctk.CTkFrame(c, fg_color="transparent")
    method_btn_row.pack(fill="x", padx=6, pady=(0, 4))

    def _add_method(tool_name):
        if tool_name == "Metashape":
            path = filedialog.askopenfilename(
                title="Select Metashape Project",
                filetypes=[("Metashape Project", "*.psx"), ("All Files", "*.*")])
        else:
            path = filedialog.askdirectory(title=f"Select {tool_name} folder")
        if not path:
            return
        app._proj_form_methods.append((tool_name, path))
        _rebuild_form_list(app, app._proj_form_method_display, app._proj_form_methods, "method")

    for tool in ("Metashape", "RealityScan", "COLMAP"):
        ctk.CTkButton(method_btn_row, text=tool, width=90, height=28,
                      fg_color="#343638",
                      command=lambda t=tool: _add_method(t)).pack(side="left", padx=(0, 4))

    # ── Notes & Tags section ──────────────────────────────────────────
    nt_sec = CollapsibleSection(scroll, "Notes & Tags", expanded=False)
    nt_sec.pack(fill="x", pady=(0, 6), padx=4)
    nt = nt_sec.content

    # Notes
    ctk.CTkLabel(nt, text="Notes:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(4, 0))
    app._proj_form_notes_display = ctk.CTkFrame(nt, fg_color="transparent")
    notes_add_row = ctk.CTkFrame(nt, fg_color="transparent")
    notes_add_row.pack(fill="x", padx=6, pady=(0, 4))
    notes_entry = ctk.CTkEntry(notes_add_row, placeholder_text="Add a note...", height=28)
    notes_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_note():
        text = notes_entry.get().strip()
        if not text:
            return
        app._proj_form_notes.append(text)
        notes_entry.delete(0, "end")
        _rebuild_form_list(app, app._proj_form_notes_display, app._proj_form_notes, "note")

    ctk.CTkButton(notes_add_row, text="+", width=36, height=28,
                  command=_add_note).pack(side="right")
    notes_entry.bind("<Return>", lambda e: _add_note())

    # Tags
    ctk.CTkLabel(nt, text="Tags:", anchor="w").pack(
        side="top", anchor="w", padx=6, pady=(2, 0))
    app._proj_form_tags_display = ctk.CTkFrame(nt, fg_color="transparent")
    tags_add_row = ctk.CTkFrame(nt, fg_color="transparent")
    tags_add_row.pack(fill="x", padx=6, pady=(0, 4))
    tags_entry = ctk.CTkEntry(tags_add_row, placeholder_text="Add tag...", height=28)
    tags_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_tag():
        text = tags_entry.get().strip()
        if not text or text in app._proj_form_tags:
            return
        app._proj_form_tags.append(text)
        tags_entry.delete(0, "end")
        _rebuild_form_list(app, app._proj_form_tags_display, app._proj_form_tags, "tag")

    ctk.CTkButton(tags_add_row, text="+", width=36, height=28,
                  command=_add_tag).pack(side="right")
    tags_entry.bind("<Return>", lambda e: _add_tag())

    # ── Create Project button ─────────────────────────────────────────
    ctk.CTkButton(
        scroll, text="Create Project", height=34,
        font=ctk.CTkFont(size=13, weight="bold"),
        command=lambda: _on_create_project(app),
    ).pack(fill="x", padx=8, pady=(2, 6))

    # ── Scan for Projects section ─────────────────────────────────────
    scan_sec = CollapsibleSection(
        scroll, "Scan for Projects",
        subtitle="discover existing projects on disk",
        expanded=False,
    )
    scan_sec.pack(fill="x", pady=(0, 6), padx=4)
    sc = scan_sec.content

    scan_dir_row = ctk.CTkFrame(sc, fg_color="transparent")
    scan_dir_row.pack(fill="x", padx=6, pady=(4, 4))
    scan_entry = ctk.CTkEntry(scan_dir_row, placeholder_text="Directory to scan...")
    scan_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    scan_entry.insert(0, "D:\\")

    def _browse_scan():
        p = filedialog.askdirectory(title="Select root to scan")
        if p:
            scan_entry.delete(0, "end")
            scan_entry.insert(0, p)

    ctk.CTkButton(scan_dir_row, text="...", width=36, command=_browse_scan).pack(side="right")

    scan_btn = ctk.CTkButton(
        sc, text="Scan", height=30,
        font=ctk.CTkFont(size=12, weight="bold"),
        fg_color="#343638",
    )
    scan_btn.pack(fill="x", padx=6, pady=(0, 4))

    scan_status = ctk.CTkLabel(sc, text="", text_color="#9ca3af",
                               font=("Consolas", 9))
    scan_status.pack(fill="x", padx=6, pady=(0, 2))

    scan_results_frame = ctk.CTkScrollableFrame(sc, height=150)
    scan_results_frame.pack(fill="x", padx=4, pady=(0, 4))

    def _run_scan():
        from project_scanner import scan_directory

        root = scan_entry.get().strip()
        if not root:
            return

        for w in scan_results_frame.winfo_children():
            w.destroy()
        scan_status.configure(text="Scanning...")
        scan_btn.configure(state="disabled")

        def _thread():
            results = scan_directory(
                root, max_depth=5,
                progress_callback=lambda msg: scroll.after(
                    0, lambda m=msg: scan_status.configure(text=m)),
            )
            scroll.after(0, lambda: _show_scan_results(
                app, scan_results_frame, scan_status, scan_btn, results))

        threading.Thread(target=_thread, daemon=True).start()

    scan_btn.configure(command=_run_scan)


def _show_scan_results(app, container, status_label, scan_btn, results):
    """Show scan results with Import buttons that populate the New Project form."""
    results.sort(key=lambda r: (len(r.tool_files) == 0, r.suggested_title().lower()))
    total = len(results)
    status_label.configure(text=f"Found {total} projects")
    scan_btn.configure(state="normal")

    for res in results[:50]:
        row = ctk.CTkFrame(container, fg_color="#333333", corner_radius=4)
        row.pack(fill="x", pady=(0, 2))

        # Title + tools
        title = res.suggested_title()
        tool_names = sorted(set(res.tool_files.keys()))
        display = title
        if tool_names:
            display += f"  ({', '.join(tool_names)})"

        ctk.CTkLabel(
            row, text=display,
            font=("Consolas", 9), anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=6, pady=2)

        def _import(r=res):
            # Populate form fields
            app._proj_new_title.delete(0, "end")
            app._proj_new_title.insert(0, r.suggested_title())
            app._proj_new_dir.delete(0, "end")
            app._proj_new_dir.insert(0, r.root_dir)

            # Populate sources
            app._proj_form_sources.clear()
            for img_dir in r.image_dirs:
                label = Path(img_dir).name
                count = r.image_counts.get(img_dir, 0)
                app._proj_form_sources.append((label, img_dir, "images"))
            for vid in r.video_files:
                app._proj_form_sources.append((Path(vid).name, vid, "video"))
            _rebuild_form_list(app, app._proj_form_src_display,
                               app._proj_form_sources, "src")

            # Populate methods
            app._proj_form_methods.clear()
            for tool_name, tool_path in r.tool_files.items():
                app._proj_form_methods.append((tool_name, tool_path))
            _rebuild_form_list(app, app._proj_form_method_display,
                               app._proj_form_methods, "method")

            # Scroll to top
            app._proj_new_scroll._parent_canvas.yview_moveto(0)

        ctk.CTkButton(
            row, text="Import \u2191", width=60, height=20,
            font=("Consolas", 9), fg_color="#1f6aa5",
            command=_import,
        ).pack(side="right", padx=4, pady=2)

    if not results:
        ctk.CTkLabel(container, text="No projects found",
                     text_color="#9ca3af").pack(pady=10)


def _rebuild_form_list(app, container, items, kind):
    """Rebuild a form section's display of accumulated items."""
    for w in container.winfo_children():
        w.destroy()

    # Only pack the container when it has items; hide when empty
    if items:
        if not container.winfo_ismapped():
            container.pack(fill="x", padx=6, pady=(0, 2))
    else:
        container.pack_forget()
        return

    for i, item in enumerate(items):
        row = ctk.CTkFrame(container, fg_color="#333333", corner_radius=4)
        row.pack(fill="x", pady=(0, 2))

        if kind == "loc":
            label, path, derived = item
            ctk.CTkLabel(row, text=f"{label}: {Path(path).name}/",
                         font=ctk.CTkFont(size=11), anchor="w").pack(
                side="left", fill="x", expand=True, padx=6, pady=2)
        elif kind == "src":
            label, path, mtype = item
            ctk.CTkLabel(row, text=f"{label}: {Path(path).name}/",
                         font=ctk.CTkFont(size=11), anchor="w").pack(
                side="left", fill="x", expand=True, padx=6, pady=2)
        elif kind == "method":
            tool_name, path = item
            ctk.CTkLabel(row, text=f"{tool_name}: {Path(path).name}",
                         font=ctk.CTkFont(size=11), anchor="w").pack(
                side="left", fill="x", expand=True, padx=6, pady=2)
        elif kind == "note":
            ctk.CTkLabel(row, text=item, font=ctk.CTkFont(size=10),
                         anchor="w").pack(
                side="left", fill="x", expand=True, padx=6, pady=2)
        elif kind == "tag":
            ctk.CTkLabel(row, text=item, font=("Consolas", 9),
                         text_color="#9ca3af", anchor="w").pack(
                side="left", padx=6, pady=2)

        def _remove(idx=i):
            items.pop(idx)
            _rebuild_form_list(app, container, items, kind)

        ctk.CTkButton(row, text="\u2715", width=24, height=20,
                      font=ctk.CTkFont(size=9), fg_color="transparent",
                      hover_color="#dc2626", command=_remove).pack(side="right", padx=2)


def _on_create_project(app):
    """Create a new project from the form fields."""
    title = app._proj_new_title.get().strip()
    if not title:
        title = datetime.now().strftime("%Y-%m-%d Untitled")
    root_dir = app._proj_new_dir.get().strip()

    proj = app._project_store.create_project(title, root_dir=root_dir)

    # Add accumulated form items
    for label, path, media_type in app._proj_form_sources:
        proj.add_source(label, path, media_type)
    for tool_name, path in app._proj_form_methods:
        from project_store import ProjectSource
        proj.sources.append(ProjectSource(label=tool_name, path=path, media_type="other"))
    for label, path, derived in app._proj_form_locations:
        proj.add_work_dir(label, path, derived_from=derived)
    if app._proj_form_notes:
        proj.notes = "\n".join(app._proj_form_notes)
    if app._proj_form_tags:
        proj.tags = list(app._proj_form_tags)

    proj.updated_at = datetime.now().isoformat()
    app._project_store.save()

    # Reset form
    app._proj_new_title.delete(0, "end")
    app._proj_new_title.insert(0, datetime.now().strftime("%Y-%m-%d Untitled"))
    app._proj_new_dir.delete(0, "end")
    app._proj_form_locations.clear()
    app._proj_form_sources.clear()
    app._proj_form_methods.clear()
    app._proj_form_notes.clear()
    app._proj_form_tags.clear()
    _rebuild_form_list(app, app._proj_form_loc_display, app._proj_form_locations, "loc")
    _rebuild_form_list(app, app._proj_form_src_display, app._proj_form_sources, "src")
    _rebuild_form_list(app, app._proj_form_method_display, app._proj_form_methods, "method")
    _rebuild_form_list(app, app._proj_form_notes_display, app._proj_form_notes, "note")
    _rebuild_form_list(app, app._proj_form_tags_display, app._proj_form_tags, "tag")

    # Switch to Current Projects and select the new project
    app._selected_project_id = proj.id
    app._proj_mode_var.set("Current Projects")
    _switch_left_mode(app, "Current Projects")
    _populate_right_panel(app)


# ======================================================================
#  Scan Directory mode
# ======================================================================
#  Current Projects mode
# ======================================================================

def _build_current_projects_mode(app):
    """Current Projects: search + scrollable project list.

    Uses the same CTkScrollableFrame → Section structure as New Project
    so that the section headers land in identical positions.
    """
    f = app._proj_current_frame

    scroll = ctk.CTkScrollableFrame(f)
    scroll.pack(fill="both", expand=True)
    app._proj_current_scroll_outer = scroll

    sec = Section(scroll, "Project Library")
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    # Search row
    search_row = ctk.CTkFrame(c, fg_color="transparent")
    search_row.pack(fill="x", padx=6, pady=(4, 4))
    app._proj_search_var = tk.StringVar()
    app._proj_search_var.trace_add("write", lambda *_: _refresh_current_projects_list(app))
    ctk.CTkEntry(
        search_row, placeholder_text="Search projects...",
        height=28,
        textvariable=app._proj_search_var,
    ).pack(side="left", fill="x", expand=True, padx=(0, 8))
    app._proj_count_label = ctk.CTkLabel(
        search_row, text="", font=ctk.CTkFont(size=11), text_color="#9ca3af",
    )
    app._proj_count_label.pack(side="right")

    # Project cards are packed directly into the scroll frame (after the section)
    # by _refresh_current_projects_list — we point _proj_current_scroll here.
    app._proj_current_scroll = scroll
    app._proj_current_widgets = []



def build_project_right_panel(app):
    """Build the right column: info panel + file viewer (or empty state)."""
    panel = ctk.CTkFrame(app._main_frame, corner_radius=0)
    app._proj_right_panel = panel

    # Empty state (shown when no project selected)
    app._proj_empty_state = ctk.CTkFrame(panel, fg_color="transparent")
    app._proj_empty_state.pack(fill="both", expand=True)
    ctk.CTkLabel(
        app._proj_empty_state,
        text="Create or select a project\nto view details",
        text_color="#6b7280", font=ctk.CTkFont(size=13),
        justify="center",
    ).place(relx=0.5, rely=0.5, anchor="center")

    # Content frame (shown when project selected — populated by _populate_right_panel)
    app._proj_content_frame = ctk.CTkFrame(panel, fg_color="transparent")


def _populate_right_panel(app):
    """Populate the right panel with info + viewer for the selected project."""
    proj = app._project_store.get_project(app._selected_project_id) if app._selected_project_id else None
    if not proj:
        app._proj_content_frame.pack_forget()
        app._proj_empty_state.pack(fill="both", expand=True)
        return

    app._proj_empty_state.pack_forget()

    # Clear old content
    for w in app._proj_content_frame.winfo_children():
        w.destroy()

    app._proj_content_frame.pack(fill="both", expand=True)

    # Info panel (top) — edit mode or read-only
    if getattr(app, '_proj_edit_mode', False):
        _build_info_panel_edit(app, proj)
    else:
        _build_info_panel(app, proj)

    # Viewer panel (bottom) — tree + thumbnails
    _build_viewer_panel(app, proj)


# ======================================================================
#  Info panel (read-only + edit mode)
# ======================================================================

def _build_info_panel(app, proj):
    """Build the read-only info panel at the top of the right column."""
    from project_store import STAGE_ORDER
    from widgets import (
        COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
        COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
    )
    TOOL_LABELS = {"Metashape", "RealityScan", "COLMAP"}

    sec = Section(app._proj_content_frame, "Project Summary")
    sec.pack(fill="x", padx=4, pady=(4, 0))
    app._proj_info_panel = sec
    c = sec.content

    # Title + action buttons row
    title_row = ctk.CTkFrame(c, fg_color="transparent")
    title_row.pack(fill="x", padx=6, pady=(0, 4))
    ctk.CTkLabel(title_row, text=proj.title,
                 font=ctk.CTkFont(size=14, weight="bold"),
                 anchor="w").pack(side="left")

    def _on_edit():
        app._proj_edit_mode = not app._proj_edit_mode
        _populate_right_panel(app)

    # Buttons in top-right: Edit > Export > Delete
    ctk.CTkButton(
        title_row, text="Delete", width=55, height=24,
        fg_color=COLOR_ACTION_DANGER, hover_color=COLOR_ACTION_DANGER_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _on_delete_project(app, proj),
    ).pack(side="right", padx=(4, 0))
    ctk.CTkButton(
        title_row, text="Export", width=55, height=24,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=lambda: _do_export(app, proj),
    ).pack(side="right", padx=(4, 0))
    ctk.CTkButton(
        title_row, text="Edit", width=45, height=24,
        fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
        font=ctk.CTkFont(size=12),
        command=_on_edit,
    ).pack(side="right")

    from widgets import COLOR_ACTION_SECONDARY, COLOR_ACTION_MUTED, COLOR_TEXT_DIM, RADIUS_PILL

    # 2-column info grid — no fixed height, content-driven
    grid = ctk.CTkFrame(c, fg_color="transparent")
    grid.pack(fill="x", padx=6, pady=(0, 4))
    grid.grid_columnconfigure(0, weight=1, uniform="infocol")
    grid.grid_columnconfigure(1, weight=1, uniform="infocol")

    row_idx = 0

    # Row 0: Root | Sources
    _info_field(grid, row_idx, 0, "Root:",
                proj.root_dir or "(not set)")
    src_parts = []
    for s in proj.sources:
        if s.label not in TOOL_LABELS:
            count = f" ({s.file_count})" if s.file_count else ""
            src_parts.append(f"{s.label}{count}")
    _info_field(grid, row_idx, 1, "Sources:",
                " · ".join(src_parts) if src_parts else "(none)")
    row_idx += 1

    # Row 1: Location | Method + Stage
    loc_names = [wd.label for wd in proj.work_dirs]
    _info_field(grid, row_idx, 0, "Location:",
                " · ".join(loc_names) if loc_names else "(none)")

    method_frame = ctk.CTkFrame(grid, fg_color="transparent")
    method_frame.grid(row=row_idx, column=1, sticky="nw", padx=(8, 0), pady=(0, 4))
    tool_names = sorted(set(s.label for s in proj.sources if s.label in TOOL_LABELS))
    ctk.CTkLabel(method_frame, text="Method:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    ctk.CTkLabel(method_frame, text=" · ".join(tool_names) if tool_names else "(none)",
                 font=ctk.CTkFont(size=11), anchor="w",
                 wraplength=300).pack(anchor="w", padx=(12, 0))
    max_stage_idx = -1
    for wd in proj.work_dirs:
        try:
            si = STAGE_ORDER.index(wd.stage)
            max_stage_idx = max(max_stage_idx, si)
        except ValueError:
            pass
    if max_stage_idx >= 0:
        pills_frame = ctk.CTkFrame(method_frame, fg_color="transparent")
        pills_frame.pack(anchor="w", padx=(12, 0), pady=(4, 0))
        for si, sn in enumerate(STAGE_ORDER):
            fg = COLOR_ACTION_SECONDARY if si <= max_stage_idx else COLOR_ACTION_MUTED
            tc = "#ffffff" if si <= max_stage_idx else COLOR_TEXT_DIM
            ctk.CTkLabel(
                pills_frame, text=sn.replace("_", " ").title(),
                font=ctk.CTkFont(size=10), fg_color=fg, text_color=tc,
                corner_radius=RADIUS_PILL, height=20,
            ).pack(side="left", padx=(0, 3))
    row_idx += 1

    # Row 2: Tags | Notes
    all_tags = list(proj.tags)
    if all_tags:
        tag_frame = ctk.CTkFrame(grid, fg_color="transparent")
        tag_frame.grid(row=row_idx, column=0, sticky="nw", pady=(0, 4))
        ctk.CTkLabel(tag_frame, text="Tags:", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        tag_pills = ctk.CTkFrame(tag_frame, fg_color="transparent")
        tag_pills.pack(anchor="w", padx=(12, 0))
        for t in all_tags:
            ctk.CTkLabel(
                tag_pills, text=t,
                font=ctk.CTkFont(size=10),
                fg_color=COLOR_ACTION_MUTED, text_color="#9ca3af",
                corner_radius=RADIUS_PILL, height=20,
            ).pack(side="left", padx=(0, 4))
    else:
        _info_field(grid, row_idx, 0, "Tags:", "(none)")

    notes_list = [n for n in (proj.notes or "").split("\n") if n.strip()]
    notes_text = " · ".join(notes_list) if notes_list else "(none)"
    _info_field(grid, row_idx, 1, "Notes:", notes_text)


def _on_delete_project(app, proj):
    """Delete a project after confirmation."""
    result = messagebox.askyesno(
        "Delete Project",
        f"Permanently delete '{proj.title}'?\n\nThis cannot be undone.",
        icon="warning",
    )
    if result:
        app._project_store.delete_project(proj.id)
        app._selected_project_id = None
        _populate_right_panel(app)
        _refresh_current_projects_list(app)


def _info_field(grid, row, col, label, value):
    """Add a labeled field to the info grid."""
    frame = ctk.CTkFrame(grid, fg_color="transparent")
    frame.grid(row=row, column=col, sticky="nw", padx=(0 if col == 0 else 8, 0), pady=(0, 4))
    ctk.CTkLabel(frame, text=label, anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=11), anchor="w",
                 wraplength=300).pack(anchor="w", padx=(12, 0))


def _info_field_wide(grid, row, label, value):
    """Add a full-width labeled field spanning both columns."""
    frame = ctk.CTkFrame(grid, fg_color="transparent")
    frame.grid(row=row, column=0, columnspan=2, sticky="nw", pady=(0, 4))
    ctk.CTkLabel(frame, text=label, anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=11),
                 text_color="#9ca3af", anchor="w", wraplength=500).pack(anchor="w", padx=(12, 0))


def _build_info_panel_edit(app, proj):
    """Build the inline edit mode — mirrors the read-only layout with editable fields."""
    from project_store import ProjectSource, ProjectWorkDir, STAGE_ORDER
    from widgets import (
        COLOR_ACTION_PRIMARY, COLOR_ACTION_PRIMARY_H,
        COLOR_ACTION_SECONDARY, COLOR_ACTION_SECONDARY_H,
        COLOR_ACTION_DANGER, COLOR_ACTION_DANGER_H,
        COLOR_ACTION_MUTED, COLOR_ACTION_MUTED_H,
        COLOR_TEXT_MUTED, BROWSE_BUTTON_WIDTH, RADIUS_PILL,
    )
    TOOL_LABELS = {"Metashape", "RealityScan", "COLMAP"}

    sec = Section(app._proj_content_frame, "Project Summary")
    sec.pack(fill="x", padx=4, pady=(4, 0))
    app._proj_info_panel = sec
    c = sec.content

    # Title row + Done button (prominent green)
    title_row = ctk.CTkFrame(c, fg_color="transparent")
    title_row.pack(fill="x", padx=6, pady=(0, 4))
    title_entry = ctk.CTkEntry(title_row, font=ctk.CTkFont(size=14, weight="bold"))
    title_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
    title_entry.insert(0, proj.title)

    root_entry = None  # forward ref for _done closure

    def _done():
        new_title = title_entry.get().strip()
        if new_title and new_title != proj.title:
            proj.title = new_title
        if root_entry:
            new_root = root_entry.get().strip()
            if new_root != proj.root_dir:
                proj.root_dir = new_root
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        app._proj_edit_mode = False
        _populate_right_panel(app)
        _refresh_current_projects_list(app)

    ctk.CTkButton(
        title_row, text="Done", width=70, height=28,
        fg_color=COLOR_ACTION_PRIMARY, hover_color=COLOR_ACTION_PRIMARY_H,
        font=ctk.CTkFont(size=12, weight="bold"),
        command=_done,
    ).pack(side="right")

    # 2-column grid — same layout and fixed height as read-only
    grid = ctk.CTkFrame(c, fg_color="transparent", height=200)
    grid.pack(fill="x", padx=6, pady=(0, 4))
    grid.pack_propagate(False)
    grid.grid_columnconfigure(0, weight=1, uniform="editcol")
    grid.grid_columnconfigure(1, weight=1, uniform="editcol")

    row_idx = 0

    # Root (editable)
    root_frame = ctk.CTkFrame(grid, fg_color="transparent")
    root_frame.grid(row=row_idx, column=0, sticky="new", pady=(0, 4))
    ctk.CTkLabel(root_frame, text="Root:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    root_row = ctk.CTkFrame(root_frame, fg_color="transparent")
    root_row.pack(fill="x", padx=(12, 0))
    root_entry = ctk.CTkEntry(root_row, font=ctk.CTkFont(size=11))
    root_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    root_entry.insert(0, proj.root_dir or "")
    def _browse_root():
        p = filedialog.askdirectory(title="Select project root")
        if p:
            root_entry.delete(0, "end")
            root_entry.insert(0, p)
    ctk.CTkButton(root_row, text="...", width=BROWSE_BUTTON_WIDTH,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=_browse_root).pack(side="right")

    # Capture (read-only display)
    tags_as_capture = [t.upper() for t in proj.tags if t in {"drone", "dslr", "phone", "360"}]
    _info_field(grid, row_idx, 1, "Capture:",
                " · ".join(tags_as_capture) if tags_as_capture else "(none)")
    row_idx += 1

    # Location (with inline add)
    loc_frame = ctk.CTkFrame(grid, fg_color="transparent")
    loc_frame.grid(row=row_idx, column=0, sticky="new", pady=(0, 4))
    ctk.CTkLabel(loc_frame, text="Location:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    for idx, wd in enumerate(proj.work_dirs):
        wd_row = ctk.CTkFrame(loc_frame, fg_color="transparent")
        wd_row.pack(fill="x", padx=(12, 0), pady=(0, 1))
        desc = wd.label
        if wd.derived_from:
            desc += f" > {wd.derived_from}"
        exists = Path(wd.path).exists()
        ctk.CTkLabel(wd_row, text=desc, font=ctk.CTkFont(size=11),
                     text_color="#e0e0e0" if exists else "#ef4444",
                     anchor="w").pack(side="left")
        def _remove_wd(i=idx):
            proj.work_dirs.pop(i)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _populate_right_panel(app)
        ctk.CTkButton(wd_row, text="\u2715", width=20, height=18,
                      font=ctk.CTkFont(size=9), fg_color="transparent",
                      hover_color=COLOR_ACTION_DANGER,
                      command=_remove_wd).pack(side="right")
    # Add row
    loc_add = ctk.CTkFrame(loc_frame, fg_color="transparent")
    loc_add.pack(fill="x", padx=(12, 0), pady=(2, 0))
    loc_entry = ctk.CTkEntry(loc_add, placeholder_text="Label...", height=24,
                             font=ctk.CTkFont(size=10))
    loc_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    def _add_wd():
        label = loc_entry.get().strip()
        if not label:
            return
        path = filedialog.askdirectory(title="Select Working Directory")
        if not path:
            return
        proj.add_work_dir(label, path)
        app._project_store.save()
        _populate_right_panel(app)
    ctk.CTkButton(loc_add, text="...", width=BROWSE_BUTTON_WIDTH, height=24,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=_add_wd).pack(side="right")

    # Sources (with inline add)
    src_frame = ctk.CTkFrame(grid, fg_color="transparent")
    src_frame.grid(row=row_idx, column=1, sticky="new", padx=(8, 0), pady=(0, 4))
    ctk.CTkLabel(src_frame, text="Sources:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    media_sources = [(i, s) for i, s in enumerate(proj.sources) if s.label not in TOOL_LABELS]
    for idx, src in media_sources:
        sr = ctk.CTkFrame(src_frame, fg_color="transparent")
        sr.pack(fill="x", padx=(12, 0), pady=(0, 1))
        info = src.label
        if src.file_count:
            info += f" ({src.file_count})"
        exists = Path(src.path).exists()
        ctk.CTkLabel(sr, text=info, font=ctk.CTkFont(size=11),
                     text_color="#e0e0e0" if exists else "#ef4444",
                     anchor="w").pack(side="left")
        def _remove_src(i=idx):
            proj.sources.pop(i)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _populate_right_panel(app)
        ctk.CTkButton(sr, text="\u2715", width=20, height=18,
                      font=ctk.CTkFont(size=9), fg_color="transparent",
                      hover_color=COLOR_ACTION_DANGER,
                      command=_remove_src).pack(side="right")
    if not media_sources:
        ctk.CTkLabel(src_frame, text="(none)", font=ctk.CTkFont(size=11),
                     anchor="w").pack(anchor="w", padx=(12, 0))
    src_add = ctk.CTkFrame(src_frame, fg_color="transparent")
    src_add.pack(fill="x", padx=(12, 0), pady=(2, 0))
    src_entry = ctk.CTkEntry(src_add, placeholder_text="Label...", height=24,
                             font=ctk.CTkFont(size=10))
    src_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    def _add_src():
        label = src_entry.get().strip()
        if not label:
            return
        path = filedialog.askdirectory(title="Select Source Directory")
        if not path:
            return
        proj.add_source(label, path, "images")
        app._project_store.save()
        _populate_right_panel(app)
    ctk.CTkButton(src_add, text="...", width=BROWSE_BUTTON_WIDTH, height=24,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=_add_src).pack(side="right")
    row_idx += 1

    # Row 2: Method + stage pills (left) | Tags (right)
    method_frame = ctk.CTkFrame(grid, fg_color="transparent")
    method_frame.grid(row=row_idx, column=0, sticky="new", pady=(0, 4))
    ctk.CTkLabel(method_frame, text="Method:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    tool_sources = [s for s in proj.sources if s.label in TOOL_LABELS]
    for src in tool_sources:
        tr = ctk.CTkFrame(method_frame, fg_color="transparent")
        tr.pack(fill="x", padx=(12, 0), pady=(0, 1))
        ctk.CTkLabel(tr, text=src.label, font=ctk.CTkFont(size=11),
                     anchor="w").pack(side="left")
        def _remove_tool(s=src):
            proj.sources.remove(s)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _populate_right_panel(app)
        ctk.CTkButton(tr, text="\u2715", width=20, height=18,
                      font=ctk.CTkFont(size=9), fg_color="transparent",
                      hover_color=COLOR_ACTION_DANGER,
                      command=_remove_tool).pack(side="right")
    if not tool_sources:
        ctk.CTkLabel(method_frame, text="(none)", font=ctk.CTkFont(size=11),
                     anchor="w").pack(anchor="w", padx=(12, 0))
    tool_btns = ctk.CTkFrame(method_frame, fg_color="transparent")
    tool_btns.pack(fill="x", padx=(12, 0), pady=(4, 0))
    def _add_tool(tool_name):
        if tool_name == "Metashape":
            path = filedialog.askopenfilename(
                title="Select Metashape Project",
                filetypes=[("Metashape Project", "*.psx"), ("All Files", "*.*")])
        else:
            path = filedialog.askdirectory(title=f"Select {tool_name} folder")
        if not path:
            return
        proj.sources.append(ProjectSource(label=tool_name, path=path, media_type="other"))
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        _populate_right_panel(app)
    for tool in ("Metashape", "RealityScan", "COLMAP"):
        ctk.CTkButton(tool_btns, text=tool, width=80, height=22,
                      font=ctk.CTkFont(size=10),
                      fg_color=COLOR_ACTION_MUTED, hover_color=COLOR_ACTION_MUTED_H,
                      command=lambda t=tool: _add_tool(t)).pack(side="left", padx=(0, 4))
    # Stage pills under method (clickable)
    max_stage_idx = -1
    for wd in proj.work_dirs:
        try:
            si = STAGE_ORDER.index(wd.stage)
            max_stage_idx = max(max_stage_idx, si)
        except ValueError:
            pass
    for wd_idx, wd in enumerate(proj.work_dirs):
        wd_stage_row = ctk.CTkFrame(method_frame, fg_color="transparent")
        wd_stage_row.pack(fill="x", padx=(12, 0), pady=(4, 0))
        current_stage = wd.stage or ""
        try:
            current_stage_idx = STAGE_ORDER.index(current_stage)
        except ValueError:
            current_stage_idx = -1
        for si, sn in enumerate(STAGE_ORDER):
            fg = COLOR_ACTION_SECONDARY if si <= current_stage_idx else COLOR_ACTION_MUTED
            tc = "#ffffff" if si <= current_stage_idx else "#888888"
            ctk.CTkButton(
                wd_stage_row, text=sn.capitalize(), fg_color=fg,
                hover_color=COLOR_ACTION_SECONDARY, text_color=tc,
                width=60, height=20, font=ctk.CTkFont(size=9),
                command=lambda sn=sn, i=wd_idx: _cycle_work_dir_stage(app, proj.id, i, sn),
            ).pack(side="left", padx=(0, 3))

    # Tags (right column, editable)
    tag_frame = ctk.CTkFrame(grid, fg_color="transparent")
    tag_frame.grid(row=row_idx, column=1, sticky="new", padx=(8, 0), pady=(0, 4))
    ctk.CTkLabel(tag_frame, text="Tags:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    custom_tags = [t for t in proj.tags if t not in {"drone", "dslr", "phone", "360"}]
    if custom_tags:
        pills_row = ctk.CTkFrame(tag_frame, fg_color="transparent")
        pills_row.pack(fill="x", padx=(12, 0), pady=(2, 0))
        for tag in custom_tags:
            pill = ctk.CTkFrame(pills_row, fg_color=COLOR_ACTION_MUTED,
                                corner_radius=RADIUS_PILL, cursor="hand2")
            pill.pack(side="left", padx=(0, 4), pady=1)
            tag_lbl = ctk.CTkLabel(pill, text=tag, font=ctk.CTkFont(size=10),
                                   text_color="#e0e0e0")
            tag_lbl.pack(padx=6, pady=2)
            def _remove_tag(t=tag):
                proj.tags = [x for x in proj.tags if x != t]
                proj.updated_at = datetime.now().isoformat()
                app._project_store.save()
                _populate_right_panel(app)
            for w in (pill, tag_lbl):
                w.bind("<Button-1>", lambda e, t=tag: _remove_tag(t))
    else:
        ctk.CTkLabel(tag_frame, text="(none)", font=ctk.CTkFont(size=11),
                     text_color=COLOR_TEXT_MUTED, anchor="w").pack(anchor="w", padx=(12, 0))
    tag_add = ctk.CTkFrame(tag_frame, fg_color="transparent")
    tag_add.pack(fill="x", padx=(12, 0), pady=(2, 0))
    tag_entry = ctk.CTkEntry(tag_add, placeholder_text="Add tag...", height=24,
                             font=ctk.CTkFont(size=10))
    tag_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    def _add_tag():
        text = tag_entry.get().strip()
        if not text or text in proj.tags:
            return
        proj.tags.append(text)
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        _populate_right_panel(app)
    ctk.CTkButton(tag_add, text="+", width=BROWSE_BUTTON_WIDTH, height=24,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=_add_tag).pack(side="right")
    tag_entry.bind("<Return>", lambda e: _add_tag())

    row_idx += 1

    # Row 3: Notes (left column)
    notes_frame = ctk.CTkFrame(grid, fg_color="transparent")
    notes_frame.grid(row=row_idx, column=0, sticky="new", pady=(0, 4))
    ctk.CTkLabel(notes_frame, text="Notes:", anchor="w",
                 font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
    notes_list = [n for n in (proj.notes or "").split("\n") if n.strip()]
    for i, note in enumerate(notes_list):
        nr = ctk.CTkFrame(notes_frame, fg_color="transparent")
        nr.pack(fill="x", padx=(12, 0), pady=(0, 1))
        lbl = ctk.CTkLabel(nr, text=note, font=ctk.CTkFont(size=11),
                           anchor="w", cursor="hand2")
        lbl.pack(side="left", fill="x", expand=True)
        def _remove_note(idx=i):
            del notes_list[idx]
            proj.notes = "\n".join(notes_list)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _populate_right_panel(app)
        ctk.CTkButton(nr, text="\u2715", width=20, height=18,
                      font=ctk.CTkFont(size=9), fg_color="transparent",
                      hover_color=COLOR_ACTION_DANGER,
                      command=_remove_note).pack(side="right")
    if not notes_list:
        ctk.CTkLabel(notes_frame, text="(none)", font=ctk.CTkFont(size=11),
                     text_color=COLOR_TEXT_MUTED, anchor="w").pack(anchor="w", padx=(12, 0))
    notes_add = ctk.CTkFrame(notes_frame, fg_color="transparent")
    notes_add.pack(fill="x", padx=(12, 0), pady=(2, 0))
    notes_entry = ctk.CTkEntry(notes_add, placeholder_text="Add note...", height=24,
                               font=ctk.CTkFont(size=10))
    notes_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    def _add_note():
        text = notes_entry.get().strip()
        if not text:
            return
        notes_list.append(text)
        proj.notes = "\n".join(notes_list)
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        _populate_right_panel(app)
    ctk.CTkButton(notes_add, text="+", width=BROWSE_BUTTON_WIDTH, height=24,
                  fg_color=COLOR_ACTION_SECONDARY, hover_color=COLOR_ACTION_SECONDARY_H,
                  command=_add_note).pack(side="right")
    notes_entry.bind("<Return>", lambda e: _add_note())


# ======================================================================
#  Viewer panel: directory tree + thumbnail grid
# ======================================================================

def _build_viewer_panel(app, proj):
    """Build the split viewer: directory tree (left) + thumbnail grid (right)."""
    viewer = ctk.CTkFrame(app._proj_content_frame, corner_radius=8)
    viewer.pack(fill="both", expand=True, padx=6, pady=(4, 6))
    app._proj_viewer = viewer

    root_dir = proj.root_dir.strip() if proj.root_dir else ""

    if not root_dir or not Path(root_dir).is_dir():
        ctk.CTkLabel(
            viewer,
            text="No root directory set" if not root_dir else f"Root not found: {root_dir}",
            text_color="#6b7280", font=ctk.CTkFont(size=11),
            justify="center",
        ).place(relx=0.5, rely=0.5, anchor="center")
        return

    # Split: tree left, thumbnails right
    viewer.grid_columnconfigure(0, weight=0, minsize=360)
    viewer.grid_columnconfigure(1, weight=1)
    viewer.grid_rowconfigure(0, weight=1)

    # Tree pane
    tree_frame = ctk.CTkScrollableFrame(viewer, width=360, fg_color="#1a1a1a")
    tree_frame.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)

    # Thumbnail pane
    thumb_pane = ctk.CTkFrame(viewer, fg_color="transparent")
    thumb_pane.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
    app._proj_thumb_pane = thumb_pane

    # Thumb pane header + grid
    app._proj_thumb_header = ctk.CTkLabel(
        thumb_pane, text="Select a directory to browse",
        font=("Consolas", 10), text_color="#6b7280", anchor="w",
    )
    app._proj_thumb_header.pack(fill="x", padx=4, pady=(4, 4))

    app._proj_thumb_scroll = ctk.CTkScrollableFrame(thumb_pane)
    app._proj_thumb_scroll.pack(fill="both", expand=True)
    app._proj_thumb_widgets = []

    # Selected directory tracking
    app._proj_viewer_selected_dir = None
    app._proj_viewer_tree_rows = []

    # Loading indicator
    loading_lbl = ctk.CTkLabel(
        tree_frame, text="Loading directory tree...",
        font=("Consolas", 10), text_color="#6b7280",
    )
    loading_lbl.pack(pady=10)

    # Build tree in background thread
    root_path = Path(root_dir)

    def _scan_tree():
        """Walk directory tree in background, return list of (depth, name, path, img_count, has_subdirs)."""
        results = []
        _scan_recursive(results, root_path, depth=0, max_depth=3)
        return results

    def _scan_recursive(results, dir_path, depth, max_depth):
        if depth >= max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except (PermissionError, OSError):
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            img_count = 0
            has_subdirs = False
            try:
                for f in entry.iterdir():
                    if f.is_dir() and not f.name.startswith("."):
                        has_subdirs = True
                    elif f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        img_count += 1
            except (PermissionError, OSError):
                pass
            results.append((depth, entry.name, str(entry), img_count, has_subdirs))
            _scan_recursive(results, entry, depth + 1, max_depth)

    # Track parent→children for expand/collapse
    # _tree_children maps dir_path → list of child row widgets
    app._proj_tree_children = {}
    app._proj_tree_expanded = {}

    def _build_from_results(results):
        """Create tree widgets on the main thread from scan results."""
        loading_lbl.destroy()

        # Build parent mapping: for each entry, find its parent path
        row_widgets = {}  # dir_path → row widget

        _tree_bg = "#1a1a1a"
        for depth, name, dir_path_str, img_count, has_subdirs in results:
            indent = depth * 18
            parent_path = str(Path(dir_path_str).parent)

            row = tk.Frame(tree_frame, bg=_tree_bg,
                           height=24, cursor="hand2")
            row.pack(fill="x", pady=0)
            row.pack_propagate(False)

            # Arrow for expandable folders
            arrow_text = "\u25BC" if has_subdirs else "  "
            arrow_lbl = tk.Label(
                row, text=arrow_text, font=("Consolas", 9),
                fg="#6b7280", bg=_tree_bg, cursor="hand2" if has_subdirs else "",
            )
            arrow_lbl.pack(side="left", padx=(indent + 2, 0))

            name_text = f"\U0001F4C1 {name}"
            name_lbl = tk.Label(
                row, text=name_text, font=("Consolas", 14),
                fg="#dce4ee", bg=_tree_bg, anchor="w",
            )
            name_lbl.pack(side="left", padx=(2, 0))

            if img_count > 0:
                tk.Label(
                    row, text=f"{img_count} imgs", font=("Consolas", 12),
                    fg="#6b7280", bg=_tree_bg,
                ).pack(side="right", padx=(0, 6))

            row_widgets[dir_path_str] = row

            # Track parent→child relationships
            if parent_path not in app._proj_tree_children:
                app._proj_tree_children[parent_path] = []
            app._proj_tree_children[parent_path].append(dir_path_str)

            # Depth 0 and 1 visible (depth 0 starts expanded); depth 2+ hidden
            if depth == 0:
                app._proj_tree_expanded[dir_path_str] = True
            else:
                app._proj_tree_expanded[dir_path_str] = False
                if depth > 1:
                    row.pack_forget()

            # Click to select directory (load thumbnails)
            def _on_dir_click(event=None, dp=dir_path_str):
                app._proj_viewer_selected_dir = dp
                _highlight_tree_row(app, dp)
                _load_directory_thumbnails(app, dp)

            for w in (name_lbl,):
                w.bind("<Button-1>", _on_dir_click)
            row.bind("<Button-1>", _on_dir_click)

            # Click arrow to expand/collapse
            if has_subdirs:
                def _on_toggle(event=None, dp=dir_path_str, arrow=arrow_lbl):
                    expanded = app._proj_tree_expanded.get(dp, False)
                    app._proj_tree_expanded[dp] = not expanded
                    arrow.configure(text="\u25BC" if not expanded else "\u25B6")
                    _toggle_children(app, dp, not expanded, row_widgets)
                arrow_lbl.bind("<Button-1>", _on_toggle)
                # Start depth-0 expanded (arrow down), others collapsed (arrow right)
                if depth == 0:
                    arrow_lbl.configure(text="\u25BC")
                    app._proj_tree_expanded[dir_path_str] = True
                else:
                    arrow_lbl.configure(text="\u25B6")

            app._proj_viewer_tree_rows.append((dir_path_str, row))

    def _thread():
        results = _scan_tree()
        app.after(0, lambda: _build_from_results(results))

    threading.Thread(target=_thread, daemon=True).start()


def _highlight_tree_row(app, selected_dir):
    """Highlight the selected directory row in the tree."""
    for dir_path, row in app._proj_viewer_tree_rows:
        if dir_path == selected_dir:
            bg = "#1f6aa5"
        else:
            bg = "#1a1a1a"
        row.configure(bg=bg)
        for child in row.winfo_children():
            try:
                child.configure(bg=bg)
            except tk.TclError:
                pass


def _toggle_children(app, parent_path, show, row_widgets):
    """Show or hide direct children of a directory in the tree.
    Returns the last widget placed (for correct pack ordering)."""
    children = app._proj_tree_children.get(parent_path, [])
    last_widget = row_widgets.get(parent_path)
    for child_path in children:
        widget = row_widgets.get(child_path)
        if not widget:
            continue
        if show:
            widget.pack(fill="x", pady=0, after=last_widget)
            last_widget = widget
            # If this child was expanded, also show its grandchildren
            if app._proj_tree_expanded.get(child_path, False):
                last_widget = _toggle_children(app, child_path, True, row_widgets)
        else:
            widget.pack_forget()
            _toggle_children(app, child_path, False, row_widgets)
    return last_widget


def _load_directory_thumbnails(app, dir_path):
    """Load thumbnails for the given directory into the thumbnail grid."""
    # Clear existing thumbnails
    for w in app._proj_thumb_widgets:
        w.destroy()
    app._proj_thumb_widgets.clear()

    path = Path(dir_path)
    try:
        img_files = sorted(
            [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda f: f.name.lower(),
        )
    except PermissionError:
        img_files = []

    count = len(img_files)
    dir_name = path.name
    app._proj_thumb_header.configure(
        text=f"{dir_name}/ — {count} image{'s' if count != 1 else ''}",
        text_color="#dce4ee",
    )

    if not img_files:
        app._proj_thumb_header.configure(text=f"{dir_name}/ — no images")
        return

    # Create placeholder grid — thumbnails fit within a bounding box
    cols = 4
    thumb_box = (200, 150)  # max bounding box per cell
    thumb_widgets = {}

    # Store image file list for preview navigation
    app._proj_thumb_img_files = img_files[:200]

    for i, img_path in enumerate(img_files[:200]):  # Cap at 200 thumbnails
        r, c = divmod(i, cols)
        cell = ctk.CTkLabel(
            app._proj_thumb_scroll, text=img_path.stem[:12],
            fg_color="#343638", corner_radius=4,
            width=thumb_box[0], height=thumb_box[1],
            font=("Consolas", 8), text_color="#565b5e",
        )
        cell.grid(row=r, column=c, padx=6, pady=6)
        app._proj_thumb_widgets.append(cell)
        thumb_widgets[str(img_path)] = (cell, i)

        # Double-click to open preview
        def _on_dblclick(event=None, idx=i):
            _open_image_preview(app, idx)
        cell.bind("<Double-Button-1>", _on_dblclick)

    # Configure grid columns
    for c in range(cols):
        app._proj_thumb_scroll.grid_columnconfigure(c, weight=1)

    # Load thumbnails async
    try:
        from thumb_cache import load_thumbnails_async

        def _on_thumb(path, pil_img, widgets=thumb_widgets, box=thumb_box):
            key = str(path)
            if key not in widgets:
                return
            widget, idx = widgets[key]

            def _update():
                try:
                    # Fit within bounding box, preserving aspect ratio
                    iw, ih = pil_img.size
                    scale = min(box[0] / iw, box[1] / ih)
                    display_size = (max(int(iw * scale), 1), max(int(ih * scale), 1))
                    ctk_img = ctk.CTkImage(light_image=pil_img, size=display_size)
                    widget.configure(image=ctk_img, text="")
                    widget._ctk_image = ctk_img  # prevent GC
                except Exception:
                    pass

            app.after(0, _update)

        load_thumbnails_async(img_files[:200], _on_thumb)
    except ImportError:
        pass  # PIL not available — placeholders remain


def _open_image_preview(app, index):
    """Open a full-pane image preview overlay, replacing the thumbnail grid."""
    if not hasattr(app, '_proj_thumb_img_files') or not app._proj_thumb_img_files:
        return

    img_files = app._proj_thumb_img_files
    if index < 0 or index >= len(img_files):
        return

    # Hide the thumbnail scroll and header
    app._proj_thumb_header.pack_forget()
    app._proj_thumb_scroll.pack_forget()

    # Create preview overlay in the thumb pane using pack layout
    preview = ctk.CTkFrame(app._proj_thumb_pane, fg_color="#1a1a1a", corner_radius=0)
    preview.pack(fill="both", expand=True)
    app._proj_preview_frame = preview
    app._proj_preview_index = index

    # Top bar: close button
    top_bar = ctk.CTkFrame(preview, fg_color="transparent", height=36)
    top_bar.pack(fill="x")
    top_bar.pack_propagate(False)
    ctk.CTkButton(
        top_bar, text="X  Close", width=70, height=26,
        font=ctk.CTkFont(size=11), fg_color="#343638",
        hover_color="#dc2626",
        command=lambda: _close_image_preview(app),
    ).pack(side="right", padx=6, pady=4)

    # Middle area: left arrow + image + right arrow
    mid = ctk.CTkFrame(preview, fg_color="transparent")
    mid.pack(fill="both", expand=True)

    left_btn = ctk.CTkButton(
        mid, text="<", width=36, height=60,
        font=ctk.CTkFont(size=18, weight="bold"), fg_color="#343638",
        hover_color="#4a4d50", corner_radius=4,
        command=lambda: _preview_navigate(app, -1),
    )
    left_btn.pack(side="left", padx=(4, 0), pady=4)

    right_btn = ctk.CTkButton(
        mid, text=">", width=36, height=60,
        font=ctk.CTkFont(size=18, weight="bold"), fg_color="#343638",
        hover_color="#4a4d50", corner_radius=4,
        command=lambda: _preview_navigate(app, 1),
    )
    right_btn.pack(side="right", padx=(0, 4), pady=4)

    img_container = ctk.CTkFrame(mid, fg_color="transparent")
    img_container.pack(fill="both", expand=True, padx=4, pady=4)

    img_label = tk.Label(img_container, text="Loading...", fg="#6b7280",
                         bg="#1a1a1a", bd=0, highlightthickness=0)
    img_label.place(relx=0.5, rely=0.5, anchor="center")
    app._proj_preview_img_label = img_label
    app._proj_preview_img_container = img_container

    # Re-fit image when container resizes (debounced to avoid feedback loops)
    app._proj_preview_resize_id = None
    def _on_container_resize(event=None):
        if app._proj_preview_resize_id:
            app.after_cancel(app._proj_preview_resize_id)
        app._proj_preview_resize_id = app.after(
            150, lambda: _preview_load_image(app, app._proj_preview_index))
    img_container.bind("<Configure>", _on_container_resize)

    # Bottom bar: filename + index
    bottom_bar = ctk.CTkFrame(preview, fg_color="#2b2b2b", height=28, corner_radius=0)
    bottom_bar.pack(fill="x")
    bottom_bar.pack_propagate(False)
    app._proj_preview_filename = ctk.CTkLabel(
        bottom_bar, text="", font=("Consolas", 10), text_color="#dce4ee", anchor="w",
    )
    app._proj_preview_filename.pack(side="left", padx=12, pady=2)
    app._proj_preview_counter = ctk.CTkLabel(
        bottom_bar, text="", font=("Consolas", 9), text_color="#6b7280", anchor="e",
    )
    app._proj_preview_counter.pack(side="right", padx=12, pady=2)

    # Keyboard bindings on the main window
    app.bind("<Escape>", lambda e: _close_image_preview(app))
    app.bind("<Left>", lambda e: _preview_navigate(app, -1))
    app.bind("<Right>", lambda e: _preview_navigate(app, 1))

    # Defer image load so the frame has rendered and has real dimensions
    preview.after(50, lambda: _preview_load_image(app, index))


def _preview_load_image(app, index):
    """Load and display the image at the given index in the preview."""
    img_files = app._proj_thumb_img_files
    if index < 0 or index >= len(img_files):
        return

    app._proj_preview_index = index
    img_path = img_files[index]

    app._proj_preview_filename.configure(text=img_path.name)
    app._proj_preview_counter.configure(text=f"{index + 1} / {len(img_files)}")

    try:
        from PIL import Image, ImageTk
        pil_img = Image.open(str(img_path))

        # Fit image to the container's actual rendered size
        container = app._proj_preview_img_container
        container.update_idletasks()
        pane_w = container.winfo_width()
        pane_h = container.winfo_height()
        if pane_w < 50:
            pane_w = 800
        if pane_h < 50:
            pane_h = 600

        img_w, img_h = pil_img.size
        scale = min(pane_w / img_w, pane_h / img_h)
        display_w = max(int(img_w * scale), 1)
        display_h = max(int(img_h * scale), 1)

        resized = pil_img.resize((display_w, display_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        app._proj_preview_img_label.configure(image=tk_img, text="")
        app._proj_preview_img_label._tk_image = tk_img  # prevent GC
    except Exception as e:
        app._proj_preview_img_label.configure(text=f"Could not load image: {e}", image="")


def _preview_navigate(app, direction):
    """Navigate the preview by +1 or -1."""
    new_idx = app._proj_preview_index + direction
    img_files = app._proj_thumb_img_files
    if 0 <= new_idx < len(img_files):
        _preview_load_image(app, new_idx)


def _close_image_preview(app):
    """Close the preview overlay and restore the thumbnail grid."""
    if hasattr(app, '_proj_preview_frame') and app._proj_preview_frame:
        app._proj_preview_frame.destroy()
        app._proj_preview_frame = None

    # Unbind keyboard shortcuts
    app.unbind("<Escape>")
    app.unbind("<Left>")
    app.unbind("<Right>")

    # Restore thumbnail header + scroll
    app._proj_thumb_header.pack(fill="x", padx=4, pady=(4, 4))
    app._proj_thumb_scroll.pack(fill="both", expand=True)


def _format_size(size_bytes):
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _refresh_current_projects_list(app):
    """Rebuild the current projects list in the left column."""
    if not hasattr(app, '_proj_current_scroll'):
        return

    for w in app._proj_current_widgets:
        w.destroy()
    app._proj_current_widgets.clear()

    search = app._proj_search_var.get().strip() if hasattr(app, '_proj_search_var') else ""
    projects = app._project_store.list_projects(search=search)

    if hasattr(app, '_proj_count_label'):
        app._proj_count_label.configure(
            text=f"{len(projects)} project{'s' if len(projects) != 1 else ''}")

    from project_store import STAGE_ORDER
    TOOL_LABELS = {"Metashape", "RealityScan", "COLMAP"}

    from widgets import (
        COLOR_ACTION_SECONDARY, COLOR_ACTION_MUTED,
        COLOR_BORDER_SECTION, COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
        RADIUS_PILL,
    )

    for proj in projects:
        is_selected = (proj.id == app._selected_project_id)
        card = ctk.CTkFrame(
            app._proj_current_scroll,
            fg_color=COLOR_ACTION_SECONDARY if is_selected else "transparent",
            border_width=1,
            border_color=COLOR_ACTION_SECONDARY if is_selected else COLOR_BORDER_SECTION,
            corner_radius=6, cursor="hand2",
        )
        card.pack(fill="x", pady=(0, 6))

        # Row 1: title + relative time
        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(8, 0))
        ctk.CTkLabel(
            top_row, text=proj.title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#ffffff" if is_selected else "#dce4ee",
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            top_row, text=_format_relative_time(proj.updated_at),
            font=ctk.CTkFont(size=11),
            text_color="#93c5fd" if is_selected else COLOR_TEXT_DIM,
        ).pack(side="right")

        # Row 2: root dir path
        if proj.root_dir:
            ctk.CTkLabel(
                card, text=proj.root_dir,
                font=ctk.CTkFont(size=11),
                text_color="#93c5fd" if is_selected else COLOR_TEXT_DIM,
                anchor="w",
            ).pack(fill="x", padx=10, pady=(2, 0))

        # Row 3: method badges (left) + stage pills (right) on same line
        tool_names = sorted(set(s.label for s in proj.sources if s.label in TOOL_LABELS))
        capture_tags = [t for t in proj.tags if t in {"drone", "dslr", "phone", "360"}]
        method_badges = tool_names + [t.capitalize() for t in capture_tags]

        max_stage = ""
        if proj.work_dirs:
            for wd in proj.work_dirs:
                if wd.stage and (not max_stage or
                        STAGE_ORDER.index(wd.stage) > STAGE_ORDER.index(max_stage)):
                    max_stage = wd.stage

        if method_badges or max_stage:
            status_row = ctk.CTkFrame(card, fg_color="transparent")
            status_row.pack(fill="x", padx=10, pady=(6, 0))

            # Stage pills — right aligned
            if max_stage:
                max_idx = STAGE_ORDER.index(max_stage)
                for stage_name in reversed(STAGE_ORDER):
                    si = STAGE_ORDER.index(stage_name)
                    done = si <= max_idx
                    pill = ctk.CTkLabel(
                        status_row, text=stage_name.capitalize(),
                        font=ctk.CTkFont(size=10),
                        text_color="#ffffff" if done else COLOR_TEXT_DIM,
                        fg_color=COLOR_ACTION_SECONDARY if done else COLOR_ACTION_MUTED,
                        corner_radius=RADIUS_PILL, padx=4, pady=1,
                    )
                    pill.pack(side="right", padx=(3, 0))

            # Method badges — left aligned
            for badge_text in method_badges:
                badge = ctk.CTkLabel(
                    status_row, text=badge_text,
                    font=ctk.CTkFont(size=10),
                    text_color=COLOR_TEXT_MUTED if not is_selected else "#dce4ee",
                    fg_color=COLOR_ACTION_MUTED, corner_radius=RADIUS_PILL,
                    padx=4, pady=1,
                )
                badge.pack(side="left", padx=(0, 4))

        # Bottom padding
        ctk.CTkFrame(card, fg_color="transparent", height=8).pack(fill="x")

        pid = proj.id

        def _on_click(event=None, pid=pid):
            app._selected_project_id = pid
            _refresh_current_projects_list(app)
            _populate_right_panel(app)

        def _bind_recursive(widget, handler):
            widget.bind("<Button-1>", handler)
            for child in widget.winfo_children():
                _bind_recursive(child, handler)

        _bind_recursive(card, _on_click)

        app._proj_current_widgets.append(card)

    if not projects:
        lbl = ctk.CTkLabel(
            app._proj_current_scroll,
            text="No projects found" if search else "No projects yet",
            text_color="#9ca3af",
        )
        lbl.pack(pady=20)
        app._proj_current_widgets.append(lbl)


def _format_relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to relative time string."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        if delta.days > 7:
            return dt.strftime("%Y-%m-%d")
        elif delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        else:
            return "just now"
    except (ValueError, TypeError):
        return ""


def _cycle_work_dir_stage(app, project_id, work_dir_index, clicked_stage):
    """Set a working directory's stage. Clicking a done stage rolls back."""
    from project_store import STAGE_ORDER

    proj = app._project_store.get_project(project_id)
    if not proj or work_dir_index >= len(proj.work_dirs):
        return
    wd = proj.work_dirs[work_dir_index]
    current = wd.stage or ""

    try:
        current_idx = STAGE_ORDER.index(current)
    except ValueError:
        current_idx = -1

    clicked_idx = STAGE_ORDER.index(clicked_stage)

    if clicked_idx <= current_idx:
        if clicked_idx == 0:
            wd.stage = ""
        else:
            wd.stage = STAGE_ORDER[clicked_idx - 1]
    else:
        wd.stage = clicked_stage

    proj.updated_at = datetime.now().isoformat()
    app._project_store.save()
    _refresh_project_list(app)


# ======================================================================
#  Export logic
# ======================================================================

def _do_export(app, proj):
    """Export project in all formats."""
    import webbrowser

    out_dir = Path(app._project_store.store_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = proj.title.replace(" ", "_").lower()[:30]

    exported = []
    exported.append(_export_single_md(proj, out_dir / f"{suffix}.md"))
    exported.append(_export_single_html(proj, out_dir / f"{suffix}.html"))
    exported.append(_export_single_json(proj, out_dir / f"{suffix}.json"))

    app.log(f"Exported {len(exported)} file(s) to {out_dir}")

    html_files = [f for f in exported if str(f).endswith(".html")]
    if html_files:
        webbrowser.open(str(html_files[0]))


def _export_single_md(proj, path: Path) -> str:
    """Export a single project to markdown."""
    from project_store import STAGE_ORDER

    METHOD_LABELS = ("Metashape", "RealityScan", "COLMAP")
    lines = [
        f"# {proj.title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if proj.root_dir:
        lines.append(f"**Root:** `{proj.root_dir}`")
        lines.append("")
    if proj.tags:
        lines.append(f"**Tags:** {', '.join(proj.tags)}")
    lines.append("")

    if proj.sources:
        lines.append("## Sources")
        for src in proj.sources:
            exists = Path(src.path).exists()
            status = "" if exists else " [MISSING]"
            count = f" ({src.file_count} files)" if src.file_count else ""
            stage = ""
            if src.label not in METHOD_LABELS and src.stage:
                stage = f" \u2014 {src.stage.replace('_', ' ').title()}"
            lines.append(f"- {src.label}: `{src.path}`{count}{status}{stage}")
        lines.append("")

    if proj.notes:
        lines.append("## Notes")
        lines.append(proj.notes)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _export_single_html(proj, path: Path) -> str:
    """Export a single project to styled HTML."""
    from project_store import STAGE_ORDER

    METHOD_LABELS = ("Metashape", "RealityScan", "COLMAP")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    root_html = ""
    if proj.root_dir:
        root_html = f'<p style="font-family:monospace;color:#9ca3af">{proj.root_dir}</p>'

    sources_html = ""
    if proj.sources:
        items = []
        for src in proj.sources:
            exists = Path(src.path).exists()
            clr = "#22c55e" if exists else "#ef4444"
            ind = "OK" if exists else "MISSING"
            cnt = f" ({src.file_count})" if src.file_count else ""
            stage_html = ""
            if src.label not in METHOD_LABELS and src.stage:
                try:
                    stage_idx = STAGE_ORDER.index(src.stage)
                except ValueError:
                    stage_idx = -1
                pills = []
                for si, sn in enumerate(STAGE_ORDER):
                    pill_color = "#22c55e" if si <= stage_idx else "#6b7280"
                    pills.append(
                        f'<span style="color:{pill_color};font-size:10px;margin-right:6px">'
                        f'{"+" if si <= stage_idx else "-"} {sn.replace("_"," ").title()}</span>'
                    )
                stage_html = ('<div style="padding-left:20px;margin-top:2px">'
                              + "".join(pills) + '</div>')
            items.append(
                f'<div style="font-family:monospace;font-size:12px;padding:2px 0">'
                f'<span style="color:{clr}">[{ind}]</span> '
                f'<b>{src.label}</b>: {src.path}{cnt}</div>{stage_html}'
            )
        sources_html = "<div style='margin:12px 0'>" + "\n".join(items) + "</div>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{proj.title}</title>
<style>body {{ background:#111827;color:#f9fafb;font-family:-apple-system,sans-serif;padding:24px;max-width:900px;margin:0 auto; }}</style>
</head><body>
<h1>{proj.title}</h1>
<p style="color:#9ca3af">Generated: {now}</p>
{root_html}
{sources_html}
{f'<div style="color:#9ca3af;margin-top:12px"><b>Notes:</b><br>{proj.notes}</div>' if proj.notes else ''}
{f'<div style="margin-top:8px"><b>Tags:</b> {", ".join(proj.tags)}</div>' if proj.tags else ''}
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return str(path)


def _export_single_json(proj, path: Path) -> str:
    """Export a single project to JSON."""
    import json
    data = {
        "generated": datetime.now().isoformat(),
        "project": proj.to_dict(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
