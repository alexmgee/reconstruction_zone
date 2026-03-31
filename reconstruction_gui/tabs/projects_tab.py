"""
Projects Tab -- Central registry for photogrammetry projects.

Project pane (left tab content): searchable project list.
Details pane (right, replaces preview panel): project details, sections.
All interactions are inline -- no popout dialogs.
"""

import threading
import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from widgets import Section, CollapsibleSection, Tooltip


# ── Capture method checkboxes (pre-tags) ──

CAPTURE_METHODS = [
    "drone", "dslr", "phone", "360",
]


# ======================================================================
#  Entry point
# ======================================================================

def build_projects_tab(app, parent):
    """Build the Projects tab UI. Called once during app init.

    The project list lives in the tab content (left column of the main layout).
    Project details go into a separate panel that replaces the preview panel
    in column 1 of _main_frame when the Projects tab is active.
    """
    app._selected_project_id = None

    # Initialize store
    from project_store import ProjectStore
    store_path = app._prefs.get("tracker_store_path", "D:\\tracker.json")
    app._project_store = ProjectStore(store_path)

    # -- Project pane (tab content): project list --
    _build_list_panel(app, parent)

    # -- Details pane (lives in _main_frame column 1) --
    # Created here but only shown when Projects tab is active.
    app._proj_detail_panel = ctk.CTkFrame(app._main_frame)

    # Empty-state label (centered placeholder)
    app._proj_empty_label = ctk.CTkLabel(
        app._proj_detail_panel, text="Select a project or click + New",
        font=ctk.CTkFont(size=13), text_color="#9ca3af",
    )
    app._proj_empty_label.place(relx=0.5, rely=0.5, anchor="center")

    # Scrollable detail area -- created once, content cleared/rebuilt on selection
    app._proj_detail_scroll = ctk.CTkScrollableFrame(app._proj_detail_panel)

    # Initial list population (deferred so UI is built first)
    parent.after(100, lambda: _refresh_project_list(app))


# ======================================================================
#  Project pane: project list
# ======================================================================

def _build_list_panel(app, parent):
    """Project pane: search bar, filter, scrollable project list."""
    # -- Header row: title + buttons --
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", padx=6, pady=(6, 2))
    ctk.CTkLabel(
        header, text="Projects",
        font=ctk.CTkFont(size=14, weight="bold"),
    ).pack(side="left")
    ctk.CTkButton(
        header, text="+ New", width=60,
        command=lambda: _on_new_project(app),
    ).pack(side="right")
    ctk.CTkButton(
        header, text="Scan", width=60,
        command=lambda: _on_scan(app),
    ).pack(side="right", padx=(0, 4))

    # -- Search bar --
    app._proj_search_var = tk.StringVar()
    app._proj_search_var.trace_add("write", lambda *_: _refresh_project_list(app))
    search = ctk.CTkEntry(
        parent, placeholder_text="Search projects...",
        textvariable=app._proj_search_var,
    )
    search.pack(fill="x", padx=6, pady=(2, 4))

    # -- Scrollable project list --
    app._proj_list_frame = ctk.CTkScrollableFrame(parent)
    app._proj_list_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
    app._proj_list_widgets = []

    # -- Store path at bottom --
    settings_frame = ctk.CTkFrame(parent, fg_color="transparent")
    settings_frame.pack(fill="x", padx=6, pady=(4, 6), side="bottom")
    ctk.CTkLabel(
        settings_frame, text="Store:", font=("Consolas", 9), text_color="#6b7280",
    ).pack(side="left")
    app._proj_store_label = ctk.CTkLabel(
        settings_frame,
        text=str(app._project_store.store_path) if app._project_store else "Not set",
        font=("Consolas", 9), text_color="#6b7280",
    )
    app._proj_store_label.pack(side="left", padx=(4, 0))


# ======================================================================
#  Project list refresh + row widget
# ======================================================================

def _refresh_project_list(app):
    """Rebuild the project list from the store."""
    if not app._project_store:
        return

    for w in app._proj_list_widgets:
        w.destroy()
    app._proj_list_widgets.clear()

    search = app._proj_search_var.get().strip()
    projects = app._project_store.list_projects(include_archived=False, search=search)

    for proj in projects:
        row = _create_project_list_row(app, proj)
        row.pack(fill="x", pady=(0, 2))
        app._proj_list_widgets.append(row)

    if not projects:
        lbl = ctk.CTkLabel(
            app._proj_list_frame,
            text="No projects found" if search else "No projects yet -- click + New",
            text_color="#9ca3af",
        )
        lbl.pack(pady=20)
        app._proj_list_widgets.append(lbl)


def _create_project_list_row(app, proj) -> ctk.CTkFrame:
    """Create a clickable row for one project in the list."""
    row = ctk.CTkFrame(app._proj_list_frame, cursor="hand2")

    title_lbl = ctk.CTkLabel(
        row, text=proj.title,
        font=ctk.CTkFont(size=12, weight="bold"),
        anchor="w",
    )
    title_lbl.pack(fill="x", padx=8, pady=(4, 0))

    # Subtitle: source count + tags preview
    src_count = len(proj.sources)
    subtitle = f"{src_count} source{'s' if src_count != 1 else ''}"
    if proj.tags:
        subtitle += f"  |  {', '.join(proj.tags[:3])}"

    sub_lbl = ctk.CTkLabel(
        row, text=subtitle,
        font=("Consolas", 10), text_color="#9ca3af",
        anchor="w",
    )
    sub_lbl.pack(fill="x", padx=8, pady=(0, 4))

    # Click to select
    pid = proj.id
    for widget in (row, title_lbl, sub_lbl):
        widget.bind("<Button-1>", lambda e, pid=pid: _show_project_detail(app, pid))

    # Highlight if selected
    if app._selected_project_id == proj.id:
        row.configure(border_width=2, border_color=("#3b82f6", "#3b82f6"))

    return row


# ======================================================================
#  New project (instant creation, no dialog)
# ======================================================================

def _on_new_project(app):
    """Create a new project with a date-based default title and select it."""
    title = datetime.now().strftime("%Y-%m-%d Untitled")
    proj = app._project_store.create_project(title)
    _refresh_project_list(app)
    _show_project_detail(app, proj.id)


# ======================================================================
#  Details pane: project detail
# ======================================================================

def _show_project_detail(app, project_id: str):
    """Populate the details pane with full project details."""
    proj = app._project_store.get_project(project_id)
    if not proj:
        return

    app._selected_project_id = project_id
    _refresh_project_list(app)

    # Hide placeholder, clear old content, show scroll frame
    app._proj_empty_label.place_forget()
    scroll = app._proj_detail_scroll
    for w in scroll.winfo_children():
        w.destroy()
    scroll.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Pipeline header (always visible at top) ──
    _build_pipeline_header(app, scroll, proj)

    # ── Separator ──
    ctk.CTkFrame(scroll, height=1, fg_color=("#d1d5db", "#404040")).pack(
        fill="x", padx=4, pady=(4, 8),
    )

    # ── Sections (all expanded by default) ──
    _build_project_info_section(app, scroll, proj)
    _build_notes_section(app, scroll, proj)
    _build_media_section(app, scroll, proj)
    _build_export_section(app, scroll, proj)
    _build_tags_section(app, scroll, proj)
    _build_delete_button(app, scroll, proj)


# ── Pipeline header ──

def _build_pipeline_header(app, parent, proj):
    """Fixed header: project title."""
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", padx=4, pady=(4, 2))

    ctk.CTkLabel(
        header, text=proj.title,
        font=ctk.CTkFont(size=16, weight="bold"), anchor="w",
    ).pack(side="left")


# ── Project Info section ──

def _build_project_info_section(app, parent, proj):
    """Editable project info: title, subject, capture method checkboxes, notes."""
    sec = CollapsibleSection(parent, "Project Info", expanded=True)
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    # Title
    title_row = ctk.CTkFrame(c, fg_color="transparent")
    title_row.pack(fill="x", padx=6, pady=3)
    ctk.CTkLabel(title_row, text="Title:", width=60, anchor="w").pack(side="left")
    title_entry = ctk.CTkEntry(title_row)
    title_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
    title_entry.insert(0, proj.title)

    def _save_title(event=None):
        t = title_entry.get().strip()
        if t and t != proj.title:
            proj.title = t
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _refresh_project_list(app)
    title_entry.bind("<FocusOut>", _save_title)
    title_entry.bind("<Return>", _save_title)

    # Capture method checkboxes (act as pre-tags)
    method_label = ctk.CTkFrame(c, fg_color="transparent")
    method_label.pack(fill="x", padx=6, pady=(6, 2))
    ctk.CTkLabel(
        method_label, text="Capture method:",
        font=ctk.CTkFont(size=11), anchor="w",
    ).pack(side="left")

    method_frame = ctk.CTkFrame(c, fg_color="transparent")
    method_frame.pack(fill="x", padx=6, pady=(0, 4))

    # Current tags that match capture methods
    current_tags = set(proj.tags)
    app._proj_method_vars = {}

    for method in CAPTURE_METHODS:
        var = tk.BooleanVar(value=method in current_tags)
        app._proj_method_vars[method] = var

        def _on_method_toggle(m=method, v=var):
            if v.get():
                if m not in proj.tags:
                    proj.tags.append(m)
            else:
                if m in proj.tags:
                    proj.tags.remove(m)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _refresh_project_list(app)

        ctk.CTkCheckBox(
            method_frame, text=method.upper(), variable=var,
            command=_on_method_toggle,
            font=ctk.CTkFont(size=10), height=22, width=80,
        ).pack(side="left", padx=(0, 6))


# ── Notes section (entry + submit + selectable removal) ──

def _build_notes_section(app, parent, proj):
    """Notes as a list: type a note, submit to add, select to remove."""
    sec = CollapsibleSection(parent, "Notes", expanded=True)
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    # Parse existing notes (newline-separated)
    notes_list = [n for n in (proj.notes or "").split("\n") if n.strip()]

    # Entry + Submit row
    entry_row = ctk.CTkFrame(c, fg_color="transparent")
    entry_row.pack(fill="x", padx=6, pady=(4, 4))
    note_entry = ctk.CTkEntry(entry_row, placeholder_text="Add a note...")
    note_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_note():
        text = note_entry.get().strip()
        if not text:
            return
        notes_list.append(text)
        proj.notes = "\n".join(notes_list)
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        _show_project_detail(app, proj.id)

    ctk.CTkButton(
        entry_row, text="Add", width=60, command=_add_note,
    ).pack(side="right")
    note_entry.bind("<Return>", lambda e: _add_note())

    # Listed notes with selection for removal
    if notes_list:
        notes_container = ctk.CTkFrame(c, fg_color="transparent")
        notes_container.pack(fill="x", padx=6, pady=(0, 4))

        selected_indices = set()

        def _toggle_select(idx, row_frame):
            if idx in selected_indices:
                selected_indices.discard(idx)
                row_frame.configure(fg_color="transparent")
            else:
                selected_indices.add(idx)
                row_frame.configure(fg_color=("#dbeafe", "#1e3a5f"))

        for i, note in enumerate(notes_list):
            note_row = ctk.CTkFrame(notes_container, cursor="hand2")
            note_row.pack(fill="x", pady=(0, 2))
            lbl = ctk.CTkLabel(
                note_row, text=note, font=("Consolas", 10), anchor="w",
            )
            lbl.pack(fill="x", padx=6, pady=2)
            for w in (note_row, lbl):
                w.bind("<Button-1>", lambda e, idx=i, rf=note_row: _toggle_select(idx, rf))

        def _remove_selected():
            if not selected_indices:
                return
            new_notes = [n for i, n in enumerate(notes_list) if i not in selected_indices]
            proj.notes = "\n".join(new_notes)
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _show_project_detail(app, proj.id)

        ctk.CTkButton(
            notes_container, text="Remove selected", width=120,
            fg_color="#6b7280", command=_remove_selected,
        ).pack(anchor="w", pady=(4, 2))


# ── Media section ──

def _build_media_section(app, parent, proj):
    """Media section with two subsections: Add Media and Add Method."""
    sec = CollapsibleSection(parent, "Media", expanded=True)
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    # ── Add Media subsection ──
    media_sub = CollapsibleSection(c, "Add Media", expanded=True)
    media_sub.pack(fill="x", padx=2, pady=(0, 4))
    mc = media_sub.content

    # Existing media entries (skip method entries — shown in Add Method)
    METHOD_LABELS = ("Metashape", "RealityScan", "COLMAP")
    for i, src in enumerate(proj.sources):
        if src.label in METHOD_LABELS:
            continue
        _build_source_row(app, mc, proj, src, i)

    # Inline add area: label + Add button (Add launches directory dialog)
    add_frame = ctk.CTkFrame(mc, fg_color="transparent")
    add_frame.pack(fill="x", padx=4, pady=(4, 4))

    row1 = ctk.CTkFrame(add_frame, fg_color="transparent")
    row1.pack(fill="x", pady=2)
    ctk.CTkLabel(row1, text="Label:", width=50, anchor="w").pack(side="left")
    label_entry = ctk.CTkEntry(row1, placeholder_text="e.g. ERP frames, DJI video")
    label_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))

    def _add():
        label = label_entry.get().strip()
        if not label:
            messagebox.showwarning("Add Media", "Please enter a label first.")
            return
        path = filedialog.askdirectory(title="Select Media Directory")
        if not path:
            return
        proj_obj = app._project_store.get_project(proj.id)
        if proj_obj:
            proj_obj.add_source(label, path, "other")
            app._project_store.save()
        _show_project_detail(app, proj.id)

    ctk.CTkButton(row1, text="Add", width=60, command=_add).pack(side="right")

    # ── Add Method subsection ──
    method_sub = CollapsibleSection(c, "Add Method", expanded=True)
    method_sub.pack(fill="x", padx=2, pady=(0, 4))
    tc = method_sub.content

    # Show existing method entries (Metashape/RealityScan/COLMAP) from sources
    METHOD_LABELS = ("Metashape", "RealityScan", "COLMAP")
    for src in proj.sources:
        if src.label in METHOD_LABELS:
            _build_project_file_row(tc, src.label, src.path)

    # Buttons row for setting project files
    proj_file_row = ctk.CTkFrame(tc, fg_color="transparent")
    proj_file_row.pack(fill="x", padx=4, pady=(4, 4))
    ctk.CTkButton(
        proj_file_row, text="Metashape", width=90, fg_color="#6b7280",
        command=lambda: _set_metashape_path(app, proj.id),
    ).pack(side="left", padx=(0, 4))
    ctk.CTkButton(
        proj_file_row, text="RealityScan", width=90, fg_color="#6b7280",
        command=lambda: _set_project_file(app, proj.id, "RealityScan"),
    ).pack(side="left", padx=(0, 4))
    ctk.CTkButton(
        proj_file_row, text="COLMAP", width=90, fg_color="#6b7280",
        command=lambda: _set_project_file(app, proj.id, "COLMAP"),
    ).pack(side="left")


def _build_project_file_row(parent, label, path):
    """Display a project file path with health indicator."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=2)
    exists = Path(path).exists()
    color = "#22c55e" if exists else "#ef4444"
    indicator = "+" if exists else "X"
    ctk.CTkLabel(
        row, text=indicator, font=("Consolas", 12),
        text_color=color, width=20,
    ).pack(side="left")
    ctk.CTkLabel(
        row, text=f"{label}: {path}",
        font=("Consolas", 10), anchor="w",
    ).pack(side="left", fill="x", expand=True)


def _build_source_row(app, parent, proj, src, index):
    """Single source entry with path health, relocate, and stage progression."""
    from project_store import STAGE_ORDER

    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(fill="x", padx=4, pady=(4, 2))

    # Top row: health indicator + label + path + relocate
    row = ctk.CTkFrame(container, fg_color="transparent")
    row.pack(fill="x")

    exists = Path(src.path).exists()
    indicator = "+" if exists else "X"
    color = "#22c55e" if exists else "#ef4444"

    ctk.CTkLabel(
        row, text=indicator, font=("Consolas", 12),
        text_color=color, width=20,
    ).pack(side="left")

    info = f"{src.label}: {src.path}"
    if src.file_count:
        info += f"  ({src.file_count} files)"

    ctk.CTkLabel(
        row, text=info, font=("Consolas", 10), anchor="w",
    ).pack(side="left", fill="x", expand=True)

    ctk.CTkButton(
        row, text="Relocate", width=70, fg_color="#6b7280",
        command=lambda idx=index: _relocate_source(app, proj.id, idx),
    ).pack(side="right", padx=(4, 0))

    # Stage progression row
    stage_row = ctk.CTkFrame(container, fg_color="transparent")
    stage_row.pack(fill="x", padx=(20, 0), pady=(2, 0))

    current_stage = src.stage or ""
    # Find index of current stage (-1 if none)
    try:
        current_idx = STAGE_ORDER.index(current_stage)
    except ValueError:
        current_idx = -1

    for si, stage_name in enumerate(STAGE_ORDER):
        if si <= current_idx:
            fg = "#22c55e"
            symbol = "+"
        else:
            fg = "#6b7280"
            symbol = "-"

        display = stage_name.replace("_", " ").title()

        btn = ctk.CTkButton(
            stage_row, text=f"{symbol} {display}",
            fg_color="transparent", hover_color=("#e5e7eb", "#374151"),
            text_color=fg, border_width=1, border_color=fg,
            width=90, height=24,
            font=ctk.CTkFont(size=10),
            command=lambda sn=stage_name: _cycle_source_stage(app, proj.id, index, sn),
        )
        btn.pack(side="left", padx=(0, 3))


def _cycle_source_stage(app, project_id, source_index, clicked_stage):
    """Set a source's stage. Clicking a done stage clears it (and later ones)."""
    from project_store import STAGE_ORDER

    proj = app._project_store.get_project(project_id)
    if not proj or source_index >= len(proj.sources):
        return
    src = proj.sources[source_index]
    current = src.stage or ""

    try:
        current_idx = STAGE_ORDER.index(current)
    except ValueError:
        current_idx = -1

    clicked_idx = STAGE_ORDER.index(clicked_stage)

    if clicked_idx <= current_idx:
        # Clicking a completed stage: roll back to one before it
        if clicked_idx == 0:
            src.stage = ""
        else:
            src.stage = STAGE_ORDER[clicked_idx - 1]
    else:
        # Clicking an incomplete stage: advance to it (all prior are implicitly done)
        src.stage = clicked_stage

    proj.updated_at = datetime.now().isoformat()
    app._project_store.save()
    _show_project_detail(app, project_id)


def _relocate_source(app, project_id, source_index):
    """Browse for a new path for a source that has moved."""
    new_path = filedialog.askdirectory(title="Select new location")
    if new_path:
        app._project_store.relocate_source(project_id, source_index, new_path)
        _show_project_detail(app, project_id)


def _set_metashape_path(app, project_id):
    """Browse for a .psx file and store as a source entry."""
    path = filedialog.askopenfilename(
        title="Select Metashape Project",
        filetypes=[("Metashape Project", "*.psx"), ("All Files", "*.*")],
    )
    if path:
        proj = app._project_store.get_project(project_id)
        if proj:
            from project_store import ProjectSource
            proj.sources.append(ProjectSource(
                label="Metashape", path=path, media_type="other",
            ))
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _show_project_detail(app, project_id)


def _set_project_file(app, project_id, tool_name):
    """Browse for a project file (RealityScan, COLMAP, etc.) and store as a source."""
    if tool_name == "RealityScan":
        path = filedialog.askdirectory(title="Select RealityScan project folder")
    elif tool_name == "COLMAP":
        path = filedialog.askdirectory(title="Select COLMAP workspace folder")
    else:
        path = filedialog.askdirectory(title=f"Select {tool_name} folder")
    if path:
        proj = app._project_store.get_project(project_id)
        if proj:
            # Store as a source with the tool name as label
            from project_store import ProjectSource
            proj.sources.append(ProjectSource(
                label=tool_name, path=path, media_type="other",
            ))
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _show_project_detail(app, project_id)


# ── Export section ──

def _build_export_section(app, parent, proj):
    """Export project reports."""
    sec = CollapsibleSection(parent, "Export", expanded=True)
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    export_frame = ctk.CTkFrame(c, fg_color="transparent")
    export_frame.pack(fill="x", padx=6, pady=(4, 4))

    # Scope: this project vs all
    scope_row = ctk.CTkFrame(export_frame, fg_color="transparent")
    scope_row.pack(fill="x", pady=(2, 2))
    app._proj_export_scope = tk.StringVar(value="this")
    ctk.CTkRadioButton(
        scope_row, text="This project", variable=app._proj_export_scope, value="this",
    ).pack(side="left", padx=(0, 12))
    ctk.CTkRadioButton(
        scope_row, text="All projects", variable=app._proj_export_scope, value="all",
    ).pack(side="left")

    # Format
    fmt_row = ctk.CTkFrame(export_frame, fg_color="transparent")
    fmt_row.pack(fill="x", pady=(2, 4))
    app._proj_export_fmt = tk.StringVar(value="all")
    for label, val in [("All", "all"), ("MD", "md"), ("HTML", "html"), ("JSON", "json")]:
        ctk.CTkRadioButton(
            fmt_row, text=label, variable=app._proj_export_fmt, value=val,
        ).pack(side="left", padx=(0, 8))

    # Output directory
    dir_row = ctk.CTkFrame(export_frame, fg_color="transparent")
    dir_row.pack(fill="x", pady=(0, 4))
    app._proj_export_dir_entry = ctk.CTkEntry(dir_row)
    app._proj_export_dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
    app._proj_export_dir_entry.insert(0, str(Path(app._project_store.store_path).parent))
    ctk.CTkButton(
        dir_row, text="...", width=36,
        command=lambda: _browse_export_dir(app._proj_export_dir_entry),
    ).pack(side="right")

    ctk.CTkButton(
        export_frame, text="Export", width=80,
        command=lambda: _do_export(app, proj.id),
    ).pack(anchor="w", pady=(0, 4))


# ── Delete button (standalone at bottom) ──

def _build_delete_button(app, parent, proj):
    """Delete button with messagebox confirmation."""
    delete_frame = ctk.CTkFrame(parent, fg_color="transparent")
    delete_frame.pack(fill="x", padx=4, pady=(8, 6))

    def _on_delete():
        result = messagebox.askyesno(
            "Delete Project",
            f"Permanently delete '{proj.title}'?\n\nThis cannot be undone.",
            icon="warning",
        )
        if result:
            app._project_store.delete_project(proj.id)
            app._selected_project_id = None
            _refresh_project_list(app)
            _reset_detail_panel(app)

    ctk.CTkButton(
        delete_frame, text="Delete Project", width=120,
        fg_color="#dc2626", hover_color="#b91c1c",
        command=_on_delete,
    ).pack(anchor="w")


# ── Tags section ──

def _build_tags_section(app, parent, proj):
    """Tags as a list: type a tag, submit to add, select to remove.
    Capture method checkboxes also add tags automatically."""
    method_set = set(CAPTURE_METHODS)
    custom_tags = [t for t in proj.tags if t not in method_set]

    sec = CollapsibleSection(parent, "Tags", expanded=True)
    sec.pack(fill="x", padx=4, pady=(0, 6))
    c = sec.content

    # Entry + Add row
    entry_row = ctk.CTkFrame(c, fg_color="transparent")
    entry_row.pack(fill="x", padx=6, pady=(4, 4))
    tag_entry = ctk.CTkEntry(entry_row, placeholder_text="Add a tag...")
    tag_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _add_tag():
        text = tag_entry.get().strip()
        if not text or text in proj.tags:
            return
        proj.tags.append(text)
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()
        _refresh_project_list(app)
        _show_project_detail(app, proj.id)

    ctk.CTkButton(
        entry_row, text="Add", width=60, command=_add_tag,
    ).pack(side="right")
    tag_entry.bind("<Return>", lambda e: _add_tag())

    # Listed tags with selection for removal
    all_tags = proj.tags
    if all_tags:
        tags_container = ctk.CTkFrame(c, fg_color="transparent")
        tags_container.pack(fill="x", padx=6, pady=(0, 4))

        selected_indices = set()

        def _toggle_select(idx, row_frame):
            if idx in selected_indices:
                selected_indices.discard(idx)
                row_frame.configure(fg_color="transparent")
            else:
                selected_indices.add(idx)
                row_frame.configure(fg_color=("#dbeafe", "#1e3a5f"))

        for i, tag in enumerate(all_tags):
            # Mark method-origin tags visually
            is_method = tag in method_set
            tag_row = ctk.CTkFrame(tags_container, cursor="hand2")
            tag_row.pack(fill="x", pady=(0, 2))
            display = f"{tag}  (capture)" if is_method else tag
            lbl = ctk.CTkLabel(
                tag_row, text=display, font=("Consolas", 10), anchor="w",
                text_color="#9ca3af" if is_method else None,
            )
            lbl.pack(fill="x", padx=6, pady=2)
            for w in (tag_row, lbl):
                w.bind("<Button-1>", lambda e, idx=i, rf=tag_row: _toggle_select(idx, rf))

        def _remove_selected():
            if not selected_indices:
                return
            # Remove selected tags, also uncheck method boxes
            removed = {all_tags[i] for i in selected_indices}
            proj.tags = [t for t in proj.tags if t not in removed]
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _refresh_project_list(app)
            _show_project_detail(app, proj.id)

        ctk.CTkButton(
            tags_container, text="Remove selected", width=120,
            fg_color="#6b7280", command=_remove_selected,
        ).pack(anchor="w", pady=(4, 2))


def _reset_detail_panel(app):
    """Return detail panel to empty state."""
    app._proj_detail_scroll.pack_forget()
    for w in app._proj_detail_scroll.winfo_children():
        w.destroy()
    app._proj_empty_label.place(relx=0.5, rely=0.5, anchor="center")


# ── Export logic ──

def _do_export(app, current_project_id: str):
    """Run the export with current settings."""
    from project_exporters import export_markdown, export_html, export_json

    dir_entry = getattr(app, "_proj_export_dir_entry", None)
    if dir_entry:
        out_dir = Path(dir_entry.get().strip())
    else:
        out_dir = Path(app._project_store.store_path).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = getattr(app, "_proj_export_fmt", tk.StringVar(value="all")).get()
    scope = getattr(app, "_proj_export_scope", tk.StringVar(value="this")).get()
    store = app._project_store

    exported = []
    if scope == "this":
        # Export single project
        proj = store.get_project(current_project_id)
        if not proj:
            return
        suffix = proj.title.replace(" ", "_").lower()[:30]
        if fmt in ("all", "md"):
            exported.append(_export_single_md(proj, out_dir / f"{suffix}.md"))
        if fmt in ("all", "html"):
            exported.append(_export_single_html(proj, out_dir / f"{suffix}.html"))
        if fmt in ("all", "json"):
            exported.append(_export_single_json(proj, out_dir / f"{suffix}.json"))
    else:
        # Export all projects
        if fmt in ("all", "md"):
            exported.append(export_markdown(store, str(out_dir / "project_index.md")))
        if fmt in ("all", "html"):
            exported.append(export_html(store, str(out_dir / "project_index.html")))
        if fmt in ("all", "json"):
            exported.append(export_json(store, str(out_dir / "project_index.json")))

    app.log(f"Exported {len(exported)} file(s) to {out_dir}")

    # Open HTML in browser if generated
    html_files = [f for f in exported if str(f).endswith(".html")]
    if html_files:
        import webbrowser
        webbrowser.open(str(html_files[0]))


def _export_single_md(proj, path: Path) -> str:
    """Export a single project to markdown."""
    from project_store import STAGE_ORDER
    lines = [
        f"# {proj.title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if proj.scene_type:
        lines.append(f"**Subject:** {proj.scene_type}")
    if proj.tags:
        lines.append(f"**Tags:** {', '.join(proj.tags)}")
    lines.append("")

    if proj.sources:
        lines.append("## Sources")
        for src in proj.sources:
            exists = Path(src.path).exists()
            status = "" if exists else " [MISSING]"
            count = f" ({src.file_count} files)" if src.file_count else ""
            lines.append(f"- {src.label}: `{src.path}`{count}{status}")
        lines.append("")

    lines.append("## Pipeline")
    for sn in STAGE_ORDER:
        st = proj.stages.get(sn)
        status = st.status if st else "not_started"
        marker = "[x]" if status == "done" else "[-]" if status == "in_progress" else "[ ]"
        label = sn.replace("_", " ").title()
        lines.append(f"- {marker} {label}")
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
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    done = sum(1 for s in proj.stages.values() if s.status == "done")
    pct = int(done / len(STAGE_ORDER) * 100)

    stages_html = ""
    for sn in STAGE_ORDER:
        st = proj.stages.get(sn)
        status = st.status if st else "not_started"
        if status == "done":
            icon, color = "+", "#22c55e"
        elif status == "in_progress":
            icon, color = "o", "#f59e0b"
        else:
            icon, color = "-", "#6b7280"
        label = sn.replace("_", " ").title()
        stages_html += f'<span style="color:{color};margin-right:16px">{icon} {label}</span>'

    sources_html = ""
    if proj.sources:
        items = []
        for src in proj.sources:
            exists = Path(src.path).exists()
            clr = "#22c55e" if exists else "#ef4444"
            ind = "OK" if exists else "MISSING"
            cnt = f" ({src.file_count})" if src.file_count else ""
            items.append(
                f'<div style="font-family:monospace;font-size:12px;padding:2px 0">'
                f'<span style="color:{clr}">[{ind}]</span> '
                f'<b>{src.label}</b>: {src.path}{cnt}</div>'
            )
        sources_html = "<div style='margin:12px 0'>" + "\n".join(items) + "</div>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{proj.title}</title>
<style>body {{ background:#111827;color:#f9fafb;font-family:-apple-system,sans-serif;padding:24px;max-width:900px;margin:0 auto; }}</style>
</head><body>
<h1>{proj.title}</h1>
<p style="color:#9ca3af">Generated: {now}</p>
<div style="background:#374151;border-radius:4px;height:6px;margin:8px 0"><div style="background:#3b82f6;border-radius:4px;height:6px;width:{pct}%"></div></div>
<div style="margin:12px 0">{stages_html}</div>
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


def _browse_export_dir(entry):
    p = filedialog.askdirectory(title="Select export directory")
    if p:
        entry.delete(0, "end")
        entry.insert(0, p)


# ======================================================================
#  Drive scanner (inline in details pane)
# ======================================================================

def _on_scan(app):
    """Show scan UI inline in the details pane."""
    app._proj_empty_label.place_forget()
    scroll = app._proj_detail_scroll
    for w in scroll.winfo_children():
        w.destroy()
    scroll.pack(fill="both", expand=True, padx=4, pady=4)

    # Header
    ctk.CTkLabel(
        scroll, text="Scan for Projects",
        font=ctk.CTkFont(size=16, weight="bold"), anchor="w",
    ).pack(fill="x", padx=4, pady=(4, 8))

    # Path entry
    path_row = ctk.CTkFrame(scroll, fg_color="transparent")
    path_row.pack(fill="x", padx=4, pady=(0, 4))
    ctk.CTkLabel(path_row, text="Root:", width=40, anchor="w").pack(side="left")
    scan_entry = ctk.CTkEntry(path_row)
    scan_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
    scan_entry.insert(0, "D:\\")

    def _browse():
        p = filedialog.askdirectory(title="Select root to scan")
        if p:
            scan_entry.delete(0, "end")
            scan_entry.insert(0, p)

    ctk.CTkButton(path_row, text="...", width=36, command=_browse).pack(side="right")

    # Status
    status_label = ctk.CTkLabel(scroll, text="", text_color="#9ca3af")
    status_label.pack(fill="x", padx=4, pady=(2, 4))

    # Results container
    results_frame = ctk.CTkFrame(scroll, fg_color="transparent")
    results_frame.pack(fill="both", expand=True, padx=4)

    # Scan button + back button
    btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
    btn_row.pack(fill="x", padx=4, pady=(4, 4))

    def _run_scan():
        from project_scanner import scan_directory

        root = scan_entry.get().strip()
        if not root:
            return

        for w in results_frame.winfo_children():
            w.destroy()
        status_label.configure(text="Scanning...")
        scan_btn.configure(state="disabled")

        def _thread():
            results = scan_directory(
                root, max_depth=5,
                progress_callback=lambda msg: scroll.after(
                    0, lambda m=msg: status_label.configure(text=m)
                ),
            )
            scroll.after(0, lambda: _display_results(results))

        def _display_results(results):
            status_label.configure(text=f"Found {len(results)} project clusters")
            scan_btn.configure(state="normal")

            for res in results:
                row = ctk.CTkFrame(results_frame)
                row.pack(fill="x", pady=(0, 4))

                title = res.suggested_title()
                has_psx = "PSX" if res.psx_path else "no PSX"
                img_info = f"{len(res.image_dirs)} img dirs"
                mask_info = f"{len(res.mask_dirs)} mask dirs"

                ctk.CTkLabel(
                    row, text=f"{title}  ({has_psx}, {img_info}, {mask_info})",
                    font=("Consolas", 11), anchor="w",
                ).pack(side="left", fill="x", expand=True, padx=4, pady=4)

                ctk.CTkButton(
                    row, text="Import", width=70,
                    command=lambda r=res: _import_scan_result(app, r),
                ).pack(side="right", padx=4, pady=4)

            if not results:
                ctk.CTkLabel(
                    results_frame, text="No projects found in this directory",
                    text_color="#9ca3af",
                ).pack(pady=20)

        threading.Thread(target=_thread, daemon=True).start()

    scan_btn = ctk.CTkButton(btn_row, text="Scan", width=80, command=_run_scan)
    scan_btn.pack(side="left", padx=(0, 4))

    ctk.CTkButton(
        btn_row, text="Back", width=60, fg_color="#6b7280",
        command=lambda: _back_to_detail(app),
    ).pack(side="left")


def _import_scan_result(app, scan_result):
    """Create a project from a scan result."""
    from project_store import ProjectSource

    title = scan_result.suggested_title()
    proj = app._project_store.create_project(title)

    if scan_result.psx_path:
        proj.sources.append(ProjectSource(
            label="Metashape", path=scan_result.psx_path, media_type="other",
        ))

    for img_dir in scan_result.image_dirs:
        count = scan_result.image_counts.get(img_dir, 0)
        label = Path(img_dir).name
        proj.sources.append(ProjectSource(
            label=label, path=img_dir, media_type="images", file_count=count,
        ))

    for msk_dir in scan_result.mask_dirs:
        label = Path(msk_dir).name
        proj.sources.append(ProjectSource(
            label=label, path=msk_dir, media_type="masks",
        ))

    for vid in scan_result.video_files:
        proj.sources.append(ProjectSource(
            label=Path(vid).name, path=vid, media_type="video",
        ))

    proj.set_stage("extracted", "done")
    app._project_store.save()
    _refresh_project_list(app)
    _show_project_detail(app, proj.id)


def _back_to_detail(app):
    """Return from scan view to project detail or empty state."""
    if app._selected_project_id:
        _show_project_detail(app, app._selected_project_id)
    else:
        _reset_detail_panel(app)
