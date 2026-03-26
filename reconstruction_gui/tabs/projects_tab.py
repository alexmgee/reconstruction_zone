"""
Projects Tab — Central registry for photogrammetry projects.

Provides a searchable project list (left) and detail/edit panel (right).
"""

import customtkinter as ctk
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Optional

from widgets import Section, CollapsibleSection, Tooltip


def build_projects_tab(app, parent):
    """Build the Projects tab UI. Called once during app init."""
    app._selected_project_id = None

    # Initialize store
    from project_store import ProjectStore
    store_path = app._prefs.get("tracker_store_path", "D:\\tracker.json")
    app._project_store = ProjectStore(store_path)

    # Two-column layout: list (left) + detail (right)
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(fill="both", expand=True, padx=4, pady=4)
    container.grid_columnconfigure(0, weight=1, minsize=280)
    container.grid_columnconfigure(1, weight=2, minsize=400)
    container.grid_rowconfigure(0, weight=1)

    # -- Left panel: project list --
    left = ctk.CTkFrame(container)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    _build_list_panel(app, left)

    # -- Right panel: project detail --
    right = ctk.CTkFrame(container)
    right.grid(row=0, column=1, sticky="nsew")
    _build_detail_panel(app, right)

    # Initial list population (deferred so UI is built first)
    parent.after(100, lambda: _refresh_project_list(app))


def _build_list_panel(app, parent):
    """Left panel: search bar, filter, scrollable project list."""
    # -- Header row: title + buttons --
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.pack(fill="x", padx=6, pady=(6, 2))
    ctk.CTkLabel(
        header, text="Projects",
        font=ctk.CTkFont(size=14, weight="bold"),
    ).pack(side="left")
    app._proj_new_btn = ctk.CTkButton(
        header, text="+ New", width=60,
        command=lambda: _on_new_project(app),
    )
    app._proj_new_btn.pack(side="right")

    # -- Search bar --
    app._proj_search_var = tk.StringVar()
    app._proj_search_var.trace_add("write", lambda *_: _refresh_project_list(app))
    search = ctk.CTkEntry(
        parent, placeholder_text="Search projects...",
        textvariable=app._proj_search_var,
    )
    search.pack(fill="x", padx=6, pady=(2, 4))

    # -- Filter row --
    filter_frame = ctk.CTkFrame(parent, fg_color="transparent")
    filter_frame.pack(fill="x", padx=6, pady=(0, 4))
    app._proj_filter_var = tk.StringVar(value="all")
    for label, val in [("All", "all"), ("Active", "active"), ("Archived", "archived")]:
        ctk.CTkRadioButton(
            filter_frame, text=label, variable=app._proj_filter_var,
            value=val, command=lambda: _refresh_project_list(app),
        ).pack(side="left", padx=(0, 8))

    # -- Scrollable project list --
    app._proj_list_frame = ctk.CTkScrollableFrame(parent)
    app._proj_list_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
    app._proj_list_widgets = []  # track for refresh

    # -- Settings row at bottom --
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


def _build_detail_panel(app, parent):
    """Right panel: project detail view with stages, sources, notes."""
    app._proj_detail_parent = parent

    # Placeholder when nothing selected
    app._proj_empty_label = ctk.CTkLabel(
        parent, text="Select a project or create a new one",
        font=ctk.CTkFont(size=13), text_color="#9ca3af",
    )
    app._proj_empty_label.place(relx=0.5, rely=0.5, anchor="center")

    # Scrollable detail (hidden until project selected)
    app._proj_detail_scroll = ctk.CTkScrollableFrame(parent)
    # Will be packed when a project is selected


# ── New Project Dialog ──

def _on_new_project(app):
    """Create a new project via a popup dialog."""
    dialog = ctk.CTkToplevel(app)
    dialog.title("New Project")
    dialog.geometry("420x280")
    dialog.transient(app)
    dialog.grab_set()

    ctk.CTkLabel(dialog, text="Project Title:", anchor="w").pack(
        fill="x", padx=16, pady=(16, 2),
    )
    title_entry = ctk.CTkEntry(dialog, placeholder_text="e.g. Grand Island Interior")
    title_entry.pack(fill="x", padx=16, pady=(0, 8))
    title_entry.focus_set()

    ctk.CTkLabel(dialog, text="Scene Type:", anchor="w").pack(
        fill="x", padx=16, pady=(0, 2),
    )
    scene_var = tk.StringVar(value="interior")
    scene_menu = ctk.CTkOptionMenu(
        dialog, variable=scene_var,
        values=["interior", "exterior", "object", "landscape", "aerial", "other"],
    )
    scene_menu.pack(fill="x", padx=16, pady=(0, 8))

    ctk.CTkLabel(dialog, text="Tags (comma-separated):", anchor="w").pack(
        fill="x", padx=16, pady=(0, 2),
    )
    tags_entry = ctk.CTkEntry(dialog, placeholder_text="e.g. edgeworks, 360, client-work")
    tags_entry.pack(fill="x", padx=16, pady=(0, 16))

    def _create():
        t = title_entry.get().strip()
        if not t:
            return
        proj = app._project_store.create_project(t, scene_var.get())
        raw_tags = tags_entry.get().strip()
        if raw_tags:
            proj.tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
            app._project_store.save()
        dialog.destroy()
        _refresh_project_list(app)
        _show_project_detail(app, proj.id)

    ctk.CTkButton(dialog, text="Create Project", command=_create).pack(
        padx=16, pady=(0, 16),
    )
    dialog.bind("<Return>", lambda e: _create())


# ── Project List ──

def _refresh_project_list(app):
    """Rebuild the project list from the store."""
    if not app._project_store:
        return

    # Clear existing widgets
    for w in app._proj_list_widgets:
        w.destroy()
    app._proj_list_widgets.clear()

    # Determine filters
    filt = app._proj_filter_var.get()
    search = app._proj_search_var.get().strip()

    if filt == "all":
        projects = app._project_store.list_projects(
            include_archived=True, search=search,
        )
    elif filt == "archived":
        projects = app._project_store.list_projects(
            include_archived=True, search=search,
        )
        projects = [p for p in projects if p.archived]
    else:  # "active"
        projects = app._project_store.list_projects(
            include_archived=False, search=search,
        )

    for proj in projects:
        row = _create_project_list_row(app, proj)
        row.pack(fill="x", pady=(0, 2))
        app._proj_list_widgets.append(row)

    if not projects:
        lbl = ctk.CTkLabel(
            app._proj_list_frame,
            text="No projects found" if search else "No projects yet \u2014 click + New",
            text_color="#9ca3af",
        )
        lbl.pack(pady=20)
        app._proj_list_widgets.append(lbl)


def _create_project_list_row(app, proj) -> ctk.CTkFrame:
    """Create a clickable row for one project in the list."""
    from project_store import STAGE_ORDER

    row = ctk.CTkFrame(app._proj_list_frame, cursor="hand2")

    title_lbl = ctk.CTkLabel(
        row, text=proj.title,
        font=ctk.CTkFont(size=12, weight="bold"),
        anchor="w",
    )
    title_lbl.pack(fill="x", padx=8, pady=(4, 0))

    # Stage indicator: compact text showing current stage
    stage = proj.current_stage().replace("_", " ").title()
    done_count = sum(1 for s in proj.stages.values() if s.status == "done")
    subtitle = f"{stage}  |  {done_count}/{len(STAGE_ORDER)} stages"
    if proj.scene_type:
        subtitle = f"{proj.scene_type}  |  {subtitle}"

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


# ── Project Detail Panel ──

def _show_project_detail(app, project_id: str):
    """Populate the right panel with full project details."""
    from project_store import STAGE_ORDER

    proj = app._project_store.get_project(project_id)
    if not proj:
        return

    app._selected_project_id = project_id
    _refresh_project_list(app)  # Update selection highlight

    # Hide placeholder, destroy old detail scroll, create fresh one
    app._proj_empty_label.place_forget()
    if app._proj_detail_scroll.winfo_exists():
        app._proj_detail_scroll.destroy()
    app._proj_detail_scroll = ctk.CTkScrollableFrame(app._proj_detail_parent)
    app._proj_detail_scroll.pack(fill="both", expand=True, padx=4, pady=4)
    scroll = app._proj_detail_scroll

    # -- Title + scene type --
    header = ctk.CTkFrame(scroll, fg_color="transparent")
    header.pack(fill="x", padx=4, pady=(4, 8))
    ctk.CTkLabel(
        header, text=proj.title,
        font=ctk.CTkFont(size=16, weight="bold"), anchor="w",
    ).pack(side="left")
    if proj.scene_type:
        ctk.CTkLabel(
            header, text=f"  [{proj.scene_type}]",
            font=("Consolas", 11), text_color="#9ca3af", anchor="w",
        ).pack(side="left")

    # Archive / Delete buttons
    btn_frame = ctk.CTkFrame(header, fg_color="transparent")
    btn_frame.pack(side="right")
    ctk.CTkButton(
        btn_frame, text="Archive", width=70, fg_color="#6b7280",
        command=lambda: _archive_project(app, project_id),
    ).pack(side="left", padx=(0, 4))
    ctk.CTkButton(
        btn_frame, text="Delete", width=60, fg_color="#dc2626", hover_color="#b91c1c",
        command=lambda: _delete_project(app, project_id),
    ).pack(side="left")

    # -- Stages --
    stages_sec = Section(scroll, "Stages")
    stages_sec.pack(fill="x", padx=4, pady=(0, 6))
    _build_stages_grid(app, stages_sec.content, proj)

    # -- Sources --
    sources_sec = Section(scroll, "Sources", subtitle=f"{len(proj.sources)} entries")
    sources_sec.pack(fill="x", padx=4, pady=(0, 6))
    _build_sources_list(app, sources_sec.content, proj)

    # Add source button
    ctk.CTkButton(
        sources_sec.content, text="+ Add Source", width=120,
        command=lambda: _add_source_dialog(app, project_id),
    ).pack(pady=(4, 4))

    # -- Metashape --
    meta_sec = Section(scroll, "Metashape Project")
    meta_sec.pack(fill="x", padx=4, pady=(0, 6))
    _build_metashape_section(app, meta_sec.content, proj)

    # -- Notes --
    notes_sec = Section(scroll, "Notes")
    notes_sec.pack(fill="x", padx=4, pady=(0, 6))
    app._proj_notes_text = ctk.CTkTextbox(notes_sec.content, height=100)
    app._proj_notes_text.pack(fill="x", padx=4, pady=4)
    if proj.notes:
        app._proj_notes_text.insert("1.0", proj.notes)
    ctk.CTkButton(
        notes_sec.content, text="Save Notes", width=100,
        command=lambda: _save_notes(app, project_id),
    ).pack(pady=(0, 4))

    # -- Tags --
    tags_sec = Section(scroll, "Tags")
    tags_sec.pack(fill="x", padx=4, pady=(0, 6))
    app._proj_tags_entry = ctk.CTkEntry(
        tags_sec.content,
        placeholder_text="comma-separated tags",
    )
    app._proj_tags_entry.pack(fill="x", padx=4, pady=4)
    if proj.tags:
        app._proj_tags_entry.insert(0, ", ".join(proj.tags))
    ctk.CTkButton(
        tags_sec.content, text="Save Tags", width=100,
        command=lambda: _save_tags(app, project_id),
    ).pack(pady=(0, 4))

    # -- Path Health --
    health_sec = Section(scroll, "Path Health")
    health_sec.pack(fill="x", padx=4, pady=(0, 6))
    _build_path_health(app, health_sec.content, proj)


# ── Stages Grid ──

def _build_stages_grid(app, parent, proj):
    """Build a 2-column grid of stage checkboxes."""
    from project_store import STAGE_ORDER

    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(fill="x", padx=4, pady=4)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)

    for i, stage_name in enumerate(STAGE_ORDER):
        stage = proj.stages.get(stage_name, None)
        status = stage.status if stage else "not_started"

        row = i // 2
        col = i % 2

        stage_frame = ctk.CTkFrame(frame, fg_color="transparent")
        stage_frame.grid(row=row, column=col, sticky="w", padx=4, pady=2)

        var = tk.BooleanVar(value=(status == "done"))
        display_name = stage_name.replace("_", " ").title()
        cb = ctk.CTkCheckBox(
            stage_frame, text=display_name, variable=var,
            command=lambda sn=stage_name, v=var: _toggle_stage(app, proj.id, sn, v),
        )
        cb.pack(side="left")

        if status == "in_progress":
            ctk.CTkLabel(
                stage_frame, text="(in progress)",
                font=("Consolas", 9), text_color="#f59e0b",
            ).pack(side="left", padx=(4, 0))


def _toggle_stage(app, project_id, stage_name, var):
    """Toggle a stage between done and not_started."""
    proj = app._project_store.get_project(project_id)
    if not proj:
        return
    new_status = "done" if var.get() else "not_started"
    proj.set_stage(stage_name, new_status)
    app._project_store.save()
    _refresh_project_list(app)  # Update subtitle


# ── Sources List ──

def _build_sources_list(app, parent, proj):
    """List all source paths with file counts and existence indicators."""
    for i, src in enumerate(proj.sources):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=2)

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

        # Relocate button (for moved files)
        ctk.CTkButton(
            row, text="Relocate", width=70, fg_color="#6b7280",
            command=lambda idx=i: _relocate_source(app, proj.id, idx),
        ).pack(side="right", padx=(4, 0))


def _relocate_source(app, project_id, source_index):
    """Browse for a new path for a source that has moved."""
    new_path = filedialog.askdirectory(title="Select new location")
    if new_path:
        app._project_store.relocate_source(project_id, source_index, new_path)
        _show_project_detail(app, project_id)  # Refresh


# ── Add Source Dialog ──

def _add_source_dialog(app, project_id):
    """Dialog to add a new source to a project."""
    dialog = ctk.CTkToplevel(app)
    dialog.title("Add Source")
    dialog.geometry("500x240")
    dialog.transient(app)
    dialog.grab_set()

    ctk.CTkLabel(dialog, text="Label:", anchor="w").pack(fill="x", padx=16, pady=(16, 2))
    label_entry = ctk.CTkEntry(dialog, placeholder_text="e.g. ERP frames, Fuji stills, DJI video")
    label_entry.pack(fill="x", padx=16, pady=(0, 8))

    ctk.CTkLabel(dialog, text="Media Type:", anchor="w").pack(fill="x", padx=16, pady=(0, 2))
    type_var = tk.StringVar(value="images")
    ctk.CTkOptionMenu(
        dialog, variable=type_var,
        values=["images", "video", "masks", "other"],
    ).pack(fill="x", padx=16, pady=(0, 8))

    path_frame = ctk.CTkFrame(dialog, fg_color="transparent")
    path_frame.pack(fill="x", padx=16, pady=(0, 8))
    path_entry = ctk.CTkEntry(path_frame, placeholder_text="Path to directory or file")
    path_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

    def _browse():
        if type_var.get() == "video":
            p = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")],
            )
        else:
            p = filedialog.askdirectory(title="Select Directory")
        if p:
            path_entry.delete(0, "end")
            path_entry.insert(0, p)

    ctk.CTkButton(path_frame, text="Browse", width=70, command=_browse).pack(side="right")

    def _add():
        label = label_entry.get().strip()
        path = path_entry.get().strip()
        if not label or not path:
            return
        proj = app._project_store.get_project(project_id)
        if proj:
            proj.add_source(label, path, type_var.get())
            app._project_store.save()
        dialog.destroy()
        _show_project_detail(app, project_id)

    ctk.CTkButton(dialog, text="Add Source", command=_add).pack(padx=16, pady=(0, 16))


# ── Metashape Section ──

def _build_metashape_section(app, parent, proj):
    """Show Metashape project path with browse button."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=4)

    path_text = proj.metashape_path or "(none)"
    exists = Path(proj.metashape_path).exists() if proj.metashape_path else False
    color = "#22c55e" if exists else ("#ef4444" if proj.metashape_path else "#9ca3af")

    ctk.CTkLabel(
        row, text=path_text, font=("Consolas", 10),
        text_color=color, anchor="w",
    ).pack(side="left", fill="x", expand=True)

    ctk.CTkButton(
        row, text="Browse", width=70,
        command=lambda: _set_metashape_path(app, proj.id),
    ).pack(side="right")


def _set_metashape_path(app, project_id):
    """Browse for a .psx file."""
    path = filedialog.askopenfilename(
        title="Select Metashape Project",
        filetypes=[("Metashape Project", "*.psx"), ("All Files", "*.*")],
    )
    if path:
        proj = app._project_store.get_project(project_id)
        if proj:
            proj.metashape_path = path
            proj.updated_at = datetime.now().isoformat()
            app._project_store.save()
            _show_project_detail(app, project_id)


# ── Notes & Tags ──

def _save_notes(app, project_id):
    """Save notes from the text box to the project."""
    proj = app._project_store.get_project(project_id)
    if proj and hasattr(app, "_proj_notes_text"):
        proj.notes = app._proj_notes_text.get("1.0", "end-1c").strip()
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()


def _save_tags(app, project_id):
    """Save tags from the entry to the project."""
    proj = app._project_store.get_project(project_id)
    if proj and hasattr(app, "_proj_tags_entry"):
        raw = app._proj_tags_entry.get().strip()
        proj.tags = [t.strip() for t in raw.split(",") if t.strip()]
        proj.updated_at = datetime.now().isoformat()
        app._project_store.save()


# ── Path Health ──

def _build_path_health(app, parent, proj):
    """Show which paths exist and which are missing."""
    health = app._project_store.validate_paths(proj.id)
    if not health:
        ctk.CTkLabel(
            parent, text="No paths to check", text_color="#9ca3af",
        ).pack(padx=4, pady=4)
        return

    for path, exists in health.items():
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=1)
        indicator = "+" if exists else "MISSING"
        color = "#22c55e" if exists else "#ef4444"
        ctk.CTkLabel(
            row, text=indicator, font=("Consolas", 10),
            text_color=color, width=60, anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            row, text=path, font=("Consolas", 9), anchor="w",
        ).pack(side="left", fill="x")


# ── Archive / Delete ──

def _archive_project(app, project_id):
    """Archive a project (soft delete)."""
    app._project_store.archive_project(project_id)
    app._selected_project_id = None
    _refresh_project_list(app)
    # Reset detail panel
    app._proj_detail_scroll.pack_forget()
    app._proj_empty_label.place(relx=0.5, rely=0.5, anchor="center")


def _delete_project(app, project_id):
    """Permanently delete a project (with confirmation)."""
    proj = app._project_store.get_project(project_id)
    if not proj:
        return
    # Simple confirmation via toplevel
    dialog = ctk.CTkToplevel(app)
    dialog.title("Confirm Delete")
    dialog.geometry("350x120")
    dialog.transient(app)
    dialog.grab_set()
    ctk.CTkLabel(
        dialog, text=f"Permanently delete '{proj.title}'?",
    ).pack(pady=(16, 8))
    btn_row = ctk.CTkFrame(dialog, fg_color="transparent")
    btn_row.pack()
    ctk.CTkButton(
        btn_row, text="Cancel", width=80, fg_color="#6b7280",
        command=dialog.destroy,
    ).pack(side="left", padx=8)
    ctk.CTkButton(
        btn_row, text="Delete", width=80, fg_color="#dc2626", hover_color="#b91c1c",
        command=lambda: (_do_delete(app, project_id), dialog.destroy()),
    ).pack(side="left", padx=8)


def _do_delete(app, project_id):
    app._project_store.delete_project(project_id)
    app._selected_project_id = None
    _refresh_project_list(app)
    app._proj_detail_scroll.pack_forget()
    app._proj_empty_label.place(relx=0.5, rely=0.5, anchor="center")
