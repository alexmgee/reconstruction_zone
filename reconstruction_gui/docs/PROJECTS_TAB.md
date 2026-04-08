# Projects Tab

The Projects tab is a central registry for tracking photogrammetry projects across their full lifecycle. It records where your media lives on disk, what stage each source has reached in the reconstruction pipeline, which tools (Metashape, COLMAP, RealityScan) are involved, and any notes you want to attach. All data persists to a JSON file so it survives app restarts.

This tab is intentionally lightweight — it does not run any processing. It exists to answer: "What projects do I have, where are the files, and what's been done?"

## Layout

The Projects tab uses a two-column layout:

- **Left column** — Searchable project list
- **Right column** — Project detail panel

When you switch to any other tab, the detail panel is replaced by the shared preview panel.

---

## Project List

The left column shows all your projects in a scrollable list, sorted by most recently updated. Each row shows the project title, source count, and a preview of its tags.

Click any project to load its details in the right column. The search bar at the top filters projects by title, notes, or tags as you type.

Click **+ New** to create a project with a date-based default title (e.g. `2026-04-07 Untitled`) and open it for editing.

The store file path is shown at the bottom of the list — this is the JSON file where all project data is saved.

---

## Pipeline Stages

Before diving into the detail panel, it helps to understand how the Projects tab thinks about progress. Every media source you add to a project can be tracked through four sequential pipeline stages:

| Stage | Meaning |
|-------|---------|
| **Extracted** | Frames have been pulled from video |
| **Masked** | Object masks have been generated |
| **Aligned** | Cameras are aligned in COLMAP or Metashape |
| **Trained** | 3DGS model trained or dense reconstruction complete |

Stages are cumulative — marking a source as "Aligned" implies that "Extracted" and "Masked" are also done. You advance stages by clicking the stage buttons on each source row. Clicking a completed stage rolls it back.

This is purely a tracking tool — it doesn't trigger any processing. You update stages manually as you complete work in other tabs or external tools.

---

## Project Detail

When you select a project, the right column shows a scrollable detail view with the following sections.

### Project Info

The project's editable identity.

**Title** — Click to edit, saves on Enter or when you click away. The project list updates immediately.

**Capture method** — Checkboxes for common capture types: DRONE, DSLR, PHONE, 360. These are convenience shortcuts that automatically add or remove matching tags. You can also manage tags directly in the Tags section.

### Notes

A running list of notes attached to the project. Type a note in the entry field, click **Add** (or press Enter) to append it. To remove notes, click to select one or more (they highlight), then click **Remove selected**.

### Media

Tracks all source directories and reconstruction tool references associated with the project. Organized into two subsections:

#### Sources

Lists your image directories, video files, and mask directories. Each source shows:

- A health indicator — green if the path exists on disk, red if missing
- The label, path, and file count
- A **Relocate** button for when files move to a different drive
- Stage progression buttons (see [Pipeline Stages](#pipeline-stages) above)

To add a source, enter a descriptive label (e.g. "DJI video", "ERP frames", "Perspective crops"), click **Add**, and browse to the directory. The file count is recorded automatically.

#### Reconstruction Tools

Lists references to tool project files. These don't have stage progression — they represent the reconstruction tool itself, not a processing stage.

| Button | What it links to |
|--------|-----------------|
| **Metashape** | A `.psx` project file |
| **RealityScan** | A RealityScan project directory |
| **COLMAP** | A COLMAP workspace directory |

Each entry shows a health indicator so you can see at a glance if the project file is still accessible.

### Export

Generate a report of your project (or all projects) in one or more formats.

| Setting | Options | Default |
|---------|---------|---------|
| **Scope** | This project / All projects | This project |
| **Format** | All / MD / HTML / JSON | All |
| **Output directory** | (browse) | Same directory as the store file |

Click **Export** to generate. HTML reports open automatically in the browser. Markdown and JSON are written alongside.

- **Markdown** — Human-readable summary with source paths, stages, tags, and notes
- **HTML** — Styled dark-themed page with stage indicator pills, suitable for sharing
- **JSON** — Machine-readable project data for external tools

### Tags

Free-form tags for organizing projects. Type a tag and click **Add** (or press Enter). Click tags to select them, then **Remove selected** to delete.

Tags added via the capture method checkboxes (in Project Info) appear here marked with `(capture)` in grey. They can be removed from either location.

### Delete Project

A red **Delete Project** button at the bottom of the detail view. Shows a confirmation dialog before permanently removing the project from the store.

---

## Drive Scanner

The scanner helps you discover existing photogrammetry projects on disk so you don't have to add them manually. It is the default view in the right column when no project is selected.

### Using the scanner

1. Set the root directory to scan — this can be an entire drive (`D:\`) or a specific project folder (`D:\scans\house`). The scanner walks every non-hidden directory under the root up to 5 levels deep, so a narrower root is faster.
2. Click **Scan** — the scanner searches for reconstruction tool projects, directories containing images, mask directories, and video files
3. Review the results — each discovered project shows a suggested title, which tools were found, and how many image/mask directories exist nearby. Click a result to expand it and see the full path breakdown.
4. Click **Import** on any result to create a project from it — the scanner automatically populates tool references, image sources with file counts, mask directories, and nearby video files

The scanner recognizes three reconstruction tools:

| Tool | On-disk indicator |
|------|-------------------|
| **Metashape** | `.psx` project file |
| **RealityScan** | `.rsproj` or `.rcproj` project file |
| **COLMAP** | `sparse/` directory containing `cameras.txt` or `cameras.bin` |

The scanner only reports directories that contain a recognized tool project. Image directories, mask directories, and video files are associated with the nearest tool project within 3 directory levels. Multiple tools in the same directory tree (e.g. a Metashape `.psx` alongside a COLMAP `sparse/`) are merged into a single result.

---

## Typical workflow: track a new project

```
1. Projects tab     → Click + New
2. Project Info     → Rename to your project name
3.                  → Check capture method (drone, 360, etc.)
4. Media            → Add sources: label + browse for each directory
5.                  → Link reconstruction tools (Metashape .psx, COLMAP, etc.)
6. Notes            → Add notes about the capture session
7. Tags             → Add tags for organization (site name, client, etc.)
```

## Typical workflow: update pipeline progress

```
1. Select project   → Click on it in the list
2. Media            → Find the source that just completed a stage
3.                  → Click the next stage button (e.g. Masked)
4.                  → All prior stages turn green automatically
```

## Typical workflow: discover existing projects

```
1. Projects tab     → The scanner is shown by default in the right column
2.                  → Set root directory (e.g. D:\)
3.                  → Click Scan and wait for results
4.                  → Expand results to review paths and contents
5.                  → Click Import on each discovery to create a project
```

## Typical workflow: export a project report

```
1. Select project   → Click on it in the list
2. Export           → Choose scope and format
3.                  → Set output directory
4.                  → Click Export
5.                  → HTML opens in browser automatically
```
