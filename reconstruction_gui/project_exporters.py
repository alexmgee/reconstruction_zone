"""
Project Exporters — Generate reports from the project store.

Outputs:
- Markdown (.md) — human-readable, works anywhere
- HTML (.html) — styled, collapsible sections, suitable for browser
- JSON (.json) — machine-readable, for other tools
"""

from datetime import datetime
from pathlib import Path
from typing import List

from project_store import Project, ProjectStore, STAGE_ORDER


METHOD_LABELS = ("Metashape", "RealityScan", "COLMAP")


def _source_stage_summary(src) -> str:
    """Human-readable stage for a single source."""
    if not src.stage:
        return "Not started"
    return src.stage.replace("_", " ").title()


def _source_stage_marker(src) -> str:
    """Markdown checkbox marker based on source stage."""
    if not src.stage:
        return "[ ]"
    try:
        idx = STAGE_ORDER.index(src.stage)
        if idx == len(STAGE_ORDER) - 1:
            return "[x]"
        return "[-]"
    except ValueError:
        return "[ ]"


def export_markdown(store: ProjectStore, output_path: str):
    """Export all projects to a markdown file."""
    projects = store.list_projects()
    lines = [
        f"# Photogrammetry Project Index",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total projects: {len(projects)}",
        f"",
        f"---",
        f"",
    ]

    for proj in projects:
        lines.append(f"## {proj.title}")
        if proj.tags:
            lines.append(f"**Tags:** {', '.join(proj.tags)}")
        lines.append("")

        if proj.sources:
            lines.append("**Sources:**")
            for src in proj.sources:
                exists = Path(src.path).exists()
                status = "" if exists else " [MISSING]"
                count = f" ({src.file_count} files)" if src.file_count else ""
                stage = f" — {_source_stage_summary(src)}" if src.label not in METHOD_LABELS else ""
                lines.append(f"- {src.label}: `{src.path}`{count}{status}{stage}")
            lines.append("")

        if proj.notes:
            lines.append(f"**Notes:** {proj.notes}")
            lines.append("")

        lines.append("---")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_html(store: ProjectStore, output_path: str):
    """Export all projects to a styled HTML file."""
    projects = store.list_projects()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards = []
    for proj in projects:
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
                    # Show stage pills for media sources
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
                    stage_html = '<div style="padding-left:20px;margin-top:2px">' + "".join(pills) + '</div>'
                items.append(
                    f'<div style="font-family:monospace;font-size:12px;padding:2px 0">'
                    f'<span style="color:{clr}">[{ind}]</span> '
                    f'<b>{src.label}</b>: {src.path}{cnt}</div>{stage_html}'
                )
            sources_html = "<div style='margin:8px 0'>" + "\n".join(items) + "</div>"

        tags_html = ""
        if proj.tags:
            tags_html = " ".join(
                f'<span style="background:#374151;color:#d1d5db;padding:2px 8px;'
                f'border-radius:4px;font-size:11px;margin-right:4px">{t}</span>'
                for t in proj.tags
            )

        cards.append(f"""
        <div style="background:#1f2937;border-radius:8px;padding:16px;margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <h2 style="margin:0;color:#f9fafb">{proj.title}</h2>
            </div>
            {sources_html}
            {f'<div style="margin-top:8px">{tags_html}</div>' if tags_html else ''}
            {f'<div style="color:#9ca3af;font-size:12px;margin-top:8px;font-style:italic">{proj.notes}</div>' if proj.notes else ''}
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Photogrammetry Project Index</title>
<style>
    body {{ background: #111827; color: #f9fafb; font-family: -apple-system, sans-serif; padding: 24px; max-width: 900px; margin: 0 auto; }}
    h1 {{ border-bottom: 1px solid #374151; padding-bottom: 12px; }}
</style>
</head>
<body>
<h1>Photogrammetry Project Index</h1>
<p style="color:#9ca3af">Generated: {now} | {len(projects)} projects</p>
{"".join(cards)}
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def export_json(store: ProjectStore, output_path: str):
    """Export all projects to a JSON file (for external tools)."""
    import json
    projects = store.list_projects()
    data = {
        "generated": datetime.now().isoformat(),
        "project_count": len(projects),
        "projects": [p.to_dict() for p in projects],
    }
    Path(output_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
