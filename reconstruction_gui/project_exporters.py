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


def export_markdown(store: ProjectStore, output_path: str, include_archived: bool = False):
    """Export all projects to a markdown file."""
    projects = store.list_projects(include_archived=include_archived)
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
        stage = proj.current_stage().replace("_", " ").title()
        done = sum(1 for s in proj.stages.values() if s.status == "done")
        lines.append(f"## {proj.title}")
        if proj.scene_type:
            lines.append(f"**Type:** {proj.scene_type}")
        lines.append(f"**Stage:** {stage} ({done}/{len(STAGE_ORDER)} complete)")
        if proj.tags:
            lines.append(f"**Tags:** {', '.join(proj.tags)}")
        lines.append("")

        if proj.sources:
            lines.append("**Sources:**")
            for src in proj.sources:
                exists = Path(src.path).exists()
                status = "" if exists else " [MISSING]"
                count = f" ({src.file_count} files)" if src.file_count else ""
                lines.append(f"- {src.label}: `{src.path}`{count}{status}")
            lines.append("")

        if proj.metashape_path:
            exists = Path(proj.metashape_path).exists()
            status = "" if exists else " [MISSING]"
            lines.append(f"**Metashape:** `{proj.metashape_path}`{status}")
            lines.append("")

        if proj.notes:
            lines.append(f"**Notes:** {proj.notes}")
            lines.append("")

        lines.append("**Stages:**")
        for sn in STAGE_ORDER:
            st = proj.stages.get(sn)
            status = st.status if st else "not_started"
            marker = "[x]" if status == "done" else "[-]" if status == "in_progress" else "[ ]"
            label = sn.replace("_", " ").title()
            note = f" -- {st.notes}" if st and st.notes else ""
            lines.append(f"- {marker} {label}{note}")
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_html(store: ProjectStore, output_path: str, include_archived: bool = False):
    """Export all projects to a styled HTML file."""
    projects = store.list_projects(include_archived=include_archived)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    cards = []
    for proj in projects:
        stage = proj.current_stage().replace("_", " ").title()
        done = sum(1 for s in proj.stages.values() if s.status == "done")
        pct = int(done / len(STAGE_ORDER) * 100)

        sources_html = ""
        if proj.sources:
            src_items = []
            for src in proj.sources:
                exists = Path(src.path).exists()
                color = "#22c55e" if exists else "#ef4444"
                indicator = "OK" if exists else "MISSING"
                count = f" ({src.file_count})" if src.file_count else ""
                src_items.append(
                    f'<div style="font-family:monospace;font-size:12px;padding:2px 0">'
                    f'<span style="color:{color}">[{indicator}]</span> '
                    f'<b>{src.label}</b>: {src.path}{count}</div>'
                )
            sources_html = "<div style='margin:8px 0'>" + "\n".join(src_items) + "</div>"

        stages_html = ""
        for sn in STAGE_ORDER:
            st = proj.stages.get(sn)
            status = st.status if st else "not_started"
            if status == "done":
                icon, color = "&#x2705;", "#22c55e"
            elif status == "in_progress":
                icon, color = "&#x1F7E1;", "#f59e0b"
            else:
                icon, color = "&#x2B1C;", "#6b7280"
            label = sn.replace("_", " ").title()
            stages_html += f'<span style="color:{color};margin-right:12px">{icon} {label}</span>'

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
                <span style="color:#9ca3af;font-size:13px">{proj.scene_type or ""}</span>
            </div>
            <div style="background:#374151;border-radius:4px;height:6px;margin:8px 0">
                <div style="background:#3b82f6;border-radius:4px;height:6px;width:{pct}%"></div>
            </div>
            <div style="color:#9ca3af;font-size:12px;margin-bottom:8px">{stage} -- {done}/{len(STAGE_ORDER)} complete</div>
            <div style="margin:8px 0">{stages_html}</div>
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


def export_json(store: ProjectStore, output_path: str, include_archived: bool = False):
    """Export all projects to a JSON file (for external tools)."""
    import json
    projects = store.list_projects(include_archived=include_archived)
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
