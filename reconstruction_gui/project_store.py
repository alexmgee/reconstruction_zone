"""
Project Tracker — Data Model and Store
=======================================
Central registry for photogrammetry projects. Tracks per-source pipeline
stages, media paths across drives, and tool references (Metashape, COLMAP, etc.).

GUI-independent — usable from CLI, Metashape scripts, or the Projects tab.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


STAGE_ORDER = [
    "extracted",
    "masked",
    "aligned",
    "trained",
]

MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def count_media_files(directory: str) -> int:
    """Count media files in a directory. Shared helper to avoid duplication."""
    p = Path(directory)
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in MEDIA_EXTS)


@dataclass
class ProjectSource:
    """A single source media directory or file associated with a project.
    Sources are static inputs — raw capture data that doesn't change."""
    label: str
    path: str
    media_type: str  # "images", "video", "masks", "other"
    file_count: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "media_type": self.media_type,
            "file_count": self.file_count,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectSource":
        return cls(
            label=data.get("label", ""),
            path=data.get("path", ""),
            media_type=data.get("media_type", "other"),
            file_count=data.get("file_count", 0),
            notes=data.get("notes", ""),
        )


@dataclass
class ProjectWorkDir:
    """A working directory derived from processing source data.
    Tracks pipeline stage and optionally which source it came from."""
    label: str
    path: str
    stage: str = ""  # current stage: "" (none), or one of STAGE_ORDER
    file_count: int = 0
    derived_from: str = ""  # freeform text: which source this came from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "stage": self.stage,
            "file_count": self.file_count,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectWorkDir":
        return cls(
            label=data.get("label", ""),
            path=data.get("path", ""),
            stage=data.get("stage", ""),
            file_count=data.get("file_count", 0),
            derived_from=data.get("derived_from", ""),
        )


@dataclass
class Project:
    """A single photogrammetry project."""
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    sources: List[ProjectSource] = field(default_factory=list)
    work_dirs: List[ProjectWorkDir] = field(default_factory=list)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    root_dir: str = ""
    static_masks_dir: str = ""  # Path to static_masks/ folder (if any)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": [s.to_dict() for s in self.sources],
            "work_dirs": [w.to_dict() for w in self.work_dirs],
            "notes": self.notes,
            "tags": self.tags,
            "root_dir": self.root_dir,
            "static_masks_dir": self.static_masks_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        # Backward compat: migrate old sources with stage to work_dirs
        sources_raw = data.get("sources", [])
        sources = []
        migrated_work_dirs = []
        for s in sources_raw:
            if s.get("stage"):
                # Old source with stage data — migrate to work_dir
                migrated_work_dirs.append(ProjectWorkDir(
                    label=s.get("label", ""),
                    path=s.get("path", ""),
                    stage=s.get("stage", ""),
                    file_count=s.get("file_count", 0),
                ))
            else:
                sources.append(ProjectSource.from_dict(s))

        work_dirs = [ProjectWorkDir.from_dict(w) for w in data.get("work_dirs", [])]
        work_dirs = migrated_work_dirs + work_dirs

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            sources=sources,
            work_dirs=work_dirs,
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            root_dir=data.get("root_dir", ""),
            static_masks_dir=data.get("static_masks_dir", ""),
        )

    @classmethod
    def create(cls, title: str, root_dir: str = "") -> "Project":
        """Create a new project."""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now,
            root_dir=root_dir,
        )

    def add_source(self, label: str, path: str, media_type: str, notes: str = "") -> ProjectSource:
        """Add a source media path. Auto-counts files if path is a directory."""
        p = Path(path)
        file_count = count_media_files(str(p))
        source = ProjectSource(
            label=label, path=str(p), media_type=media_type,
            file_count=file_count, notes=notes,
        )
        self.sources.append(source)
        self.updated_at = datetime.now().isoformat()
        return source

    def add_work_dir(self, label: str, path: str, derived_from: str = "",
                     stage: str = "") -> "ProjectWorkDir":
        """Add a working directory. Auto-counts files if path is a directory."""
        p = Path(path)
        file_count = count_media_files(str(p))
        wd = ProjectWorkDir(
            label=label, path=str(p), stage=stage,
            file_count=file_count, derived_from=derived_from,
        )
        self.work_dirs.append(wd)
        self.updated_at = datetime.now().isoformat()
        return wd


class ProjectStore:
    """Persistent collection of projects. Saves to a JSON file.

    Usage::

        store = ProjectStore("D:/tracker.json")
        proj = store.create_project("My Scan")
        proj.add_source("ERP frames", "D:/scans/360/frames", "images")
        store.save()

        # Later...
        store = ProjectStore("D:/tracker.json")
        for p in store.list_projects():
            print(p.title, len(p.sources), "sources")
    """

    VERSION = 1

    def __init__(self, store_path: str = "D:\\tracker.json"):
        self.store_path = Path(store_path)
        self._projects: Dict[str, Project] = {}
        self._load()

    def _load(self):
        """Load projects from disk."""
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text(encoding="utf-8"))
                for p_data in raw.get("projects", []):
                    proj = Project.from_dict(p_data)
                    self._projects[proj.id] = proj
            except Exception as e:
                print(f"Warning: failed to load tracker: {e}")

    def save(self):
        """Persist all projects to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "store_path": str(self.store_path),
            "projects": [p.to_dict() for p in self._projects.values()],
        }
        self.store_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- CRUD --

    def create_project(self, title: str, root_dir: str = "") -> Project:
        """Create and register a new project."""
        proj = Project.create(title, root_dir=root_dir)
        self._projects[proj.id] = proj
        self.save()
        return proj

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return self._projects.get(project_id)

    def delete_project(self, project_id: str):
        """Permanently remove a project from the store."""
        self._projects.pop(project_id, None)
        self.save()

    def list_projects(
        self,
        tag_filter: str = "",
        search: str = "",
    ) -> List[Project]:
        """List projects with optional filters.

        Args:
            tag_filter: Only show projects that have this tag
            search: Case-insensitive substring match on title or notes
        """
        results = []
        for p in self._projects.values():
            if tag_filter and tag_filter not in p.tags:
                continue
            if search:
                haystack = (p.title + " " + p.notes + " " + " ".join(p.tags)).lower()
                if search.lower() not in haystack:
                    continue
            results.append(p)
        results.sort(key=lambda p: p.updated_at or p.created_at, reverse=True)
        return results

    def relocate_source(self, project_id: str, source_index: int, new_path: str) -> bool:
        """Update a source's path (for when files move between drives)."""
        proj = self._projects.get(project_id)
        if not proj or source_index >= len(proj.sources):
            return False
        src = proj.sources[source_index]
        src.path = new_path
        src.file_count = count_media_files(new_path)
        proj.updated_at = datetime.now().isoformat()
        self.save()
        return True

    def relocate_work_dir(self, project_id: str, work_dir_index: int, new_path: str) -> bool:
        """Update a working directory's path."""
        proj = self._projects.get(project_id)
        if not proj or work_dir_index >= len(proj.work_dirs):
            return False
        wd = proj.work_dirs[work_dir_index]
        wd.path = new_path
        wd.file_count = count_media_files(new_path)
        proj.updated_at = datetime.now().isoformat()
        self.save()
        return True

    def validate_paths(self, project_id: str) -> Dict[str, bool]:
        """Check which paths in a project still exist on disk.

        Returns dict mapping path -> exists (bool).
        """
        proj = self._projects.get(project_id)
        if not proj:
            return {}
        result = {src.path: Path(src.path).exists() for src in proj.sources}
        result.update({wd.path: Path(wd.path).exists() for wd in proj.work_dirs})
        return result
