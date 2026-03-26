"""
Project Tracker — Data Model and Store
=======================================
Central registry for photogrammetry projects. Tracks lifecycle stages,
source media paths across drives, and Metashape project references.

GUI-independent — usable from CLI, Metashape scripts, or the Projects tab.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


STAGE_ORDER = [
    "captured",
    "extracted",
    "masked",
    "aligned",
    "dense_cloud",
    "meshed",
    "textured",
    "exported",
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
    """A single source media directory or file associated with a project."""
    label: str
    path: str
    media_type: str  # "images", "video", "masks", "other"
    file_count: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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
class ProjectStage:
    """Status of one processing stage."""
    status: str = "not_started"  # not_started, in_progress, done, skipped
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectStage":
        return cls(
            status=data.get("status", "not_started"),
            updated_at=data.get("updated_at", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class Project:
    """A single photogrammetry project."""
    id: str
    title: str
    scene_type: str = ""
    created_at: str = ""
    updated_at: str = ""
    sources: List[ProjectSource] = field(default_factory=list)
    stages: Dict[str, ProjectStage] = field(default_factory=dict)
    metashape_path: str = ""
    export_paths: List[str] = field(default_factory=list)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "scene_type": self.scene_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": [s.to_dict() for s in self.sources],
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "metashape_path": self.metashape_path,
            "export_paths": self.export_paths,
            "notes": self.notes,
            "tags": self.tags,
            "archived": self.archived,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "Untitled"),
            scene_type=data.get("scene_type", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            sources=[ProjectSource.from_dict(s) for s in data.get("sources", [])],
            stages={k: ProjectStage.from_dict(v) for k, v in data.get("stages", {}).items()},
            metashape_path=data.get("metashape_path", ""),
            export_paths=data.get("export_paths", []),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            archived=data.get("archived", False),
        )

    @classmethod
    def create(cls, title: str, scene_type: str = "") -> "Project":
        """Create a new project with default stages."""
        now = datetime.now().isoformat()
        stages = {name: ProjectStage() for name in STAGE_ORDER}
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            scene_type=scene_type,
            created_at=now,
            updated_at=now,
            stages=stages,
        )

    def set_stage(self, stage: str, status: str, notes: str = ""):
        """Update a stage's status and timestamp."""
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}. Must be one of {STAGE_ORDER}")
        if stage not in self.stages:
            self.stages[stage] = ProjectStage()
        self.stages[stage].status = status
        self.stages[stage].updated_at = datetime.now().isoformat()
        if notes:
            self.stages[stage].notes = notes
        self.updated_at = datetime.now().isoformat()

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

    def current_stage(self) -> str:
        """Return the name of the most advanced stage that is done or in_progress."""
        latest = "captured"
        for name in STAGE_ORDER:
            st = self.stages.get(name)
            if st and st.status in ("done", "in_progress"):
                latest = name
        return latest


class ProjectStore:
    """Persistent collection of projects. Saves to a JSON file.

    Usage::

        store = ProjectStore("D:/tracker.json")
        proj = store.create_project("My Scan", scene_type="interior")
        proj.add_source("ERP frames", "D:/scans/360/frames", "images")
        proj.set_stage("captured", "done")
        store.save()

        # Later...
        store = ProjectStore("D:/tracker.json")
        for p in store.list_projects():
            print(p.title, p.current_stage())
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

    def create_project(self, title: str, scene_type: str = "") -> Project:
        """Create and register a new project."""
        proj = Project.create(title, scene_type)
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

    def archive_project(self, project_id: str):
        """Soft-delete: mark as archived (hidden from default list)."""
        proj = self._projects.get(project_id)
        if proj:
            proj.archived = True
            proj.updated_at = datetime.now().isoformat()
            self.save()

    def list_projects(
        self,
        include_archived: bool = False,
        stage_filter: str = "",
        tag_filter: str = "",
        search: str = "",
    ) -> List[Project]:
        """List projects with optional filters.

        Args:
            include_archived: Include archived projects
            stage_filter: Only show projects whose current_stage matches
            tag_filter: Only show projects that have this tag
            search: Case-insensitive substring match on title or notes
        """
        results = []
        for p in self._projects.values():
            if not include_archived and p.archived:
                continue
            if stage_filter and p.current_stage() != stage_filter:
                continue
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

    def validate_paths(self, project_id: str) -> Dict[str, bool]:
        """Check which paths in a project still exist on disk.

        Returns dict mapping path -> exists (bool).
        """
        proj = self._projects.get(project_id)
        if not proj:
            return {}
        results = {}
        for src in proj.sources:
            results[src.path] = Path(src.path).exists()
        if proj.metashape_path:
            results[proj.metashape_path] = Path(proj.metashape_path).exists()
        for ep in proj.export_paths:
            results[ep] = Path(ep).exists()
        return results
