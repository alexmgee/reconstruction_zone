"""
Activity Store — Lightweight operation tracking.
================================================
Records operations performed across all tabs. Groups by root directory
for the Recent Activity view in the Projects tab.

GUI-independent — usable from CLI or the Projects tab.
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


MAX_ACTIVITY_GROUPS = 30  # Cap on grouped root directories shown


@dataclass
class ActivityRecord:
    """A single completed operation."""
    id: str
    timestamp: str  # ISO format
    operation: str  # "extract", "mask", "align", "gap_analysis", "bridge", "reframe"
    input_path: str
    output_path: str
    status: str  # "completed", "failed", "cancelled"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "operation": self.operation,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "status": self.status,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActivityRecord":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", ""),
            operation=data.get("operation", ""),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            status=data.get("status", "completed"),
            details=data.get("details", {}),
        )

    @classmethod
    def create(cls, operation: str, input_path: str, output_path: str,
               status: str = "completed",
               details: Optional[Dict[str, Any]] = None) -> "ActivityRecord":
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            operation=operation,
            input_path=input_path,
            output_path=output_path,
            status=status,
            details=details or {},
        )


@dataclass
class ActivityGroup:
    """Activities sharing a common root directory, shown as one card."""
    root_dir: str
    display_name: str
    activities: List[ActivityRecord] = field(default_factory=list)
    latest_timestamp: str = ""

    @property
    def operation_summary(self) -> str:
        """E.g. 'Extract (342) → Mask (342) → Align'"""
        parts = []
        for act in sorted(self.activities, key=lambda a: a.timestamp):
            op = act.operation.capitalize()
            count = act.details.get("frame_count") or act.details.get("image_count") or ""
            if count:
                parts.append(f"{op} ({count})")
            else:
                parts.append(op)
        return " → ".join(parts)

    @property
    def run_count(self) -> int:
        return len(self.activities)

    @property
    def latest_status(self) -> str:
        if not self.activities:
            return ""
        latest = max(self.activities, key=lambda a: a.timestamp)
        return latest.status


def _find_common_root(paths: List[str]) -> str:
    """Find the deepest common ancestor directory of a list of paths."""
    if not paths:
        return ""
    resolved = []
    for p in paths:
        try:
            resolved.append(Path(p).resolve())
        except (OSError, ValueError):
            resolved.append(Path(p))
    if len(resolved) == 1:
        p = resolved[0]
        return str(p if p.is_dir() else p.parent)
    # Find common prefix
    parts_list = [p.parts for p in resolved]
    common = []
    for level_parts in zip(*parts_list):
        if len(set(str(p).lower() for p in level_parts)) == 1:
            common.append(level_parts[0])
        else:
            break
    if not common:
        return str(resolved[0].parent)
    return str(Path(*common))


def group_activities(records: List[ActivityRecord],
                     max_groups: int = MAX_ACTIVITY_GROUPS) -> List[ActivityGroup]:
    """Group activity records by common root directory."""
    # Phase 1: assign each record a root
    record_roots: Dict[str, List[ActivityRecord]] = {}
    for rec in records:
        paths = [p for p in [rec.input_path, rec.output_path] if p]
        root = _find_common_root(paths)
        root_key = str(Path(root)).lower()
        if root_key not in record_roots:
            record_roots[root_key] = []
        record_roots[root_key].append(rec)

    # Phase 2: merge roots that are parent/child of each other
    sorted_roots = sorted(record_roots.keys())
    merged: Dict[str, List[ActivityRecord]] = {}
    for root_key in sorted_roots:
        parent_found = None
        for existing_key in merged:
            try:
                if Path(root_key).is_relative_to(Path(existing_key)):
                    parent_found = existing_key
                    break
            except (ValueError, TypeError):
                continue
        if parent_found:
            merged[parent_found].extend(record_roots[root_key])
        else:
            merged[root_key] = list(record_roots[root_key])

    # Phase 3: build ActivityGroup objects
    groups = []
    for root_key, recs in merged.items():
        root_path = Path(root_key)
        groups.append(ActivityGroup(
            root_dir=str(root_path),
            display_name=root_path.name or str(root_path),
            activities=sorted(recs, key=lambda r: r.timestamp),
            latest_timestamp=max(r.timestamp for r in recs) if recs else "",
        ))

    groups.sort(key=lambda g: g.latest_timestamp, reverse=True)
    return groups[:max_groups]


class ActivityStore:
    """Persistent collection of activity records. Saves to a JSON file."""

    VERSION = 1
    MAX_RECORDS = 200

    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self._records: List[ActivityRecord] = []
        self._load()

    def _load(self):
        if self.store_path.exists():
            try:
                raw = json.loads(self.store_path.read_text(encoding="utf-8"))
                for r_data in raw.get("records", []):
                    self._records.append(ActivityRecord.from_dict(r_data))
            except Exception as e:
                print(f"Warning: failed to load activity log: {e}")

    def save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "records": [r.to_dict() for r in self._records],
        }
        self.store_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def record(self, operation: str, input_path: str, output_path: str,
               status: str = "completed",
               details: Optional[Dict[str, Any]] = None) -> ActivityRecord:
        """Create and persist a new activity record."""
        rec = ActivityRecord.create(
            operation=operation,
            input_path=input_path,
            output_path=output_path,
            status=status,
            details=details,
        )
        self._records.append(rec)
        self._prune()
        self.save()
        return rec

    def get_groups(self, max_groups: int = MAX_ACTIVITY_GROUPS) -> List[ActivityGroup]:
        """Return activity records grouped by root directory."""
        return group_activities(self._records, max_groups=max_groups)

    def get_records_for_root(self, root_dir: str) -> List[ActivityRecord]:
        """Get all records whose paths fall under a root directory."""
        root = Path(root_dir).resolve()
        results = []
        for rec in self._records:
            for p in [rec.input_path, rec.output_path]:
                if not p:
                    continue
                try:
                    if Path(p).resolve().is_relative_to(root):
                        results.append(rec)
                        break
                except (ValueError, TypeError, OSError):
                    continue
        return sorted(results, key=lambda r: r.timestamp)

    def _prune(self):
        """Remove oldest records if over the cap."""
        if len(self._records) > self.MAX_RECORDS:
            self._records.sort(key=lambda r: r.timestamp)
            self._records = self._records[-self.MAX_RECORDS:]

    def clear(self):
        """Remove all records."""
        self._records.clear()
        self.save()
