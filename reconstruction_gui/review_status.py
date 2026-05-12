"""
Review Status Persistence
=========================
Tracks review status for each mask across sessions.
Saves/loads from review_status.json alongside the masks directory.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, List, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaskStatus:
    """Status of a single mask in the review workflow.

    Binary status model: a mask is either reviewed or unreviewed. The
    ``last_edit_modified`` flag preserves the editor's modified-vs-unchanged
    distinction so users can sort by "actually edited" vs "rubber-stamped"
    without breaking the binary status.
    """
    status: str = "unreviewed"  # unreviewed, reviewed
    quality: str = ""           # excellent, good, review, poor, reject (informational only)
    confidence: float = 0.0
    area_percent: float = 0.0
    edited_at: Optional[str] = None
    action_history: List[str] = field(default_factory=list)
    last_edit_modified: Optional[bool] = None  # None = never edited, True = pixels changed, False = no-op save
    # Click-PVS forensic metadata (None for non-click layers).
    model_used: Optional[str] = None
    click_points: Optional[List[Dict]] = None
    decoder_score: Optional[float] = None

    def record_action(self, action: str, modified: Optional[bool] = None):
        """Record an action with timestamp.

        Args:
            action: ``"reviewed"`` or ``"deleted"``. (``"deleted"`` is also
                handled at the manager level via entry removal.)
            modified: Only relevant for ``action="reviewed"``. True if the
                editor save changed pixels, False if no-op, None if the
                action did not come from the editor.
        """
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detail = f" ({'modified' if modified else 'unchanged'})" if modified is not None else ""
        self.action_history.append(f"{ts}: {action}{detail}")
        if action == "reviewed":
            self.status = "reviewed"
            self.edited_at = ts
            if modified is not None:
                self.last_edit_modified = modified


class ReviewStatusManager:
    """Manages review status for a batch of masks.

    Status persists to ``review_status.json`` in the given directory so
    review sessions survive app restarts.
    """

    def __init__(self, status_dir: Path):
        self.status_dir = Path(status_dir)
        self.status_file = self.status_dir / "review_status.json"
        self._data: Dict[str, MaskStatus] = {}
        self._load()

    # -- persistence --

    def _load(self):
        """Load status from disk. Migrates legacy statuses to the binary model."""
        if self.status_file.exists():
            try:
                raw = json.loads(self.status_file.read_text())
                migrated = 0
                for stem, d in raw.items():
                    raw_status = d.get("status", "")
                    # Drop unknown legacy fields so MaskStatus(**d) doesn't fail
                    d = {k: v for k, v in d.items() if k in MaskStatus.__dataclass_fields__}
                    ms = MaskStatus(**d)
                    # Migrate legacy statuses → binary model
                    if raw_status in ("accepted", "edited"):
                        ms.status = "reviewed"
                        # Heuristic: "edited" means user changed pixels, "accepted" means rubber-stamped
                        if ms.last_edit_modified is None:
                            ms.last_edit_modified = (raw_status == "edited")
                        migrated += 1
                    elif raw_status in ("pending", "skipped", "rejected"):
                        # rejected masks may still exist on disk — surface them as unreviewed
                        ms.status = "unreviewed"
                        if raw_status != "pending":
                            migrated += 1
                    self._data[stem] = ms
                if migrated:
                    logger.info(f"Loaded {len(self._data)} masks ({migrated} migrated from legacy statuses)")
                    self.save()  # persist migration
                else:
                    logger.info(f"Loaded review status for {len(self._data)} masks")
            except Exception as e:
                logger.warning(f"Failed to load review status: {e}")

    def save(self):
        """Write current status to disk."""
        self.status_dir.mkdir(parents=True, exist_ok=True)
        raw = {stem: asdict(ms) for stem, ms in self._data.items()}
        self.status_file.write_text(json.dumps(raw, indent=2))

    # -- accessors --

    def get(self, stem: str) -> MaskStatus:
        """Get status for a mask stem, creating if absent."""
        if stem not in self._data:
            self._data[stem] = MaskStatus()
        return self._data[stem]

    def update(self, stem: str, **kwargs):
        """Update fields on a mask status and auto-save."""
        ms = self.get(stem)
        for k, v in kwargs.items():
            if hasattr(ms, k):
                setattr(ms, k, v)
        self.save()

    def record_action(self, stem: str, action: str, modified: Optional[bool] = None):
        """Record an action and auto-save.

        For ``action="deleted"``: removes the entry from this manager and
        persists. Caller is responsible for unlinking the mask file on disk.
        Other actions (``"reviewed"``) are forwarded to ``MaskStatus``.
        """
        if action == "deleted":
            self._data.pop(stem, None)
        else:
            ms = self.get(stem)
            ms.record_action(action, modified=modified)
        self.save()

    def set_quality_info(self, stem: str, quality: str, confidence: float, area_percent: float):
        """Set quality metadata from the pipeline."""
        ms = self.get(stem)
        ms.quality = quality
        ms.confidence = confidence
        ms.area_percent = area_percent
        self.save()

    def set_click_metadata(
        self,
        stem: str,
        model: str,
        points: List[Dict],
        score: float,
    ):
        """Record click-PVS metadata for a saved layer mask."""
        ms = self.get(stem)
        ms.model_used = model
        ms.click_points = list(points)
        ms.decoder_score = float(score)
        self.save()

    # -- queries --

    def get_summary(self) -> Dict[str, int]:
        """Return counts per status."""
        summary: Dict[str, int] = {}
        for ms in self._data.values():
            summary[ms.status] = summary.get(ms.status, 0) + 1
        return summary

    def get_filtered(self, filter_fn: Callable[[MaskStatus], bool]) -> List[str]:
        """Return stems matching a filter function."""
        return [stem for stem, ms in self._data.items() if filter_fn(ms)]

    def stems(self) -> List[str]:
        """All known stems."""
        return list(self._data.keys())
