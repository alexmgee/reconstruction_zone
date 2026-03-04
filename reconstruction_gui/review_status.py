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
    """Status of a single mask in the review workflow."""
    status: str = "pending"  # pending, accepted, edited, rejected, skipped
    quality: str = ""        # excellent, good, review, poor, reject
    confidence: float = 0.0
    area_percent: float = 0.0
    edited_at: Optional[str] = None
    action_history: List[str] = field(default_factory=list)

    def record_action(self, action: str):
        """Record an action with timestamp."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.action_history.append(f"{ts}: {action}")
        if action in ("accepted", "edited", "rejected", "skipped"):
            self.status = action
        if action == "edited":
            self.edited_at = ts


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
        """Load status from disk."""
        if self.status_file.exists():
            try:
                raw = json.loads(self.status_file.read_text())
                for stem, d in raw.items():
                    self._data[stem] = MaskStatus(**d)
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

    def record_action(self, stem: str, action: str):
        """Record an action and auto-save."""
        ms = self.get(stem)
        ms.record_action(action)
        self.save()

    def set_quality_info(self, stem: str, quality: str, confidence: float, area_percent: float):
        """Set quality metadata from the pipeline."""
        ms = self.get(stem)
        ms.quality = quality
        ms.confidence = confidence
        ms.area_percent = area_percent
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
