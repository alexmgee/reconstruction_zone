"""
Queue Manager Module

Manages a persistent queue of video files for batch processing.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class QueueItemStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ExtractionSettings:
    """Per-item extraction settings captured at queue time."""
    mode: str = "fixed"          # fixed, scene, adaptive, sharpest
    interval: float = 2.0
    quality: int = 95
    format: str = "jpg"
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    blur_filter: bool = False
    blur_percentile: int = 80
    sky_filter: bool = False
    lut_enabled: bool = False
    lut_path: str = ""
    lut_strength: float = 1.0
    shadow: int = 50
    highlight: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionSettings":
        return cls(
            mode=data.get("mode", "fixed"),
            interval=data.get("interval", 2.0),
            quality=data.get("quality", 95),
            format=data.get("format", "jpg"),
            start_sec=data.get("start_sec"),
            end_sec=data.get("end_sec"),
            blur_filter=data.get("blur_filter", False),
            blur_percentile=data.get("blur_percentile", 80),
            sky_filter=data.get("sky_filter", False),
            lut_enabled=data.get("lut_enabled", False),
            lut_path=data.get("lut_path", ""),
            lut_strength=data.get("lut_strength", 1.0),
            shadow=data.get("shadow", 50),
            highlight=data.get("highlight", 50),
        )

    def summary(self) -> str:
        """One-line summary for console logging."""
        parts = [self.mode, f"{self.interval:.1f}s", self.format, f"q{self.quality}"]
        if self.start_sec is not None or self.end_sec is not None:
            s = self.start_sec or 0
            e = f"{self.end_sec}" if self.end_sec else "end"
            parts.append(f"{s}-{e}")
        if self.mode == "sharpest":
            parts.append(f"keep {self.blur_percentile}%")
        elif self.blur_filter:
            parts.append(f"blur≤{self.blur_percentile}%")
        if self.sky_filter:
            parts.append("sky")
        if self.lut_enabled:
            parts.append("LUT")
        if self.shadow != 50 or self.highlight != 50:
            parts.append(f"sh{self.shadow}/hl{self.highlight}")
        return ", ".join(parts)


@dataclass
class QueueItem:
    """A single item in the processing queue."""
    id: str
    video_path: str
    filename: str
    status: str = "pending"  # pending, processing, done, error, cancelled
    progress: int = 0  # 0-100
    frame_count: int = 0
    error_message: str = ""
    added_time: str = ""
    completed_time: str = ""
    settings: Optional[ExtractionSettings] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "video_path": self.video_path,
            "filename": self.filename,
            "status": self.status,
            "progress": self.progress,
            "frame_count": self.frame_count,
            "error_message": self.error_message,
            "added_time": self.added_time,
            "completed_time": self.completed_time,
        }
        if self.settings:
            d["settings"] = self.settings.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueItem":
        settings = None
        if "settings" in data and isinstance(data["settings"], dict):
            settings = ExtractionSettings.from_dict(data["settings"])
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            video_path=data.get("video_path", ""),
            filename=data.get("filename", ""),
            status=data.get("status", "pending"),
            progress=data.get("progress", 0),
            frame_count=data.get("frame_count", 0),
            error_message=data.get("error_message", ""),
            added_time=data.get("added_time", ""),
            completed_time=data.get("completed_time", ""),
            settings=settings,
        )

    @classmethod
    def create(cls, video_path: str, settings: Optional[ExtractionSettings] = None) -> "QueueItem":
        """Create a new queue item from a video path."""
        path = Path(video_path)
        return cls(
            id=str(uuid.uuid4()),
            video_path=str(path.absolute()),
            filename=path.name,
            status="pending",
            progress=0,
            added_time=datetime.now().isoformat(),
            settings=settings,
        )


class VideoQueue:
    """Manages a queue of videos for batch processing."""

    def __init__(self, save_path: Optional[str] = None):
        self.items: List[QueueItem] = []
        self.save_path = save_path or str(Path.home() / ".prep360_queue.json")
        self._load()

    def _load(self):
        """Load queue state from disk."""
        try:
            path = Path(self.save_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                self.items = [QueueItem.from_dict(item) for item in data.get("items", [])]

                # Reset any "processing" items to "pending" (app was closed during processing)
                for item in self.items:
                    if item.status == "processing":
                        item.status = "pending"
                        item.progress = 0
        except Exception:
            self.items = []

    def save(self):
        """Save queue state to disk."""
        try:
            data = {
                "items": [item.to_dict() for item in self.items],
            }
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_video(self, video_path: str, settings: Optional[ExtractionSettings] = None) -> Optional[QueueItem]:
        """Add a video to the queue with optional per-item settings."""
        path = Path(video_path)
        if not path.exists():
            return None

        # Check for duplicates
        for item in self.items:
            if item.video_path == str(path.absolute()):
                return None

        item = QueueItem.create(video_path, settings=settings)
        self.items.append(item)
        self.save()
        return item

    def add_videos(self, video_paths: List[str], settings: Optional[ExtractionSettings] = None) -> int:
        """Add multiple videos to the queue. Returns count added."""
        count = 0
        for path in video_paths:
            if self.add_video(path, settings=settings):
                count += 1
        return count

    def add_folder(self, folder_path: str, extensions: List[str] = None, settings: Optional[ExtractionSettings] = None) -> int:
        """Add all videos from a folder. Returns count added."""
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.360', '.insv']

        folder = Path(folder_path)
        if not folder.is_dir():
            return 0

        videos = []
        for ext in extensions:
            videos.extend(folder.glob(f"*{ext}"))
            videos.extend(folder.glob(f"*{ext.upper()}"))

        return self.add_videos([str(v) for v in sorted(videos)], settings=settings)

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the queue."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                # Don't remove if currently processing
                if item.status == "processing":
                    return False
                self.items.pop(i)
                self.save()
                return True
        return False

    def clear_completed(self):
        """Remove all completed and error items."""
        self.items = [item for item in self.items
                      if item.status not in ("done", "error", "cancelled")]
        self.save()

    def clear_all(self):
        """Clear the entire queue (except processing items)."""
        self.items = [item for item in self.items if item.status == "processing"]
        self.save()

    def move_up(self, item_id: str) -> bool:
        """Move an item up in the queue."""
        for i, item in enumerate(self.items):
            if item.id == item_id and i > 0:
                # Don't reorder processing items
                if item.status == "processing" or self.items[i-1].status == "processing":
                    return False
                self.items[i], self.items[i-1] = self.items[i-1], self.items[i]
                self.save()
                return True
        return False

    def move_down(self, item_id: str) -> bool:
        """Move an item down in the queue."""
        for i, item in enumerate(self.items):
            if item.id == item_id and i < len(self.items) - 1:
                # Don't reorder processing items
                if item.status == "processing" or self.items[i+1].status == "processing":
                    return False
                self.items[i], self.items[i+1] = self.items[i+1], self.items[i]
                self.save()
                return True
        return False

    def get_next_pending(self) -> Optional[QueueItem]:
        """Get the next pending item to process."""
        for item in self.items:
            if item.status == "pending":
                return item
        return None

    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def update_item(self, item_id: str, **kwargs):
        """Update an item's properties."""
        item = self.get_item(item_id)
        if item:
            for key, value in kwargs.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            self.save()

    def set_processing(self, item_id: str):
        """Mark an item as processing."""
        self.update_item(item_id, status="processing", progress=0)

    def set_done(self, item_id: str, frame_count: int = 0):
        """Mark an item as done."""
        self.update_item(
            item_id,
            status="done",
            progress=100,
            frame_count=frame_count,
            completed_time=datetime.now().isoformat()
        )

    def set_error(self, item_id: str, error_message: str):
        """Mark an item as error."""
        self.update_item(
            item_id,
            status="error",
            error_message=error_message,
            completed_time=datetime.now().isoformat()
        )

    def set_cancelled(self, item_id: str):
        """Mark an item as cancelled."""
        self.update_item(item_id, status="cancelled")

    def set_progress(self, item_id: str, progress: int):
        """Update an item's progress."""
        self.update_item(item_id, progress=min(100, max(0, progress)))

    def get_pending_count(self) -> int:
        """Get count of pending items."""
        return sum(1 for item in self.items if item.status == "pending")

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {"pending": 0, "processing": 0, "done": 0, "error": 0, "cancelled": 0}
        for item in self.items:
            if item.status in stats:
                stats[item.status] += 1
        stats["total"] = len(self.items)
        return stats

    def reset_errors(self):
        """Reset all error items back to pending."""
        for item in self.items:
            if item.status == "error":
                item.status = "pending"
                item.progress = 0
                item.error_message = ""
        self.save()
