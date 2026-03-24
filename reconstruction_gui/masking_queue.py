"""
Masking Queue Module

Manages a persistent queue of image folders for batch masking.
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class MaskingQueueItem:
    """A single folder in the masking queue."""
    id: str
    folder_path: str
    folder_name: str
    status: str = "pending"  # pending, processing, done, error, cancelled
    progress: int = 0  # 0-100
    image_count: int = 0
    processed_count: int = 0
    error_message: str = ""
    added_time: str = ""
    completed_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "folder_path": self.folder_path,
            "folder_name": self.folder_name,
            "status": self.status,
            "progress": self.progress,
            "image_count": self.image_count,
            "processed_count": self.processed_count,
            "error_message": self.error_message,
            "added_time": self.added_time,
            "completed_time": self.completed_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskingQueueItem":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            folder_path=data.get("folder_path", ""),
            folder_name=data.get("folder_name", ""),
            status=data.get("status", "pending"),
            progress=data.get("progress", 0),
            image_count=data.get("image_count", 0),
            processed_count=data.get("processed_count", 0),
            error_message=data.get("error_message", ""),
            added_time=data.get("added_time", ""),
            completed_time=data.get("completed_time", ""),
        )

    @classmethod
    def create(cls, folder_path: str) -> "MaskingQueueItem":
        """Create a new queue item from a folder path."""
        path = Path(folder_path)
        return cls(
            id=str(uuid.uuid4()),
            folder_path=str(path.absolute()),
            folder_name=path.name,
            status="pending",
            progress=0,
            added_time=datetime.now().isoformat(),
        )


class MaskingQueue:
    """Manages a queue of image folders for batch masking."""

    def __init__(self, save_path: Optional[str] = None):
        self.items: List[MaskingQueueItem] = []
        self.save_path = save_path or str(Path.home() / ".prep360_masking_queue.json")
        self._load()

    def _load(self):
        """Load queue state from disk."""
        try:
            path = Path(self.save_path)
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)

                self.items = [MaskingQueueItem.from_dict(item)
                              for item in data.get("items", [])]

                # Reset any "processing" items to "pending" (app crash recovery)
                for item in self.items:
                    if item.status == "processing":
                        item.status = "pending"
                        item.progress = 0
        except Exception:
            self.items = []

    def save(self):
        """Save queue state to disk."""
        try:
            data = {"items": [item.to_dict() for item in self.items]}
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_folder(self, folder_path: str) -> Optional[MaskingQueueItem]:
        """Add an image folder to the queue. Returns None if duplicate or invalid."""
        path = Path(folder_path)
        if not path.is_dir():
            return None

        # Check for duplicates
        abs_path = str(path.absolute())
        for item in self.items:
            if item.folder_path == abs_path:
                return None

        item = MaskingQueueItem.create(folder_path)
        self.items.append(item)
        self.save()
        return item

    def add_folders(self, folder_paths: List[str]) -> int:
        """Add multiple folders to the queue. Returns count added."""
        count = 0
        for path in folder_paths:
            if self.add_folder(path):
                count += 1
        return count

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the queue."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                if item.status == "processing":
                    return False
                self.items.pop(i)
                self.save()
                return True
        return False

    def clear_completed(self):
        """Remove all completed, error, and cancelled items."""
        self.items = [item for item in self.items
                      if item.status not in ("done", "error", "cancelled")]
        self.save()

    def clear_all(self):
        """Clear the entire queue (except processing items)."""
        self.items = [item for item in self.items if item.status == "processing"]
        self.save()

    def get_next_pending(self) -> Optional[MaskingQueueItem]:
        """Get the next pending item to process."""
        for item in self.items:
            if item.status == "pending":
                return item
        return None

    def get_item(self, item_id: str) -> Optional[MaskingQueueItem]:
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

    def set_done(self, item_id: str, processed_count: int = 0):
        """Mark an item as done."""
        self.update_item(
            item_id,
            status="done",
            progress=100,
            processed_count=processed_count,
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
        """Update an item's progress (0-100)."""
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
