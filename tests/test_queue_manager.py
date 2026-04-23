"""Tests for prep360 VideoQueue persistence and item lifecycle."""

import json

from prep360.core.queue_manager import (
    VideoQueue, QueueItem, ExtractionSettings, QueueItemStatus,
)


class TestExtractionSettings:
    def test_roundtrip_dict(self):
        s = ExtractionSettings(mode="sharpest", interval=1.5, quality=90)
        d = s.to_dict()
        s2 = ExtractionSettings.from_dict(d)
        assert s2.mode == "sharpest"
        assert s2.interval == 1.5
        assert s2.quality == 90

    def test_from_dict_defaults(self):
        s = ExtractionSettings.from_dict({})
        assert s.mode == "fixed"
        assert s.interval == 2.0
        assert s.blur_filter is False

    def test_summary(self):
        s = ExtractionSettings(mode="fixed", interval=2.0, format="jpg", quality=95)
        summary = s.summary()
        assert "fixed" in summary
        assert "2.0s" in summary


class TestQueueItem:
    def test_create(self, tmp_dir):
        video = tmp_dir / "test.mp4"
        video.touch()
        item = QueueItem.create(str(video))
        assert item.filename == "test.mp4"
        assert item.status == "pending"
        assert item.id  # non-empty UUID

    def test_roundtrip_dict(self, tmp_dir):
        video = tmp_dir / "test.mp4"
        video.touch()
        item = QueueItem.create(str(video), settings=ExtractionSettings(mode="scene"))
        d = item.to_dict()
        item2 = QueueItem.from_dict(d)
        assert item2.filename == item.filename
        assert item2.settings.mode == "scene"

    def test_from_dict_no_settings(self):
        item = QueueItem.from_dict({"id": "abc", "video_path": "/x.mp4", "filename": "x.mp4"})
        assert item.settings is None


class TestVideoQueue:
    def test_add_and_list(self, tmp_dir):
        q = VideoQueue(save_path=str(tmp_dir / "queue.json"))
        video = tmp_dir / "v1.mp4"
        video.touch()
        item = QueueItem.create(str(video))
        q.items.append(item)
        q.save()
        assert len(q.items) == 1

    def test_persistence(self, tmp_dir):
        save_path = str(tmp_dir / "queue.json")
        q1 = VideoQueue(save_path=save_path)
        video = tmp_dir / "v1.mp4"
        video.touch()
        q1.items.append(QueueItem.create(str(video)))
        q1.save()

        q2 = VideoQueue(save_path=save_path)
        assert len(q2.items) == 1
        assert q2.items[0].filename == "v1.mp4"

    def test_processing_items_reset_on_load(self, tmp_dir):
        save_path = str(tmp_dir / "queue.json")
        # Write a queue file with a "processing" item
        data = {"items": [{"id": "1", "video_path": "/x.mp4", "filename": "x.mp4", "status": "processing"}]}
        with open(save_path, "w") as f:
            json.dump(data, f)

        q = VideoQueue(save_path=save_path)
        assert q.items[0].status == "pending"

    def test_empty_queue(self, tmp_dir):
        q = VideoQueue(save_path=str(tmp_dir / "nonexistent.json"))
        assert len(q.items) == 0
