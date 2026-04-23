"""Tests for ReviewStatusManager — mask review workflow persistence."""

import json

from reconstruction_gui.review_status import ReviewStatusManager, MaskStatus


class TestMaskStatus:
    def test_default_state(self):
        ms = MaskStatus()
        assert ms.status == "pending"
        assert ms.quality == ""
        assert ms.confidence == 0.0
        assert ms.action_history == []

    def test_record_action_updates_status(self):
        ms = MaskStatus()
        ms.record_action("accepted")
        assert ms.status == "accepted"
        assert len(ms.action_history) == 1
        assert "accepted" in ms.action_history[0]

    def test_record_edited_sets_timestamp(self):
        ms = MaskStatus()
        ms.record_action("edited")
        assert ms.status == "edited"
        assert ms.edited_at is not None

    def test_record_rejected(self):
        ms = MaskStatus()
        ms.record_action("rejected")
        assert ms.status == "rejected"

    def test_action_history_accumulates(self):
        ms = MaskStatus()
        ms.record_action("accepted")
        ms.record_action("edited")
        ms.record_action("accepted")
        assert len(ms.action_history) == 3
        assert ms.status == "accepted"  # last action wins

    def test_unknown_action_logged_but_status_unchanged(self):
        ms = MaskStatus()
        ms.record_action("zoomed_in")
        # "zoomed_in" is not in the status-update list, so status stays pending
        assert ms.status == "pending"
        assert len(ms.action_history) == 1


class TestReviewStatusManager:
    def test_get_creates_new(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        ms = mgr.get("frame_0001")
        assert ms.status == "pending"

    def test_update_and_retrieve(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.update("frame_0001", quality="excellent", confidence=0.95)
        ms = mgr.get("frame_0001")
        assert ms.quality == "excellent"
        assert ms.confidence == 0.95

    def test_record_action_and_save(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("frame_0001", "accepted")
        assert mgr.get("frame_0001").status == "accepted"
        # Check it persisted to disk
        assert (tmp_dir / "review_status.json").exists()

    def test_persistence_across_instances(self, tmp_dir):
        mgr1 = ReviewStatusManager(tmp_dir)
        mgr1.record_action("frame_0001", "accepted")
        mgr1.set_quality_info("frame_0001", "excellent", 0.95, 12.5)

        mgr2 = ReviewStatusManager(tmp_dir)
        ms = mgr2.get("frame_0001")
        assert ms.status == "accepted"
        assert ms.quality == "excellent"
        assert ms.confidence == 0.95
        assert ms.area_percent == 12.5

    def test_get_summary(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("a", "accepted")
        mgr.record_action("b", "accepted")
        mgr.record_action("c", "rejected")
        mgr.get("d")  # pending (just created)

        summary = mgr.get_summary()
        assert summary["accepted"] == 2
        assert summary["rejected"] == 1
        assert summary["pending"] == 1

    def test_get_filtered(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("a", "accepted")
        mgr.record_action("b", "rejected")
        mgr.record_action("c", "accepted")

        accepted = mgr.get_filtered(lambda ms: ms.status == "accepted")
        assert set(accepted) == {"a", "c"}

    def test_stems(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.get("x")
        mgr.get("y")
        mgr.get("z")
        assert set(mgr.stems()) == {"x", "y", "z"}

    def test_corrupt_file_handled(self, tmp_dir):
        """If the JSON file is corrupt, manager starts empty."""
        (tmp_dir / "review_status.json").write_text("NOT VALID JSON")
        mgr = ReviewStatusManager(tmp_dir)
        assert mgr.stems() == []
