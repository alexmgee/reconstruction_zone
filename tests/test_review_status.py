"""Tests for ReviewStatusManager — mask review workflow persistence.

Binary status model: masks are either ``unreviewed`` or ``reviewed``.
The ``last_edit_modified`` flag distinguishes "actually edited" from
"rubber-stamped" without adding more statuses.
"""

import json

from reconstruction_gui.review_status import ReviewStatusManager, MaskStatus


class TestMaskStatus:
    def test_default_state(self):
        ms = MaskStatus()
        assert ms.status == "unreviewed"
        assert ms.quality == ""
        assert ms.confidence == 0.0
        assert ms.action_history == []
        assert ms.last_edit_modified is None

    def test_record_reviewed_updates_status(self):
        ms = MaskStatus()
        ms.record_action("reviewed")
        assert ms.status == "reviewed"
        assert len(ms.action_history) == 1
        assert "reviewed" in ms.action_history[0]

    def test_record_reviewed_with_modified_flag(self):
        ms = MaskStatus()
        ms.record_action("reviewed", modified=True)
        assert ms.status == "reviewed"
        assert ms.edited_at is not None
        assert ms.last_edit_modified is True

    def test_record_reviewed_without_modification(self):
        ms = MaskStatus()
        ms.record_action("reviewed", modified=False)
        assert ms.status == "reviewed"
        assert ms.last_edit_modified is False

    def test_action_history_accumulates(self):
        ms = MaskStatus()
        ms.record_action("reviewed")
        ms.record_action("reviewed", modified=True)
        ms.record_action("reviewed", modified=False)
        assert len(ms.action_history) == 3
        assert ms.status == "reviewed"

    def test_unknown_action_logged_but_status_unchanged(self):
        ms = MaskStatus()
        ms.record_action("zoomed_in")
        # "zoomed_in" is not "reviewed", so status stays unreviewed
        assert ms.status == "unreviewed"
        assert len(ms.action_history) == 1


class TestReviewStatusManager:
    def test_get_creates_new(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        ms = mgr.get("frame_0001")
        assert ms.status == "unreviewed"

    def test_update_and_retrieve(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.update("frame_0001", quality="excellent", confidence=0.95)
        ms = mgr.get("frame_0001")
        assert ms.quality == "excellent"
        assert ms.confidence == 0.95

    def test_record_action_and_save(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("frame_0001", "reviewed")
        assert mgr.get("frame_0001").status == "reviewed"
        # Check it persisted to disk
        assert (tmp_dir / "review_status.json").exists()

    def test_persistence_across_instances(self, tmp_dir):
        mgr1 = ReviewStatusManager(tmp_dir)
        mgr1.record_action("frame_0001", "reviewed")
        mgr1.set_quality_info("frame_0001", "excellent", 0.95, 12.5)

        mgr2 = ReviewStatusManager(tmp_dir)
        ms = mgr2.get("frame_0001")
        assert ms.status == "reviewed"
        assert ms.quality == "excellent"
        assert ms.confidence == 0.95
        assert ms.area_percent == 12.5

    def test_get_summary(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("a", "reviewed")
        mgr.record_action("b", "reviewed")
        mgr.get("c")  # unreviewed (just created)
        mgr.get("d")  # unreviewed (just created)

        summary = mgr.get_summary()
        assert summary["reviewed"] == 2
        assert summary["unreviewed"] == 2

    def test_get_filtered(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.record_action("a", "reviewed")
        mgr.record_action("b", "reviewed")
        mgr.get("c")  # unreviewed

        reviewed = mgr.get_filtered(lambda ms: ms.status == "reviewed")
        assert set(reviewed) == {"a", "b"}

    def test_record_deleted_removes_entry(self, tmp_dir):
        mgr = ReviewStatusManager(tmp_dir)
        mgr.get("frame_0001")
        mgr.record_action("frame_0001", "deleted")
        assert "frame_0001" not in mgr.stems()
        # Verify persisted
        mgr2 = ReviewStatusManager(tmp_dir)
        assert "frame_0001" not in mgr2.stems()

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
