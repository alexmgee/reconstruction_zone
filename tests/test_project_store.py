"""Tests for ProjectStore — project CRUD, persistence, and migration."""

import json

from reconstruction_gui.project_store import (
    Project, ProjectSource, ProjectWorkDir, ProjectStore,
    STAGE_ORDER, count_media_files,
)


class TestProjectSource:
    def test_roundtrip_dict(self):
        src = ProjectSource(label="ERP", path="/data/frames", media_type="images", file_count=42)
        d = src.to_dict()
        src2 = ProjectSource.from_dict(d)
        assert src2.label == "ERP"
        assert src2.path == "/data/frames"
        assert src2.file_count == 42

    def test_from_dict_defaults(self):
        src = ProjectSource.from_dict({})
        assert src.label == ""
        assert src.media_type == "other"


class TestProjectWorkDir:
    def test_roundtrip_dict(self):
        wd = ProjectWorkDir(label="masked", path="/data/masks", stage="masked", file_count=100)
        d = wd.to_dict()
        wd2 = ProjectWorkDir.from_dict(d)
        assert wd2.stage == "masked"
        assert wd2.file_count == 100


class TestProject:
    def test_create(self):
        p = Project.create("My Scan", root_dir="/data/scans/001")
        assert p.title == "My Scan"
        assert p.root_dir == "/data/scans/001"
        assert p.id  # non-empty UUID
        assert p.created_at  # non-empty timestamp

    def test_roundtrip_dict(self):
        p = Project.create("Test")
        p.sources.append(ProjectSource(label="imgs", path="/imgs", media_type="images"))
        p.work_dirs.append(ProjectWorkDir(label="masks", path="/masks", stage="masked"))
        p.tags = ["outdoor", "drone"]
        p.notes = "Test scan"

        d = p.to_dict()
        p2 = Project.from_dict(d)
        assert p2.title == "Test"
        assert len(p2.sources) == 1
        assert len(p2.work_dirs) == 1
        assert p2.tags == ["outdoor", "drone"]
        assert p2.notes == "Test scan"

    def test_backward_compat_stage_migration(self):
        """Old sources that had a 'stage' field should migrate to work_dirs."""
        data = {
            "id": "test-id",
            "title": "Old Project",
            "sources": [
                {"label": "frames", "path": "/frames", "media_type": "images"},
                {"label": "masks", "path": "/masks", "media_type": "images", "stage": "masked"},
            ],
        }
        p = Project.from_dict(data)
        # The source without stage stays as a source
        assert len(p.sources) == 1
        assert p.sources[0].label == "frames"
        # The source with stage migrated to work_dirs
        assert len(p.work_dirs) == 1
        assert p.work_dirs[0].label == "masks"
        assert p.work_dirs[0].stage == "masked"

    def test_from_dict_defaults(self):
        p = Project.from_dict({})
        assert p.title == "Untitled"
        assert p.sources == []
        assert p.work_dirs == []

    def test_add_source_with_real_dir(self, tmp_dir):
        """add_source should auto-count files."""
        # Create some media files
        (tmp_dir / "frame_001.jpg").touch()
        (tmp_dir / "frame_002.jpg").touch()
        (tmp_dir / "frame_003.png").touch()
        (tmp_dir / "readme.txt").touch()  # not media

        p = Project.create("Test")
        src = p.add_source("frames", str(tmp_dir), "images")
        assert src.file_count == 3  # only .jpg and .png

    def test_add_work_dir(self, tmp_dir):
        (tmp_dir / "mask_001.png").touch()
        p = Project.create("Test")
        wd = p.add_work_dir("masks", str(tmp_dir), derived_from="frames", stage="masked")
        assert wd.stage == "masked"
        assert wd.file_count == 1


class TestProjectStore:
    def test_create_and_list(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        store.create_project("Project A")
        store.create_project("Project B")
        projects = store.list_projects()
        assert len(projects) == 2

    def test_persistence(self, tmp_dir):
        path = str(tmp_dir / "tracker.json")
        store1 = ProjectStore(path)
        store1.create_project("Persistent")

        store2 = ProjectStore(path)
        projects = store2.list_projects()
        assert len(projects) == 1
        assert projects[0].title == "Persistent"

    def test_delete_project(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p = store.create_project("Delete Me")
        store.delete_project(p.id)
        assert store.list_projects() == []

    def test_get_project(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p = store.create_project("Get Me")
        found = store.get_project(p.id)
        assert found is not None
        assert found.title == "Get Me"

    def test_get_nonexistent_returns_none(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        assert store.get_project("fake-id") is None

    def test_search_filter(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        store.create_project("Garden Scan")
        store.create_project("Interior Shoot")

        results = store.list_projects(search="garden")
        assert len(results) == 1
        assert results[0].title == "Garden Scan"

    def test_tag_filter(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p1 = store.create_project("A")
        p1.tags = ["outdoor"]
        store.save()

        p2 = store.create_project("B")
        p2.tags = ["indoor"]
        store.save()

        results = store.list_projects(tag_filter="outdoor")
        assert len(results) == 1
        assert results[0].title == "A"

    def test_sorted_by_updated_at(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p1 = store.create_project("First")
        p2 = store.create_project("Second")
        # p2 was created second → more recent → should be first in list
        projects = store.list_projects()
        assert projects[0].title == "Second"

    def test_validate_paths(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p = store.create_project("Paths")
        real_dir = tmp_dir / "real"
        real_dir.mkdir()
        p.sources.append(ProjectSource(label="real", path=str(real_dir), media_type="images"))
        p.sources.append(ProjectSource(label="fake", path="/nonexistent/path", media_type="images"))
        store.save()

        validation = store.validate_paths(p.id)
        assert validation[str(real_dir)] is True
        assert validation["/nonexistent/path"] is False

    def test_relocate_source(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "tracker.json"))
        p = store.create_project("Relocate")
        p.sources.append(ProjectSource(label="old", path="/old/path", media_type="images"))
        store.save()

        new_dir = tmp_dir / "new_location"
        new_dir.mkdir()
        (new_dir / "frame.jpg").touch()

        result = store.relocate_source(p.id, 0, str(new_dir))
        assert result is True
        assert p.sources[0].path == str(new_dir)
        assert p.sources[0].file_count == 1

    def test_empty_store(self, tmp_dir):
        store = ProjectStore(str(tmp_dir / "nonexistent.json"))
        assert store.list_projects() == []


class TestCountMediaFiles:
    def test_counts_media_only(self, tmp_dir):
        (tmp_dir / "a.jpg").touch()
        (tmp_dir / "b.png").touch()
        (tmp_dir / "c.mp4").touch()
        (tmp_dir / "d.txt").touch()
        (tmp_dir / "e.json").touch()
        assert count_media_files(str(tmp_dir)) == 3

    def test_nonexistent_dir(self):
        assert count_media_files("/this/does/not/exist") == 0


class TestStageOrder:
    def test_stages_defined(self):
        assert "extracted" in STAGE_ORDER
        assert "masked" in STAGE_ORDER
        assert "aligned" in STAGE_ORDER
        assert "trained" in STAGE_ORDER
        assert STAGE_ORDER.index("extracted") < STAGE_ORDER.index("masked")
