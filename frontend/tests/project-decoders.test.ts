import { describe, expect, it } from "vitest";
import {
  normalizeProjectDetail,
  normalizeProjectList,
  type DecodedProjectDetail,
  type DecodedProjectSummary,
} from "../src/api/decoders";

const summary: DecodedProjectSummary = {
  id: "project-one",
  title: "Project One",
  tags: ["active", "local"],
  source_count: 2,
  work_dir_count: 1,
  created_at: "2026-07-24T10:00:00",
  updated_at: "2026-07-24T11:00:00",
};

const detail: DecodedProjectDetail = {
  id: summary.id,
  title: summary.title,
  created_at: "",
  updated_at: summary.updated_at,
  sources: [
    {
      label: "Source",
      path: "sources/input",
      media_type: "images",
      file_count: 2,
      notes: "Primary source",
      exists: true,
    },
  ],
  work_dirs: [
    {
      label: "Working",
      path: "work/aligned",
      stage: "aligned",
      file_count: 1,
      derived_from: "Source",
      exists: false,
    },
  ],
  notes: "Project notes",
  tags: summary.tags,
  root_dir: "projects/project-one",
  static_masks_dir: "projects/project-one/masks",
};

describe("project response decoders", () => {
  it("normalizes a valid project list", () => {
    expect(normalizeProjectList({ projects: [summary] })).toEqual([summary]);
  });

  it.each([
    { projects: summary },
    { projects: [{ ...summary, tags: "active" }] },
    { projects: [{ ...summary, source_count: -1 }] },
    { projects: [{ ...summary, work_dir_count: 1.5 }] },
    { projects: [{ ...summary, created_at: 123 }] },
  ])("returns null for an invalid project list payload", (payload) => {
    expect(normalizeProjectList(payload)).toBeNull();
  });

  it("normalizes valid project detail with an empty created_at timestamp", () => {
    expect(normalizeProjectDetail(detail)).toEqual(detail);
  });

  it("returns null when a nested source exists value is not boolean", () => {
    expect(
      normalizeProjectDetail({
        ...detail,
        sources: [{ ...detail.sources[0], exists: "yes" }],
      }),
    ).toBeNull();
  });

  it("returns null when a nested work directory file count is negative", () => {
    expect(
      normalizeProjectDetail({
        ...detail,
        work_dirs: [{ ...detail.work_dirs[0], file_count: -1 }],
      }),
    ).toBeNull();
  });
});
