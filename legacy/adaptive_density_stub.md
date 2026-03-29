# Adaptive Density — Archived Stub

Removed from the UI on 2026-03-29. The intended design was never implemented.

## Intended Logic
- Extract at the base interval (same as Fixed Interval)
- Detect gaps in coverage via scene/motion analysis
- Extract additional frames to fill gaps, enforcing max_gap
- Config field: `ExtractionConfig.max_gap: float = 3.0`

## What Was Implemented
- ExtractionMode.ADAPTIVE existed in the enum
- `_build_fixed_command()` was called (identical to Fixed mode)
- The gap-filling second phase was never built
- `max_gap` was defined but never read by any logic

## To Resurrect
- Implement a two-pass approach: fixed extract, then scene-detect within gaps
- Or use ffmpeg's scene filter with a minimum-interval floor (similar to Scene mode)
- The `max_gap` field still exists in ExtractionConfig and Preset
