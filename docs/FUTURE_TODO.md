# Future TODO

Ideas and improvements identified during tab-by-tab GUI review. Not urgent — tracked here for when the right moment comes.

## Smart Improvements

### Direct-from-video motion selection for fisheye pipeline
Currently the fisheye pipeline extracts frames to disk at the `interval` rate, then runs `MotionSelector.select_from_paths()` on the extracted images. A smarter design would use `MotionSelector.select_from_video()` directly on the .osv file, skipping the intermediate frame dump. This would require teaching `select_from_video()` about dual-stream containers (front/back lens synchronization).

**Why not now**: `select_from_video()` only handles single-stream video. Dual-stream OSV containers need synchronized front/back reads. The current extract-then-select approach works fine — it's just wasteful of disk I/O for frames that get filtered out.

### Sky filter pitch-angle awareness
The sky filter (`SkyFilter`) uses 4 pixel-level metrics (brightness, saturation, keypoints, edges) with 3-of-4 voting. `SkyFilterConfig.pitch_threshold` exists in the config but is never used by the core `analyze()` method. A future improvement could use camera orientation metadata (from EXIF or 360 projection math) to only flag sky images when the camera is pointing steeply upward, rather than relying purely on pixel statistics.

**Why not now**: The pixel-level approach works well for most cases. Pitch awareness would need either EXIF orientation data (not always available) or integration with the reframer's view pitch angles (only relevant for 360 data).

### Feedback guidance system
Contextual, dynamic status messages across all tabs — not just static labels but responsive hints that update as the user changes settings. Examples:
- Output estimates before clicking Run ("30 frame pairs x 26 views = ~780 images")
- Disk space estimates for large extractions
- Warnings when settings conflict or produce no output
- Progress context during long operations ("Extracting frame 45/120...")
