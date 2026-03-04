# Legacy

Archived scripts kept for reference. These have been superseded by the current codebase but contain useful implementation details.

## Files

| File | Superseded by | Notes |
|------|---------------|-------|
| `panoex_gui.py` | `reconstruction_gui/reconstruction_zone.py` | Original 7-tab CustomTkinter GUI (~4200 lines). The Source, Prepare, and Gaps tabs were ported into Masking Studio as individual tab modules. |
| `equirect_to_perspectives.py` | `prep360/core/reframer.py` + `prep360/core/colmap_export.py` | Standalone equirect→perspective script with COLMAP camera.txt/images.txt export, Metashape XML, rig config, and mask reprojection. Some COLMAP export features here were later reimplemented in `colmap_export.py`. |
| `filter_blurry.py` | `prep360/core/blur_filter.py` | Original blur detection script using Laplacian variance. The core algorithm was absorbed into `blur_filter.py` with additional Sobel and Brenner metrics. |

## Migration notes

- The preset name `panoex_default` was renamed to `prep360_default`. If you have saved presets at `~/.panoex_presets.json`, they'll need the key updated manually.
- Cache directories moved from `~/.panoex_*` to `~/.prep360_*` (queue, SAM weights, shadow weights).
