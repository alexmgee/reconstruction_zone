# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Reconstruction Zone (Masking Studio)
#
# INITIAL DRAFT — needs testing and iteration. Known open items:
#   - Icon not yet created (placeholder comment below)
#   - CUDA DLL bundling may need manual binaries= entries depending on
#     PyTorch install layout (conda vs pip, cu124 vs cu126)
#   - UPX may corrupt CUDA/cuDNN DLLs — disable upx_exclude list if
#     the built app crashes on GPU inference
#   - Optional heavy deps (SAM3, RF-DETR, LiVOS, Cutie, ViTMatte) are
#     lazy-imported at runtime; they are NOT bundled here. Users download
#     model repos separately.
#   - ffmpeg/ffprobe are external tools and must be on PATH at runtime
#
# Usage:
#   python -m PyInstaller reconstruction_zone.spec --noconfirm
#   (or use scripts/build_gumroad.py for Gumroad distribution builds)

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# CustomTkinter ships theme JSON + assets that must be included
ctk_datas = collect_data_files('customtkinter')

# Include prep360 preset JSON files if any exist alongside code
prep360_datas = collect_data_files('prep360', include_py_files=False)

# Include the docs shipped with the GUI (tab help files)
gui_docs = [('reconstruction_gui/docs', 'reconstruction_gui/docs')]

a = Analysis(
    ['reconstruction_gui/reconstruction_zone.py'],
    pathex=[],
    binaries=[],
    datas=ctk_datas + prep360_datas + gui_docs,
    hiddenimports=[
        # --- PyTorch + CUDA ---
        # collect_submodules pulls in all torch.* / torchvision.* subpackages
        # so that lazy imports inside the app resolve correctly.
        *collect_submodules('torch'),
        *collect_submodules('torchvision'),

        # --- GUI framework ---
        *collect_submodules('customtkinter'),
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',

        # --- Core dependencies ---
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'cv2',
        'numpy',
        'scipy',
        'scipy.ndimage',

        # --- Optional but commonly present ---
        'py360convert',
        'ultralytics',

        # --- App packages (ensure PyInstaller finds them) ---
        'prep360',
        'prep360.distribution',
        'prep360.cli',
        'prep360.core',
        'prep360.core.analyzer',
        'prep360.core.extractor',
        'prep360.core.reframer',
        'prep360.core.fisheye_reframer',
        'prep360.core.presets',
        'prep360.core.queue_manager',
        'prep360.core.sky_filter',
        'prep360.core.blur_filter',
        'prep360.core.lut',
        'prep360.core.segmenter',
        'prep360.core.colmap_export',
        'prep360.core.gap_detector',
        'prep360.core.bridge_extractor',
        'prep360.core.adjustments',
        'prep360.core.sharpest_extractor',
        'prep360.core.geotagger',
        'prep360.core.srt_parser',
        'prep360.core.paired_split_video_extractor',
        'prep360.core.dual_fisheye_dataset',
        'prep360.core.fisheye_calibration',
        'prep360.core.motion_selector',
        'prep360.core.osv',
        'reconstruction_gui',
        'reconstruction_gui.reconstruction_pipeline',
        'reconstruction_gui.shadow_detection',
        'reconstruction_gui.app_infra',
        'reconstruction_gui.widgets',
        'reconstruction_gui.review_gui',
        'reconstruction_gui.review_masks',
        'reconstruction_gui.review_status',
        'reconstruction_gui.masking_queue',
        'reconstruction_gui.sam_refinement',
        'reconstruction_gui.matting',
        'reconstruction_gui.vos_propagation',
        'reconstruction_gui.colmap_validation',
        'reconstruction_gui.sam3_pipeline',
        'reconstruction_gui.tabs',
        'reconstruction_gui.tabs.source_tab',
        'reconstruction_gui.tabs.gaps_tab',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Large packages we don't need
        'matplotlib',
        'matplotlib.tests',
        'mpl_toolkits',
        'IPython',
        'jupyter',
        'jupyter_client',
        'jupyter_core',
        'notebook',
        'pytest',
        'setuptools',
        'pip',
        'wheel',
        'distutils',
        # Test suites
        'tkinter.test',
        'unittest',
        'doctest',
        # Unused torch extras
        'torch.testing',
        'torch.utils.tensorboard',
        'tensorboard',
        'caffe2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ReconstructionZone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed app — no console window
    disable_windowed_traceback=False,
    # icon='assets/icon.ico',  # TODO: create and add app icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # CUDA and cuDNN DLLs can be corrupted by UPX
        'cublas*.dll',
        'cudart*.dll',
        'cudnn*.dll',
        'cufft*.dll',
        'curand*.dll',
        'cusolver*.dll',
        'cusparse*.dll',
        'nvinfer*.dll',
        'nvrtc*.dll',
        # Also skip torch's own DLLs
        'torch_*.dll',
        'c10*.dll',
    ],
    name='ReconstructionZone',
)
