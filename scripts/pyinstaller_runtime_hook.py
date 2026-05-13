"""
PyInstaller runtime hook — runs before the app starts.

1. Adds bundled ffmpeg/ directory to PATH
2. Pre-imports torch submodules in correct order to avoid circular imports
"""
import os
import sys

# When running as a PyInstaller bundle, sys._MEIPASS points to the
# extracted temp directory (onefile) or the dist directory (onedir).
if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    ffmpeg_dir = os.path.join(base, 'ffmpeg')
    if os.path.isdir(ffmpeg_dir):
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

# Fix PyTorch circular import in PyInstaller bundles.
# torch/__init__.py imports torch.nested which needs torch.autograd,
# but autograd isn't loaded yet during init. Pre-loading autograd
# via the C extension breaks the cycle.
try:
    import importlib
    # Load the C extension first — it has no Python-level circular deps
    import torch._C
    # Now load autograd before nested tries to access it
    importlib.import_module('torch.autograd')
    importlib.import_module('torch.autograd.function')
except Exception:
    pass
