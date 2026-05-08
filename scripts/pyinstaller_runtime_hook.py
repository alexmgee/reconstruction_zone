"""
PyInstaller runtime hook — runs before the app starts.

Adds bundled ffmpeg/ directory to PATH so subprocess calls find it.
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
