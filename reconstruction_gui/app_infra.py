"""
Shared application infrastructure for Reconstruction Zone.

Provides AppInfrastructure mixin with logging, console redirect,
threading helpers, browse dialogs, and preferences.

The app class should:
  1. Inherit from both ctk.CTk and AppInfrastructure
  2. Call ``self._init_infrastructure()`` in __init__ after creating
     ``self.log_queue`` and ``self.log_textbox``
  3. Call ``self.protocol("WM_DELETE_WINDOW", self._on_close)``

Example::

    class MyApp(AppInfrastructure, ctk.CTk):
        def __init__(self):
            super().__init__()
            self.log_queue = queue.Queue()
            self.log_textbox = ctk.CTkTextbox(...)
            self._init_infrastructure()
"""

import json
import queue
import sys
import threading
from pathlib import Path
from tkinter import filedialog
from typing import Dict, Optional


class AppInfrastructure:
    """Mixin providing infrastructure shared across all tabs.

    Expects the following attributes to be set by the host app:

    - ``self.log_queue``: queue.Queue for log messages
    - ``self.log_textbox``: ctk.CTkTextbox for displaying logs
    - ``self.is_running``: bool
    - ``self.cancel_flag``: threading.Event
    """

    # Override in subclass to set the prefs file location
    _PREFS_FILE: Optional[Path] = None

    def _init_infrastructure(self):
        """Call once after log_queue and log_textbox are created."""
        self._setup_console_redirect()
        self._poll_log_queue()

    # ── logging ──

    def _setup_console_redirect(self):
        q = self.log_queue

        class _QW:
            def __init__(self):
                self.q = q
            def write(self, text):
                if text.strip():
                    self.q.put(text.rstrip())
            def flush(self):
                pass

        sys.stdout = _QW()
        sys.stderr = _QW()

    def log(self, msg: str):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_textbox.insert("end", msg + "\n")
                self.log_textbox.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_log_queue)

    # ── operation control ──

    def _start_operation(self, run_btn, stop_btn):
        """Disable run button, show stop button, start progress bar."""
        self.is_running = True
        self.cancel_flag.clear()
        if hasattr(self, "progress_bar"):
            self.progress_bar.start()
        run_btn.configure(state="disabled")
        stop_btn.pack(pady=(0, 10), padx=10, fill="x")

    def _stop_operation(self, run_btn, stop_btn):
        """Re-enable run button, hide stop button, stop progress bar."""
        self.is_running = False
        if hasattr(self, "progress_bar"):
            self.progress_bar.stop()
            self.progress_bar.set(1.0)
        run_btn.configure(state="normal")
        stop_btn.pack_forget()

    def stop_operation(self):
        if self.is_running:
            self.log("Stopping...")
            self.cancel_flag.set()

    # ── browse helpers ──

    def _entry_initialdir(self, entry_widget) -> Optional[str]:
        """Best-effort starting directory based on the current entry value."""
        try:
            raw = entry_widget.get().strip()
        except Exception:
            return None
        if not raw:
            return None

        path = Path(raw)
        if path.is_dir():
            return str(path)
        if path.is_file():
            return str(path.parent)

        for candidate in [path.parent, *path.parents]:
            if candidate and candidate.exists() and candidate.is_dir():
                return str(candidate)
        return None

    def _browse_dir_into(self, entry_widget, title: str = "Select Folder"):
        kwargs = {"title": title}
        initialdir = self._entry_initialdir(entry_widget)
        if initialdir:
            kwargs["initialdir"] = initialdir
        path = filedialog.askdirectory(**kwargs)
        if path:
            entry_widget.delete(0, "end")
            entry_widget.insert(0, path)

    def _browse_folder_for(self, entry, callback=None):
        kwargs = {"title": "Select Folder"}
        initialdir = self._entry_initialdir(entry)
        if initialdir:
            kwargs["initialdir"] = initialdir
        folder = filedialog.askdirectory(**kwargs)
        if folder:
            entry.delete(0, "end")
            entry.insert(0, folder)
            if callback:
                callback()

    def _browse_video_for(self, entry):
        kwargs = {
            "title": "Select Video",
            "filetypes": [
                ("Video Files", "*.mp4 *.mov *.avi *.mkv *.360 *.insv *.osv"),
                ("All Files", "*.*"),
            ],
        }
        initialdir = self._entry_initialdir(entry)
        if initialdir:
            kwargs["initialdir"] = initialdir
        path = filedialog.askopenfilename(**kwargs)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    def _browse_file_for(self, entry, title="Select File", filetypes=None):
        if filetypes is None:
            filetypes = [("All Files", "*.*")]
        kwargs = {"title": title, "filetypes": filetypes}
        initialdir = self._entry_initialdir(entry)
        if initialdir:
            kwargs["initialdir"] = initialdir
        path = filedialog.askopenfilename(**kwargs)
        if path:
            entry.delete(0, "end")
            entry.insert(0, path)

    # ── preferences ──

    def _load_prefs(self) -> dict:
        if self._PREFS_FILE is None:
            return {}
        try:
            return json.loads(self._PREFS_FILE.read_text())
        except Exception:
            return {}

    def _save_prefs(self):
        if self._PREFS_FILE is None:
            return
        try:
            self._PREFS_FILE.write_text(json.dumps(self._prefs, indent=2))
        except Exception:
            pass

    # ── textbox helper ──

    def _set_textbox(self, textbox, text: str):
        """Update a disabled CTkTextbox with new content."""
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", text)
        textbox.configure(state="disabled")

    # ── section header (legacy prep360 compat) ──

    def _section_header(self, parent, text):
        """Create a bold section label (used by prep360-style tabs)."""
        import customtkinter as ctk
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(pady=(15, 5), padx=10, anchor="w")
