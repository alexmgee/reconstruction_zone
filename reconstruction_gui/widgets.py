"""
Shared widgets for Reconstruction Zone (and legacy apps).

Provides Section, CollapsibleSection, Tooltip, and slider_row — the
building blocks used across all tabs.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Optional


class Section(ctk.CTkFrame):
    """Always-visible section with thin border, title, and inline subtitle.

    Access the content area via ``self.content``.

    Usage::

        sec = Section(parent, "Model / Geometry", subtitle="YOLO + SAM")
        ctk.CTkLabel(sec.content, text="...").pack()
    """

    def __init__(
        self,
        master,
        title: str,
        subtitle: str = "",
        **kwargs,
    ):
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", ("#d1d5db", "#404040"))
        super().__init__(master, **kwargs)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=6, pady=(4, 0))
        ctk.CTkLabel(
            header, text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).pack(side="left")
        if subtitle:
            ctk.CTkLabel(
                header, text=f"— {subtitle}",
                font=("Consolas", 10), text_color="#9ca3af",
                anchor="w",
            ).pack(side="left", padx=(6, 0))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="x", padx=4, pady=(0, 4))


class CollapsibleSection(ctk.CTkFrame):
    """Section with ▼/▶ toggle to show/hide content.

    Access the content area via ``self.content``.

    Usage::

        sec = CollapsibleSection(parent, "Advanced", expanded=False)
        ctk.CTkSlider(sec.content, ...).pack()
        sec.expand()   # open programmatically
        sec.collapse() # close programmatically
    """

    def __init__(
        self,
        master,
        title: str,
        subtitle: str = "",
        expanded: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", ("#d1d5db", "#404040"))
        super().__init__(master, **kwargs)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=(4, 0))

        self._expanded = expanded
        self._title = title
        self.toggle_btn = ctk.CTkButton(
            hdr, text=f"{'▼' if expanded else '▶'} {title}",
            anchor="w", fg_color="transparent",
            hover_color=("gray75", "gray25"),
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._toggle, height=24,
        )
        self.toggle_btn.pack(side="left")
        if subtitle:
            ctk.CTkLabel(
                hdr, text=f"— {subtitle}",
                font=("Consolas", 10), text_color="#9ca3af",
            ).pack(side="left", padx=6)

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=4, pady=(0, 4))

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.configure(
            text=f"{'▼' if self._expanded else '▶'} {self._title}")
        if self._expanded:
            self.content.pack(fill="x", padx=4, pady=(0, 4))
        else:
            self.content.pack_forget()

    def is_expanded(self) -> bool:
        return self._expanded

    def expand(self):
        if not self._expanded:
            self._toggle()

    def collapse(self):
        if self._expanded:
            self._toggle()


class Tooltip:
    """Hover tooltip for any tkinter/CTk widget.

    Usage::

        btn = ctk.CTkButton(parent, text="Run")
        Tooltip(btn, "Start the extraction pipeline")

    Multi-line text is supported — just pass a string with newlines.
    """

    DELAY_MS = 400  # hover delay before showing
    OFFSET_X = 12
    OFFSET_Y = 8

    def __init__(self, widget, text: str):
        self._widget = widget
        self._text = text
        self._tip_window: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event):
        self._cancel()
        self._after_id = self._widget.after(self.DELAY_MS, self._show)

    def _on_leave(self, event):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window or not self._text:
            return
        x = self._widget.winfo_rootx() + self.OFFSET_X
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + self.OFFSET_Y
        tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.wm_attributes("-topmost", True)
        # Dark tooltip matching the app theme
        label = tk.Label(
            tw, text=self._text, justify="left",
            background="#1e1e2e", foreground="#e0e0e0",
            relief="solid", borderwidth=1,
            font=("Consolas", 10), padx=8, pady=4,
            wraplength=350,
        )
        label.pack()
        self._tip_window = tw

    def _hide(self):
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value


def slider_row(
    parent,
    label: str,
    from_: float,
    to: float,
    default: float,
    steps: Optional[int] = None,
    fmt: str = ".2f",
    width: int = 100,
    pad_left: int = 0,
    tooltip: Optional[str] = None,
) -> ctk.DoubleVar:
    """Pack label + slider + value readout into *parent*, return the DoubleVar.

    Usage::

        conf_var = slider_row(frame, "Confidence", 0, 1, 0.7, steps=100)
    """
    ctk.CTkLabel(parent, text=f"{label}:").pack(side="left", padx=(pad_left, 2))
    var = ctk.DoubleVar(value=default)
    val_lbl = ctk.CTkLabel(
        parent, text=f"{default:{fmt}}", width=40,
        font=("Consolas", 11),
    )
    slider_kwargs = dict(
        from_=from_, to=to, variable=var, width=width,
        command=lambda v: val_lbl.configure(text=f"{v:{fmt}}"),
    )
    if steps is not None:
        slider_kwargs["number_of_steps"] = steps
    _sl = ctk.CTkSlider(parent, **slider_kwargs)
    _sl.pack(side="left", padx=2)
    if tooltip:
        Tooltip(_sl, tooltip)
    val_lbl.pack(side="left")
    return var
