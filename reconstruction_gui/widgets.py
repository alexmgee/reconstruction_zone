"""
Shared widgets for Reconstruction Zone (and legacy apps).

Provides Section, CollapsibleSection, Tooltip, and slider_row — the
building blocks used across all tabs.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Optional

# ── Design System Constants ──────────────────────────────────────────
# See docs/DESIGN_SYSTEM.md for rationale and usage rules.

# Color roles
COLOR_ACTION_PRIMARY     = "#16a34a"
COLOR_ACTION_PRIMARY_H   = "#15803d"
COLOR_ACTION_SECONDARY   = "#1976D2"
COLOR_ACTION_SECONDARY_H = "#1565C0"
COLOR_ACTION_DANGER      = "#ab3434"
COLOR_ACTION_DANGER_H    = "#7a0000"
COLOR_ACTION_MUTED       = "#666666"
COLOR_ACTION_MUTED_H     = "#555555"

# Surfaces
COLOR_SURFACE_BG         = "#1a1a1a"
COLOR_SURFACE_CARD       = "#2b2b2b"
COLOR_SURFACE_INPUT      = "#343638"
COLOR_SURFACE_CONSOLE    = "#1e1e1e"

# Borders
COLOR_BORDER_SECTION     = ("#d1d5db", "#404040")  # (light, dark)

# Text
COLOR_TEXT_PRIMARY        = "#e0e0e0"
COLOR_TEXT_DIM            = "#888888"
COLOR_TEXT_MUTED          = "#9ca3af"
COLOR_TEXT_PLACEHOLDER    = "#6b6b6b"
COLOR_TEXT_DISABLED       = "#555555"
COLOR_TEXT_ON_ACTION      = "#ffffff"

# Slider
COLOR_SLIDER_TRACK        = "#4a4d50"
COLOR_SLIDER_THUMB        = "#1f6aa5"

# Section title states
COLOR_SECTION_BRIGHT      = ("#1a1a1a", "#e0e0e0")
COLOR_SECTION_DIM         = ("#888888", "#666666")

# Typography
FONT_HEADING_SECTION      = ("", 12, "bold")       # pass to CTkFont(size=, weight=)
FONT_LABEL_FIELD          = ("", 12)
FONT_LABEL_INLINE         = ("", 12, "bold")
FONT_LABEL_CHECKBOX       = ("", 12)
FONT_TEXT_SUBTITLE        = ("Consolas", 10)
FONT_TEXT_MONO_VALUE      = ("Consolas", 11)
FONT_TEXT_UNIT            = ("Consolas", 10)
FONT_TEXT_STATUS          = ("", 11)
FONT_TEXT_PLACEHOLDER     = ("", 12)
FONT_TEXT_CONSOLE         = ("Consolas", 10)
FONT_TEXT_BTN_PRIMARY     = ("", 13, "bold")
FONT_TEXT_BTN_SECONDARY   = ("", 12)

# Layout
LABEL_FIELD_WIDTH         = 75
BROWSE_BUTTON_WIDTH       = 36

# Spacing
SECTION_OUTER_PADX        = 4
SECTION_OUTER_PADY        = (0, 8)
SUBSECTION_PADX           = (10, 2)
SUBSECTION_PADY           = (4, 0)
ROW_PADX                  = 6
ROW_PADY                  = 3
ENTRY_GAP                 = (6, 4)     # (left of entry, right before button)
ACTION_BAR_PADX           = 6
ACTION_BAR_PADY           = (12, 4)
CONTENT_INSET             = 8

# Widget heights
HEIGHT_ACTION_BAR         = 38
HEIGHT_INLINE             = 28         # inline buttons, entries, browse, queue buttons
HEIGHT_NAV                = 24         # nav arrows, toolbar toggles
HEIGHT_PROGRESS           = 8

# Corner radius (explicit overrides — CTk default ~6 is fine for most)
RADIUS_TOOLTIP            = 0          # sharp — visually distinct from UI
RADIUS_PILL               = 4          # stage pills, tags

# Tooltip
COLOR_TOOLTIP_BG          = "#1e1e2e"
COLOR_TOOLTIP_FG          = "#e0e0e0"
FONT_TOOLTIP              = ("Consolas", 10)
TOOLTIP_PADX              = 8
TOOLTIP_PADY              = 4
TOOLTIP_WRAPLENGTH        = 350
TOOLTIP_DELAY_MS          = 400

# Textbox variants
FONT_TEXTBOX_COMPACT      = ("Consolas", 10)   # console, queue stats, gap summary
FONT_TEXTBOX_TALL         = ("Consolas", 11)   # analysis results, alignment log


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
        kwargs.setdefault("border_color", COLOR_BORDER_SECTION)
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
                header, text=subtitle,
                font=FONT_TEXT_SUBTITLE, text_color=COLOR_TEXT_MUTED,
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

    _CLR_BRIGHT = COLOR_SECTION_BRIGHT
    _CLR_DIM = COLOR_SECTION_DIM

    def __init__(
        self,
        master,
        title: str,
        subtitle: str = "",
        expanded: bool = False,
        core: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", COLOR_BORDER_SECTION)
        super().__init__(master, **kwargs)

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=(4, 0))
        self.header = hdr

        self._expanded = expanded
        self._title = title
        self._core = core          # core sections are always bright
        self._active = False       # feature enabled via checkbox
        self._hovering = False

        # Initial text color: core sections always bright, others follow state
        init_color = self._CLR_BRIGHT if (core or expanded) else self._CLR_DIM

        self.toggle_btn = ctk.CTkButton(
            hdr, text=f"{'▼' if expanded else '▶'} {title}",
            width=0, anchor="w", fg_color="transparent",
            hover_color=("gray75", "gray25"),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=init_color,
            command=self._toggle, height=24,
        )
        self.toggle_btn.pack(side="left")

        # Hover brightening for non-core sections
        if not core:
            self.toggle_btn.bind("<Enter>", self._on_hover_enter, add="+")
            self.toggle_btn.bind("<Leave>", self._on_hover_leave, add="+")

        if subtitle:
            ctk.CTkLabel(
                hdr, text=subtitle,
                font=FONT_TEXT_SUBTITLE, text_color=COLOR_TEXT_MUTED,
            ).pack(side="left", padx=(6, 0))

        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if expanded:
            self.content.pack(fill="x", padx=4, pady=(0, 4))

    def _resolve_color(self):
        """Return the correct text color based on current state."""
        if self._core or self._active or self._expanded or self._hovering:
            return self._CLR_BRIGHT
        return self._CLR_DIM

    def _update_title_color(self):
        self.toggle_btn.configure(text_color=self._resolve_color())

    def _on_hover_enter(self, _event):
        self._hovering = True
        self._update_title_color()

    def _on_hover_leave(self, _event):
        self._hovering = False
        self._update_title_color()

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.configure(
            text=f"{'▼' if self._expanded else '▶'} {self._title}")
        self._update_title_color()
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

    def set_active(self, active: bool):
        """Visually indicate whether this section's feature is enabled.

        Active sections get bright title text; inactive ones go grey.
        Call this from an enable/disable checkbox command callback.
        """
        self._active = active
        self._update_title_color()


class Tooltip:
    """Hover tooltip for any tkinter/CTk widget.

    Usage::

        btn = ctk.CTkButton(parent, text="Run")
        Tooltip(btn, "Start the extraction pipeline")

    Multi-line text is supported — just pass a string with newlines.
    """

    DELAY_MS = TOOLTIP_DELAY_MS
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
        label = tk.Label(
            tw, text=self._text, justify="left",
            background=COLOR_TOOLTIP_BG, foreground=COLOR_TOOLTIP_FG,
            relief="solid", borderwidth=1,
            font=FONT_TOOLTIP, padx=TOOLTIP_PADX, pady=TOOLTIP_PADY,
            wraplength=TOOLTIP_WRAPLENGTH,
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
) -> ctk.DoubleVar:
    """Pack label + slider + value readout into *parent*, return the DoubleVar.

    Usage::

        conf_var = slider_row(frame, "Confidence", 0, 1, 0.7, steps=100)
    """
    ctk.CTkLabel(parent, text=f"{label}:").pack(side="left", padx=(pad_left, 2))
    var = ctk.DoubleVar(value=default)
    val_lbl = ctk.CTkLabel(
        parent, text=f"{default:{fmt}}", width=0,
        font=FONT_TEXT_MONO_VALUE,
    )
    slider_kwargs = dict(
        from_=from_, to=to, variable=var, width=width,
        command=lambda v: val_lbl.configure(text=f"{v:{fmt}}"),
    )
    if steps is not None:
        slider_kwargs["number_of_steps"] = steps
    ctk.CTkSlider(parent, **slider_kwargs).pack(side="left", padx=0)
    val_lbl.pack(side="left", padx=0)
    return var
