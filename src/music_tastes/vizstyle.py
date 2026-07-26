"""Shared chart style, matching the house style used across these projects.

This mirrors ``personal_projects/uk_decline/vizstyle`` and the theme embedded in
``pre1870_reapportionment_package/scripts/generate_figures.py``: warm off-white
background, serif type, muted grid, no top/right spines, terracotta as the focus
colour, and an italic source note at the figure margin.

It is vendored here rather than imported so this repository stays self-contained,
but the palette and rcParams are kept byte-identical to the shared module so charts
from all three projects sit together without looking mismatched.

Usage::

    from music_tastes.vizstyle import house_style, ACCENT, source_note, save_fig

    house_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, y, color=ACCENT, linewidth=2.8)
    source_note(fig, "Data: Billboard Hot 100 via mhollingshead/billboard-hot-100.")
    save_fig(fig, "reports/figures/example.png")
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: analyses render to PNG, never to a display

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "PALETTE", "BG", "CARD", "TEXT", "MUTED", "GRID", "ACCENT", "BLUE", "GOLD",
    "GREEN", "RC_PARAMS", "house_style", "source_note", "end_label", "save_fig",
    "white_stroke", "SERIES_COLOURS", "STANCE_COLOURS",
]

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#F7F5F0"       # background (warm off-white)
CARD = "#EFEDE8"     # slightly darker panel fill
TEXT = "#1A1A1A"     # primary text / ink
MUTED = "#6B6B6B"    # secondary text, ticks, source notes
GRID = "#D6D3CC"     # gridlines, spines
ACCENT = "#C85A3D"   # terracotta — the focus series
BLUE = "#3D6F8C"     # reference / comparison series
GOLD = "#C2993E"     # third series
GREEN = "#4A7C59"    # "better" / positive series

PALETTE: dict[str, str] = {
    "bg": BG, "card": CARD, "text": TEXT, "muted": MUTED, "grid": GRID,
    "accent": ACCENT, "blue": BLUE, "gold": GOLD, "green": GREEN,
}

# ── rcParams ─────────────────────────────────────────────────────────────────
RC_PARAMS: dict[str, Any] = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.alpha": 0.6, "grid.linewidth": 0.5,
    "font.family": "serif", "font.size": 12,
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 12,
    "figure.titlesize": 18,
    "legend.framealpha": 0.0, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "text.parse_math": False,
}

# The two headline variants shown on every trend chart. Terracotta is the focus
# series (all covered songs); blue is the coverage-controlled comparison.
SERIES_COLOURS = {
    "weighted_all": ACCENT,
    "weighted_complete_case": BLUE,
    "unweighted_all": GOLD,
    "unweighted_complete_case": GREEN,
}

# Stance colours: terracotta reserved for independence, the stance the project is
# about, so it reads as the focus wherever stances are shown together.
STANCE_COLOURS = {
    "independence": ACCENT,
    "heartbreak": BLUE,
    "devotion": GREEN,
    "longing": GOLD,
    "casual": "#8C6D8C",
    "conflict": "#7A5C3D",
}

_WHITE_STROKE = [pe.withStroke(linewidth=3.0, foreground="white")]


def white_stroke() -> list:
    """Path-effects list giving text a white outline (for labels over data)."""
    return list(_WHITE_STROKE)


def house_style() -> None:
    """Apply the shared house style to matplotlib's global rcParams. Idempotent."""
    plt.rcParams.update(RC_PARAMS)


def source_note(fig, text: str, *, x: float = 0.01, y: float = 0.01,
                ha: str = "left") -> None:
    """Add the standard italic, muted source note at the figure margin."""
    fig.text(x, y, text, ha=ha, fontsize=8, color=MUTED, style="italic")


def end_label(ax, x, y, text: str, color: str, *, fontsize: float = 10.5,
              dx: str = "  ") -> None:
    """Label a series at its end point, with a white halo for legibility."""
    ax.text(x, y, f"{dx}{text}", fontsize=fontsize, fontweight="bold",
            color=color, va="center", ha="left", path_effects=white_stroke())


def save_fig(fig, out_path, *, dpi: int = 200, bottom: float | None = 0.12,
             pad: float = 0.6) -> pathlib.Path:
    """tight_layout + optional bottom margin, save at house DPI, close the figure."""
    fig.tight_layout(pad=pad)
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
