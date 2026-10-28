"""Shared chart style, matching the house style used across these projects.

This mirrors ``personal_projects/uk_decline/vizstyle`` and the theme embedded in
``pre1870_reapportionment_package/scripts/generate_figures.py``: warm off-white
background, serif type, muted grid, no top/right spines, terracotta as the focus
colour, and an italic source note at the figure margin.

It is vendored here rather than imported so this repository stays self-contained,
and shares its palette and typography so charts from these projects sit together.

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
    "CHARTS", "chart_filename", "chart_heading",
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

# Colours for trend comparisons; headline charts show only covered songs.
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

CHARTS = {
    "relationship_share": ("relationship-songs", "Relationship songs", "Share of covered songs"),
    "independence_share": (
        "relationship-independence", "Independence in relationship songs", "Share of relationship songs",
    ),
    "heartbreak_share": (
        "relationship-heartbreak", "Heartbreak in relationship songs", "Share of relationship songs",
    ),
    "devotion_share": (
        "relationship-devotion", "Devotion in relationship songs", "Share of relationship songs",
    ),
    "lyric_valence": ("lyric-positivity-nrc", "Positivity in hit-song lyrics", "NRC word rating (0-1)"),
    "vader_valence": ("lyric-positivity-vader", "A second view of lyrical positivity", "VADER dictionary mean"),
    "lyric_sadness": ("sadness-associated-words", "Sadness-associated words", "Share of lyric words"),
    "lyric_joy": ("joy-associated-words", "Joy-associated words", "Share of lyric words"),
    "lyric_length": ("words-per-song", "How long are hit-song lyrics?", "Words per song"),
    "bpm": ("song-tempo", "The tempo of hit songs", "Beats per minute"),
    "minor_key_share": ("minor-key-songs", "Hit songs in a minor key", "Share of covered songs"),
    "acoustic_mood_happy": ("audio-happiness-score", "Audio happiness scores", "Model score"),
    "acoustic_mood_sad": ("audio-sadness-score", "Audio sadness scores", "Model score"),
    "coverage_by_year": ("lyric-coverage", "How much of the chart has lyrics?", "Share with retrieved lyrics"),
    "nrc_vad_robustness": ("lyrical-emotion-dimensions", "Three dimensions of lyrical emotion", "Word rating (0-1)"),
    "stance_shares": ("relationship-theme-shares", "Themes in relationship songs", "Share of relationship songs"),
    "stance_composition": ("relationship-theme-mix", "The main theme in relationship songs", "Share of relationship songs"),
}


def chart_filename(metric: str, *, decade: bool = False) -> str:
    return CHARTS[metric][0] + ("-by-decade" if decade else "") + ".png"


def chart_heading(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=34, fontsize=18)
    ax.text(0, 1.025, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", color=MUTED, fontsize=10)


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
             pad: float = 0.6, top: float | None = None) -> pathlib.Path:
    """tight_layout + optional bottom margin, save at house DPI, close the figure."""
    fig.tight_layout(pad=pad)
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    if top is not None:
        fig.subplots_adjust(top=top)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
