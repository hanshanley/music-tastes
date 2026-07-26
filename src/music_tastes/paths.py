"""Filesystem layout and small shared helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
DERIVED = DATA / "derived"
CACHE = DATA / "cache"
LYRICS_CACHE = CACHE / "lyrics_cache"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _d in (RAW, INTERIM, DERIVED, CACHE, LYRICS_CACHE, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Read a setting from the environment, failing loudly when it is required."""
    value = os.environ.get(name, default)
    if required and not value:
        raise SystemExit(
            f"Missing required setting {name!r}. Copy .env.example to .env and fill it in."
        )
    return value


def user_agent() -> str:
    """User-Agent string identifying this project, as MusicBrainz policy requires."""
    contact = env("MUSICBRAINZ_CONTACT") or "no-contact-configured"
    return f"music-tastes/0.1 (research; {contact})"
