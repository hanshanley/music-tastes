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


# Which pipeline stage produces each derived artefact. Used to turn a missing-file
# traceback into an instruction, because the pipeline is resumable and users
# routinely arrive at an analysis stage before its inputs exist.
_PRODUCED_BY = {
    "chart_entries.parquet": "ingest-charts",
    "chart_entries_resolved.parquet": "resolve-songs",
    "songs.parquet": "resolve-songs",
    "songs_weighted.parquet": "exposure",
    "song_year_exposure.parquet": "exposure",
    "nrc_vad.parquet": "lexicons",
    "nrc_emolex.parquet": "lexicons",
    "vader.parquet": "lexicons",
    "lyrics_index.parquet": "fetch-lyrics",
    "acoustic_features.parquet": "enrich-acoustic",
    "lyric_features_method_a.parquet": "features-a",
    "lyric_features_method_b.parquet": "stance-b",
    "language_profile.parquet": "language",
    "yearly_series.parquet": "trends",
    "decade_series.parquet": "trends",
    "contextual_valence.parquet": "validity",
    "prompt_robustness.parquet": "prompt-robustness",
}


def require(path, *, hint: str | None = None):
    """Return ``path`` if it exists, else fail with the stage that would create it.

    Analysis stages read artefacts produced by earlier stages. Letting pandas raise
    a bare FileNotFoundError tells the user which file is missing but not what to
    do about it; this tells them which command to run.
    """
    from pathlib import Path

    path = Path(path)
    if path.exists():
        return path
    stage = _PRODUCED_BY.get(path.name)
    if hint is None:
        hint = (
            f"Run `music-tastes {stage}` first."
            if stage
            else "Run the earlier pipeline stages first."
        )
    raise SystemExit(f"Missing {path.name} ({path}).\n{hint}")
