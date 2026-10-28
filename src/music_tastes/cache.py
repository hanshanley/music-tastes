"""Content-addressed dependencies for expensive analysis stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import DERIVED, atomic_json


def fingerprint(config: dict, paths: list[Path]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode())
        if not path.exists():
            digest.update(b"missing")
            continue
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def analysis_signature(config: dict) -> str:
    from .stance_nli import MODEL_REVISION

    return fingerprint(
        {**config, "model_revision": MODEL_REVISION},
        [DERIVED / name for name in (
            "songs_weighted.parquet", "lyrics_index.parquet",
            "lyric_features_method_a.parquet", "lyric_features_method_b.parquet",
        )],
    )


def cache_matches(path: Path, signature: str) -> bool:
    try:
        return path.exists() and json.loads(
            path.with_suffix(".signature.json").read_text()
        )["signature"] == signature
    except (OSError, ValueError, KeyError):
        return False


def mark_cache(path: Path, signature: str) -> None:
    atomic_json(path.with_suffix(".signature.json"), {"signature": signature})
