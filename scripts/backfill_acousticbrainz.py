#!/usr/bin/env python
"""Fetch AcousticBrainz features for every MBID already resolved.

The main enrichment stage resolves all MusicBrainz ids first and only then queries
AcousticBrainz, so acoustic features lag the MBID lookups by hours on a full run.
This fills that gap: it reads whatever is in the MBID cache now and fetches features
for the candidate recordings, letting BPM and mood analysis start before the slow
MusicBrainz pass finishes.

Safe to run alongside the main stage. Features are cached per recording, so no work
is repeated and the two processes cannot conflict.
"""

from __future__ import annotations

import json

from tqdm import tqdm

from music_tastes.enrich_acoustic import (
    AB_BATCH,
    MBID_CACHE,
    _ab_path,
    fetch_acousticbrainz,
)


def main() -> None:
    candidates: list[str] = []
    for path in MBID_CACHE.rglob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        cands = rec.get("candidate_mbids")
        if isinstance(cands, list):
            candidates.extend(cands)
        elif isinstance(rec.get("mbid"), str):
            candidates.append(rec["mbid"])

    candidates = list(dict.fromkeys(candidates))
    todo = [m for m in candidates if not _ab_path(m).exists()]
    print(f"{len(candidates):,} candidate recordings, {len(todo):,} not yet fetched")

    for i in tqdm(range(0, len(todo), AB_BATCH), desc="acousticbrainz", unit="batch"):
        fetch_acousticbrainz(todo[i : i + AB_BATCH])

    have = sum(1 for m in candidates if _ab_path(m).exists() and _ab_path(m).stat().st_size > 2)
    print(f"recordings with features: {have:,}")


if __name__ == "__main__":
    main()
