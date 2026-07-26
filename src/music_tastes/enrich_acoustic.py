"""Stage 6b: acoustic enrichment -- BPM, key and Essentia mood.

Route
-----
Chart song -> MusicBrainz recording MBID -> AcousticBrainz features.

Spotify's audio-features endpoint (tempo, valence) was deprecated for newly
registered applications on 2024-11-27 and is unavailable to this project, so the
open MetaBrainz stack is used instead. AcousticBrainz is community-submitted
Essentia analysis: coverage is uneven and skews toward catalogue that someone
bothered to submit, which is exactly the kind of year-dependent bias the coverage
audit exists to catch. Nothing here is imputed; unmatched songs stay null.

Rate limits
-----------
MusicBrainz asks anonymous clients for at most one request per second and a
descriptive User-Agent, both enforced in :mod:`music_tastes.http`. That makes the
MBID lookup the bottleneck at roughly one song per second, so songs are processed in
the same year-balanced priority order used by the stance stage: any prefix of the run
is a usable year-stratified sample.

AcousticBrainz accepts up to 25 recording ids per request, so feature retrieval is
batched and costs comparatively little.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from .http import get
from .paths import CACHE, DERIVED
from .resolve_songs import normalize_artist, normalize_title, title_variants

MB_SEARCH = "https://musicbrainz.org/ws/2/recording"
AB_LOW = "https://acousticbrainz.org/api/v1/low-level"
AB_HIGH = "https://acousticbrainz.org/api/v1/high-level"

MBID_CACHE = CACHE / "mbid"
AB_BATCH = 20

# MusicBrainz returns its own 0-100 relevance score; require a strong one *and*
# independent title/artist agreement before accepting a link.
MB_MIN_SCORE = 80
MIN_TITLE_SIM = 0.85
MIN_ARTIST_SIM = 0.70


def _mbid_path(song_id: str):
    d = MBID_CACHE / song_id[1:3]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{song_id}.json"


def _sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def _escape(text: str) -> str:
    """Escape Lucene syntax for the MusicBrainz query parser."""
    out = str(text)
    for ch in ['\\', '"', "+", "-", "&", "|", "!", "(", ")", "{", "}",
               "[", "]", "^", "~", "*", "?", ":", "/"]:
        out = out.replace(ch, " ")
    return " ".join(out.split())


def lookup_mbid(song) -> dict:
    """Find the MusicBrainz recording for a chart song.

    Returns a record with the MBID and match evidence, or a null MBID when nothing
    clears the thresholds. Either way the result is cached so the slow lookup runs
    once per song.
    """
    cached = _mbid_path(song.song_id)
    if cached.exists():
        return json.loads(cached.read_text())

    variants = title_variants(song.title_display)
    base = variants[-1] if variants else song.title_display
    artist = normalize_artist(song.artist_display)

    rec = {
        "song_id": song.song_id,
        "mbid": None,
        "mb_score": None,
        "mb_title": None,
        "mb_artist": None,
        "title_similarity": None,
        "artist_similarity": None,
        "isrc": None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    query = f'recording:"{_escape(base)}" AND artist:"{_escape(artist)}"'
    try:
        resp = get(
            MB_SEARCH,
            namespace="musicbrainz",
            params={"query": query, "fmt": "json", "limit": 8, "inc": "isrcs"},
        )
    except RuntimeError:
        cached.write_text(json.dumps(rec))
        return rec

    if resp.status != 200:
        cached.write_text(json.dumps(rec))
        return rec

    want_title_forms = title_variants(song.title_display)
    best = None
    candidates: list[str] = []
    for cand in resp.json().get("recordings", []):
        if cand.get("score", 0) < MB_MIN_SCORE:
            continue
        got_title = cand.get("title") or ""
        credits = cand.get("artist-credit") or []
        got_artist = " ".join(c.get("name", "") for c in credits if isinstance(c, dict))

        ts = max(
            (_sim(w, normalize_title(t)) for w in want_title_forms
             for t in title_variants(got_title)),
            default=0.0,
        )
        as_ = _sim(artist, normalize_artist(got_artist))
        if ts < MIN_TITLE_SIM or as_ < MIN_ARTIST_SIM:
            continue
        if cand.get("id"):
            candidates.append(cand["id"])
        key = ts + as_
        if best is None or key > best[0]:
            best = (key, cand, ts, as_, got_title, got_artist)

    if best:
        _, cand, ts, as_, got_title, got_artist = best
        isrcs = cand.get("isrcs") or []
        rec.update(
            mbid=cand.get("id"),
            mb_score=cand.get("score"),
            mb_title=got_title,
            mb_artist=got_artist,
            title_similarity=round(ts, 3),
            artist_similarity=round(as_, 3),
            isrc=isrcs[0] if isrcs else None,
        )
    # AcousticBrainz is keyed to a specific *recording*, and a hit song typically has
    # many recording MBIDs (single, album, reissue, compilation) of which only some
    # have a community submission. Keeping every acceptable candidate and querying
    # them all raises feature coverage substantially over using the best match alone.
    rec["candidate_mbids"] = candidates[:8]

    cached.write_text(json.dumps(rec))
    return rec


def fetch_acousticbrainz(mbids: list[str]) -> dict[str, dict]:
    """Fetch low-level and high-level features for a batch of MBIDs."""
    out: dict[str, dict] = {}
    if not mbids:
        return out
    ids = ";".join(mbids)

    try:
        low = get(AB_LOW, namespace="acousticbrainz", params={"recording_ids": ids})
        high = get(AB_HIGH, namespace="acousticbrainz", params={"recording_ids": ids})
    except RuntimeError:
        return out

    low_data = low.json() if low.status == 200 else {}
    high_data = high.json() if high.status == 200 else {}

    for mbid in mbids:
        rec: dict[str, float | str | None] = {"mbid": mbid}
        entry = (low_data.get(mbid) or {}).get("0") or {}
        ll = entry.get("lowlevel") or {}
        rhythm = entry.get("rhythm") or {}
        tonal = entry.get("tonal") or {}
        if rhythm:
            rec["bpm"] = rhythm.get("bpm")
            rec["onset_rate"] = rhythm.get("onset_rate")
        if tonal:
            rec["key"] = tonal.get("key_key")
            rec["scale"] = tonal.get("key_scale")
        if ll:
            rec["average_loudness"] = ll.get("average_loudness")

        hentry = (high_data.get(mbid) or {}).get("0") or {}
        hl = hentry.get("highlevel") or {}
        for field in ("mood_happy", "mood_sad", "mood_aggressive",
                      "mood_relaxed", "danceability"):
            block = hl.get(field) or {}
            probs = block.get("all") or {}
            # Store the probability of the positive class rather than the argmax
            # label, so downstream code keeps the uncertainty.
            positive = "danceable" if field == "danceability" else field.replace("mood_", "")
            if probs:
                rec[f"ab_{field}"] = probs.get(positive)
        if len(rec) > 1:
            out[mbid] = rec
    return out


def run(limit: int | None = None, workers: int = 6) -> pd.DataFrame:
    from .stance_nli import _priority_order

    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")
    order = _priority_order(list(songs["song_id"]), songs)
    if limit:
        order = order[:limit]
    order_set = set(order)
    targets = [s for s in songs.itertuples() if s.song_id in order_set]
    # Preserve the year-balanced priority order.
    rank = {sid: i for i, sid in enumerate(order)}
    targets.sort(key=lambda s: rank[s.song_id])

    # MusicBrainz throttles by delaying responses rather than rejecting them: under
    # load a single query can take 11 seconds while the next takes 0.4. A serial
    # loop therefore runs far *below* the permitted 1 request/second, because it
    # spends most of its time waiting. Issuing concurrently while the shared
    # per-host throttle in music_tastes.http caps issuance at 1/s keeps us inside
    # their published limit and restores throughput.
    def work(song):
        try:
            return lookup_mbid(song)
        except Exception as exc:  # noqa: BLE001
            return {"song_id": song.song_id, "mbid": None, "error": str(exc)[:200]}

    mb_records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, s) for s in targets]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="musicbrainz", unit="song"):
            mb_records.append(fut.result())

    mb = pd.DataFrame(mb_records)
    matched = mb[mb["mbid"].notna()]
    print(f"\nMusicBrainz: {len(matched):,}/{len(mb):,} matched ({len(matched)/max(len(mb),1):.1%})")

    # Query every acceptable recording MBID, not just the best one.
    wanted: list[str] = []
    for cands in mb.get("candidate_mbids", pd.Series(dtype=object)).dropna():
        wanted.extend(cands)
    wanted = list(dict.fromkeys(wanted))
    print(f"  querying AcousticBrainz for {len(wanted):,} candidate recordings")

    features: dict[str, dict] = {}
    for i in tqdm(range(0, len(wanted), AB_BATCH), desc="acousticbrainz", unit="batch"):
        features.update(fetch_acousticbrainz(wanted[i : i + AB_BATCH]))

    # Attach the first candidate that actually has features to each song.
    rows = []
    for r in mb.itertuples():
        row = {"song_id": r.song_id, "mbid": r.mbid,
               "isrc": getattr(r, "isrc", None),
               "mb_title": getattr(r, "mb_title", None),
               "mb_artist": getattr(r, "mb_artist", None)}
        for cand in (getattr(r, "candidate_mbids", None) or []):
            if cand in features:
                feat = dict(features[cand])
                row["feature_mbid"] = feat.pop("mbid", None)
                row.update(feat)
                break
        rows.append(row)

    out = pd.DataFrame(rows)
    path = DERIVED / "acoustic_features.parquet"
    out.to_parquet(path, index=False)

    have_bpm = int(out["bpm"].notna().sum()) if "bpm" in out else 0
    print(f"AcousticBrainz: {have_bpm:,} songs with BPM "
          f"({have_bpm/max(len(out),1):.1%} of attempted)")
    print(f"  wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    run(limit=args.limit, workers=args.workers)
