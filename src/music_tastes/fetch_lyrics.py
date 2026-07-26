"""Stage 4: fetch lyrics into a local, gitignored cache.

Copyright posture
-----------------
Lyrics are copyrighted. This module writes raw lyric text **only** to
``data/cache/lyrics_cache/``, which .gitignore excludes. Nothing downstream of this
stage emits lyric text: the analysis stages read the cache, compute numeric features,
and publish only aggregates. Do not commit the cache or copy lyric text into reports.

Method
------
Search is done through the official Genius API (which requires a token and is not
covered by the web robots.txt Disallow on /search). The lyric body is then read from
the public song page, which robots.txt permits for generic user agents. Requests are
rate limited in :mod:`music_tastes.http` and every response is cached on disk, so a
re-run costs no network traffic.

Matching is conservative: a Genius hit is accepted only if the normalized title and
artist both clear a similarity threshold. Anything below it is recorded as unmatched
rather than being attached to the wrong song.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from .http import get
from .paths import DERIVED, LYRICS_CACHE, env
from .resolve_songs import normalize_artist, normalize_title, split_artist

GENIUS_SEARCH = "https://api.genius.com/search"

TITLE_THRESHOLD = 0.82
ARTIST_THRESHOLD = 0.72

# Section headers and production credits Genius embeds in the lyric body.
_SECTION = re.compile(r"\[[^\]]{0,80}\]")
_CONTRIBUTOR_HEADER = re.compile(
    r"^.*?(?:\d+\s+Contributors?|Read More\s*)", re.IGNORECASE | re.DOTALL
)
_TRANSLATION_LINE = re.compile(r"^\s*\d*\s*Translations?.*$", re.IGNORECASE | re.MULTILINE)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _cache_file(song_id: str):
    return LYRICS_CACHE / f"{song_id}.json"


def search_genius(title: str, artist: str, token: str) -> list[dict]:
    primary, _ = split_artist(artist)
    resp = get(
        GENIUS_SEARCH,
        namespace="genius_search",
        params={"q": f"{title} {primary}"},
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status != 200:
        return []
    return [h["result"] for h in resp.json().get("response", {}).get("hits", [])]


def pick_match(hits: list[dict], title: str, artist: str) -> tuple[dict | None, float, float]:
    """Choose the best Genius hit, or None if nothing clears both thresholds."""
    want_title = normalize_title(title)
    want_artist = normalize_artist(artist)

    best, best_score = None, (0.0, 0.0)
    for hit in hits:
        got_title = normalize_title(hit.get("title") or "")
        got_artist = normalize_artist(hit.get("primary_artist", {}).get("name") or "")
        ts, as_ = _sim(want_title, got_title), _sim(want_artist, got_artist)
        # Featured artists often occupy the primary slot on one side or the other.
        full = hit.get("artist_names") or ""
        as_ = max(as_, _sim(want_artist, normalize_artist(full)))
        if ts >= TITLE_THRESHOLD and as_ >= ARTIST_THRESHOLD:
            if ts + as_ > sum(best_score):
                best, best_score = hit, (ts, as_)
    return best, best_score[0], best_score[1]


def scrape_lyrics(url: str) -> str | None:
    resp = get(url, namespace="genius_page", headers={"Accept": "text/html"})
    if resp.status != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")

    containers = soup.select("div[data-lyrics-container='true']")
    if not containers:
        legacy = soup.select_one("div.lyrics")
        if not legacy:
            return None
        containers = [legacy]

    parts = []
    for c in containers:
        # Genius marks non-lyric annotations with this attribute; drop them.
        for junk in c.select("[data-exclude-from-selection='true']"):
            junk.decompose()
        parts.append(c.get_text(separator="\n"))

    text = "\n".join(parts)
    text = _TRANSLATION_LINE.sub("", text)
    text = _CONTRIBUTOR_HEADER.sub("", text, count=1)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def looks_instrumental(text: str) -> bool:
    stripped = _SECTION.sub("", text).strip()
    return len(stripped) < 60 or "instrumental" in text.lower()[:200]


def is_probably_english(text: str) -> bool:
    """Cheap ASCII-ratio + stopword heuristic; recorded, never used to silently drop."""
    body = _SECTION.sub(" ", text).lower()
    words = re.findall(r"[a-z']+", body)
    if len(words) < 20:
        return False
    stop = {"the", "you", "and", "to", "a", "i", "me", "my", "it", "in", "of", "that"}
    hits = sum(1 for w in words if w in stop)
    ascii_ratio = sum(1 for c in body if ord(c) < 128) / max(len(body), 1)
    return (hits / len(words)) > 0.06 and ascii_ratio > 0.85


def fetch_one(song, token: str) -> dict:
    cached = _cache_file(song.song_id)
    if cached.exists():
        rec = json.loads(cached.read_text())
        return {k: v for k, v in rec.items() if k != "lyrics"}

    hits = search_genius(song.title_display, song.artist_display, token)
    match, ts, as_ = pick_match(hits, song.title_display, song.artist_display)

    rec = {
        "song_id": song.song_id,
        "matched": match is not None,
        "genius_id": match.get("id") if match else None,
        "genius_url": match.get("url") if match else None,
        "genius_title": match.get("title") if match else None,
        "genius_artist": match.get("artist_names") if match else None,
        "title_similarity": round(ts, 3),
        "artist_similarity": round(as_, 3),
        "n_hits": len(hits),
        "has_lyrics": False,
        "n_chars": 0,
        "n_words": 0,
        "is_instrumental": False,
        "is_english": False,
        "source": "genius",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    if match:
        text = scrape_lyrics(match["url"])
        if text:
            rec["has_lyrics"] = True
            rec["n_chars"] = len(text)
            rec["n_words"] = len(re.findall(r"[\w']+", _SECTION.sub(" ", text)))
            rec["is_instrumental"] = looks_instrumental(text)
            rec["is_english"] = is_probably_english(text)
            cached.write_text(json.dumps({**rec, "lyrics": text}))
            return rec

    cached.write_text(json.dumps({**rec, "lyrics": None}))
    return rec


def load_lyrics(song_id: str) -> str | None:
    """Read cached lyric text. Analysis-stage helper; never returns to a report."""
    path = _cache_file(song_id)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("lyrics")


def run(limit: int | None = None, sample: str = "top") -> pd.DataFrame:
    token = env("GENIUS_ACCESS_TOKEN", required=True)
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")

    if limit:
        if sample == "top":
            songs = songs.nlargest(limit, "points")
        else:
            # Stratified across debut years, so a smoke test spans the whole period.
            songs = (
                songs.groupby(songs["debut_year"] // 5 * 5, group_keys=False)
                .apply(lambda g: g.nlargest(max(1, limit // 14), "points"))
                .head(limit)
            )

    records = []
    for song in tqdm(list(songs.itertuples()), desc="lyrics", unit="song"):
        try:
            records.append(fetch_one(song, token))
        except Exception as exc:  # noqa: BLE001 - one bad song must not kill a long run
            records.append(
                {"song_id": song.song_id, "matched": False, "error": str(exc)[:200]}
            )

    df = pd.DataFrame(records)
    out = DERIVED / "lyrics_index.parquet"
    if out.exists() and limit:
        prior = pd.read_parquet(out)
        df = pd.concat([prior[~prior["song_id"].isin(df["song_id"])], df])
    df.to_parquet(out, index=False)

    n = len(df)
    print(f"\nProcessed {n:,} songs")
    print(f"  matched on Genius:  {df['matched'].sum():,} ({df['matched'].mean():.1%})")
    if "has_lyrics" in df:
        print(f"  lyrics retrieved:   {df['has_lyrics'].sum():,} "
              f"({df['has_lyrics'].mean():.1%})")
        got = df[df["has_lyrics"].fillna(False)]
        if len(got):
            print(f"  instrumental:       {got['is_instrumental'].sum():,}")
            print(f"  English (heuristic):{got['is_english'].sum():,}")
            print(f"  median word count:  {got['n_words'].median():.0f}")
    print(f"  wrote {out}  (lyric text stays in {LYRICS_CACHE}, gitignored)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", choices=["top", "stratified"], default="top")
    args = ap.parse_args()
    run(limit=args.limit, sample=args.sample)
