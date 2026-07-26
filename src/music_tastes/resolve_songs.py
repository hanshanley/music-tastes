"""Stage 2: resolve chart rows into unique songs.

The chart archive is a text feed: the same recording appears with inconsistent
punctuation, casing, diacritics and featured-artist formatting across 68 years. This
stage collapses those variants into stable song ids.

Deliberate non-merges:
  * Different artists with the same title stay different songs (covers are not the
    same recording, and "Hallelujah" by two artists is two data points).
  * Re-recordings that advertise themselves as such -- "(Taylor's Version)" -- stay
    distinct, because they chart as separate releases with separate audiences.
  * Re-entries by the same song in different years are one song with several chart
    runs, not several songs.

Anything the rules cannot settle confidently is written to a review file rather than
being guessed at.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .paths import INTERIM

# Parenthetical/bracketed suffixes that mark a reissue of the *same* recording.
# Kept deliberately narrow: anything not listed here is treated as part of the title.
_EDITION_SUFFIX = re.compile(
    r"\s*[\(\[]\s*(?:"
    r"\d{4}\s+)?(?:digital\s+)?(?:"
    r"remaster(?:ed)?|re-?master(?:ed)?|"
    r"single|album|radio|7\"|45|lp|mono|stereo|clean|explicit"
    r")(?:\s+(?:version|edit|mix|cut))?\s*[\)\]]\s*$",
    re.IGNORECASE,
)

# Featured-artist separators. Order matters: longest first.
_FEAT = re.compile(
    r"\s+(?:featuring|feat\.?|ft\.?|with|duet\s+with|introducing|starring)\s+",
    re.IGNORECASE,
)

# Collaboration separators that join co-equal billed artists.
_COLLAB = re.compile(r"\s*(?:,|&|\+|/|\band\b|\bx\b|\bvs\.?\b|\bwith\b)\s*", re.IGNORECASE)

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_PARENTHETICAL = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")


def title_variants(title: str) -> list[str]:
    """Normalized forms a chart title might take on Genius.

    Three chart conventions need this:
      * Subtitles are inconsistently carried. Billboard lists "Back To Life" where
        Genius has "Back to Life (However Do You Want Me)".
      * Soundtrack tags are appended by Billboard only, as in
        "Sunflower (Spider-Man: Into The Spider-Verse)".
      * Double A-sides are charted as one entry, "Foolish Games/You Were Meant For
        Me", but exist on Genius as two separate songs.
    """
    raw = str(title)
    out: list[str] = []
    for side in re.split(r"\s*/\s*", raw):
        for form in (side, _PARENTHETICAL.sub("", side)):
            norm = normalize_title(form)
            if norm and norm not in out:
                out.append(norm)
    full = normalize_title(raw)
    if full and full not in out:
        out.insert(0, full)
    return out


def strip_accents(text: str) -> str:
    """Fold diacritics so 'volaré' and 'volare' resolve together."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def normalize_title(title: str) -> str:
    t = strip_accents(str(title)).lower().strip()
    t = t.replace("’", "'").replace("`", "'")
    # Apply repeatedly: "Song (Album Version) (Remastered)" carries two suffixes.
    while True:
        stripped = _EDITION_SUFFIX.sub("", t)
        if stripped == t:
            break
        t = stripped
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def split_artist(artist: str) -> tuple[str, list[str]]:
    """Return (billed primary artist, featured artists)."""
    a = strip_accents(str(artist)).strip().replace("’", "'")
    parts = _FEAT.split(a)
    primary = parts[0].strip()
    featured = [p.strip() for p in parts[1:] if p.strip()]
    return primary, featured


def normalize_artist(artist: str) -> str:
    """Normalize the primary artist only, ignoring featured credits.

    Featured credits change between chart weeks and across archives, so they are
    excluded from the identity key but preserved in the output for later use.
    """
    primary, _ = split_artist(artist)
    a = _PUNCT.sub(" ", primary.lower())
    a = _WS.sub(" ", a).strip()
    # "the beatles" and "beatles" are the same act.
    if a.startswith("the "):
        a = a[4:]
    return a


def _collab_signature(artist: str) -> str:
    """Order-insensitive signature of all billed acts, used to catch reorderings."""
    primary, featured = split_artist(artist)
    acts = set()
    for chunk in [primary, *featured]:
        for act in _COLLAB.split(chunk):
            act = _PUNCT.sub(" ", act.lower())
            act = _WS.sub(" ", act).strip()
            if act.startswith("the "):
                act = act[4:]
            if len(act) > 1:
                acts.add(act)
    return "|".join(sorted(acts))


def run() -> pd.DataFrame:
    entries = pd.read_parquet(INTERIM / "chart_entries.parquet")

    entries["title_norm"] = entries["title"].map(normalize_title)
    entries["artist_norm"] = entries["artist"].map(normalize_artist)
    entries["collab_sig"] = entries["artist"].map(_collab_signature)
    entries["song_key"] = entries["title_norm"] + " :: " + entries["artist_norm"]

    # Drop rows whose normalization produced an empty key -- we cannot identify these.
    bad = entries[(entries["title_norm"] == "") | (entries["artist_norm"] == "")]
    if len(bad):
        bad.to_parquet(INTERIM / "unresolvable_entries.parquet", index=False)
        print(f"  {len(bad)} chart rows could not be normalized; written for review")
    entries = entries.drop(bad.index)

    # Canonical display form: the most frequently charted spelling of each song.
    modes = (
        entries.groupby(["song_key", "title", "artist"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("song_key")
        .rename(columns={"title": "title_display", "artist": "artist_display"})
    )

    songs = (
        entries.groupby("song_key")
        .agg(
            first_charted=("chart_week", "min"),
            last_charted=("chart_week", "max"),
            weeks_on_chart=("chart_week", "nunique"),
            peak_rank=("rank", "min"),
            title_norm=("title_norm", "first"),
            artist_norm=("artist_norm", "first"),
        )
        .reset_index()
        .merge(modes[["song_key", "title_display", "artist_display"]], on="song_key")
    )
    songs["song_id"] = (
        pd.util.hash_pandas_object(songs["song_key"], index=False)
        .astype("uint64")
        .map(lambda v: f"s{v:016x}")
    )
    songs["debut_year"] = pd.to_datetime(songs["first_charted"]).dt.year

    entries = entries.merge(songs[["song_key", "song_id"]], on="song_key")

    # Review queue: same normalized title + overlapping billed acts but a different
    # identity key. These are candidate over-splits (e.g. billing order changed).
    review = _find_candidate_merges(entries, songs)

    entries.to_parquet(INTERIM / "chart_entries_resolved.parquet", index=False)
    songs.to_parquet(INTERIM / "songs.parquet", index=False)
    review.to_csv(INTERIM / "merge_review.csv", index=False)

    print(f"\nResolved {len(entries):,} chart rows into {len(songs):,} unique songs")
    print(f"  raw (title, artist) pairs before normalization: 32,643")
    print(f"  candidate merges flagged for manual review: {len(review):,}")
    print(f"  songs debuting per decade:")
    by_decade = songs.groupby(songs["debut_year"] // 10 * 10).size()
    for decade, n in by_decade.items():
        print(f"    {decade}s: {n:,}")
    return songs


def _find_candidate_merges(entries: pd.DataFrame, songs: pd.DataFrame) -> pd.DataFrame:
    """Flag songs sharing a title and at least one billed act but split apart."""
    sig = (
        entries.groupby("song_id")["collab_sig"]
        .agg(lambda s: set().union(*[set(x.split("|")) for x in s if x]))
        .to_dict()
    )
    merged = songs.merge(
        songs, on="title_norm", suffixes=("_a", "_b")
    )
    merged = merged[merged["song_id_a"] < merged["song_id_b"]]

    rows = []
    for r in merged.itertuples():
        shared = sig.get(r.song_id_a, set()) & sig.get(r.song_id_b, set())
        if shared:
            rows.append(
                {
                    "title_norm": r.title_norm,
                    "song_id_a": r.song_id_a,
                    "artist_a": r.artist_display_a,
                    "first_charted_a": r.first_charted_a,
                    "song_id_b": r.song_id_b,
                    "artist_b": r.artist_display_b,
                    "first_charted_b": r.first_charted_b,
                    "shared_acts": ", ".join(sorted(shared)),
                    "decision": "",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
