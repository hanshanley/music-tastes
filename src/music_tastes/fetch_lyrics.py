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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from .http import circuit_open, get
from .paths import DERIVED, LYRICS_CACHE, env
from .resolve_songs import normalize_artist, normalize_title, title_variants

GENIUS_SEARCH = "https://api.genius.com/search"

# Matching is tiered rather than a single threshold, because two failure modes need
# different treatment:
#   * Artist renames ("Lady Antebellum" -> "Lady A", "Kanye West" -> "Ye"). Genius
#     shows the current name; the chart preserves the historical billing. The title
#     matches perfectly while the artist string barely matches at all.
#   * Genuine covers, where the title matches perfectly and the artist should NOT.
# Tier A is safe on its own. Tiers B and C trade precision for recall and are recorded
# separately so the coverage audit can re-run every headline result without them.
TIER_A = {"title": 0.85, "artist": 0.72}
TIER_B = {"title": 0.92, "artist": 0.50}
TIER_C = {"title": 0.96, "artist": 0.30, "max_rank": 2}

# Genius hosts translation and editorial pages alongside songs. These are never the
# recording we want. Anchored so real acts such as "GZA/Genius" are not caught.
_JUNK_ARTIST = re.compile(
    r"^(?:genius(?:\s|$)|pop genius|spotify$|billboard$|apple music$|"
    r"amazon music$|tidal$|charts?$|rock genius|rap genius)",
    re.IGNORECASE,
)
_JUNK_TITLE = re.compile(
    r"traducci[oó]n|tradu[çc][aã]o|traduction|übersetzung|перевод|"
    r"tracklist|playlist|\[top\s*\d+\]|annotated|credits$",
    re.IGNORECASE,
)

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


def _artist_score(want: str, hit: dict) -> float:
    """Best artist similarity across the hit's primary and full billing strings.

    Combines sequence ratio with token containment. Token containment is what rescues
    abbreviated renames such as "Lady A" for "Lady Antebellum", where the sequence
    ratio alone falls below any usable threshold.
    """
    candidates = [
        normalize_artist(hit.get("primary_artist", {}).get("name") or ""),
        normalize_artist(hit.get("artist_names") or ""),
    ]
    want_tokens = set(want.split())
    best = 0.0
    for cand in candidates:
        if not cand:
            continue
        score = _sim(want, cand)
        cand_tokens = set(cand.split())
        # Token overlap rescues abbreviated renames ("Lady A" for "Lady Antebellum"),
        # but it must not be a *containment* measure. Dividing by min() returns 1.0
        # whenever either name is a subset of the other, which let "B.o.B" match an
        # unrelated act whose tokens happened to contain b/o/b at tier A. Jaccard
        # penalises the extra tokens, and short initialisms are excluded entirely
        # because one- and two-letter tokens carry almost no identifying signal.
        if want_tokens and cand_tokens:
            informative = {w for w in want_tokens if len(w) > 2}
            if informative:
                union = want_tokens | cand_tokens
                jaccard = len(want_tokens & cand_tokens) / len(union)
                score = max(score, jaccard)
        best = max(best, score)
    return best


def _title_score(want_variants: list[str], hit_title: str) -> float:
    """Best similarity between any chart-title variant and any hit-title variant."""
    hit_forms = title_variants(hit_title)
    return max((_sim(w, h) for w in want_variants for h in hit_forms), default=0.0)


def search_genius(title: str, artist: str, token: str, use_api: bool = True) -> list[dict]:
    """Search Genius, trying several query forms and pooling the results.

    Genius's search is sensitive to both leading articles and appended subtitles: the
    query "I Gotta Feeling The Black Eyed Peas" returns only editorial pages, and
    "Sunflower (Spider-Man: Into The Spider-Verse)" fails to surface "Sunflower". We
    therefore query with the normalized artist and the stripped-down title first, and
    widen only if that fails.

    Returns an empty list rather than raising when the API is quota-blocked, so the
    caller can fall back to the slug route.
    """
    if not use_api or not token or circuit_open("api.genius.com"):
        return []

    norm_artist = normalize_artist(artist)
    variants = title_variants(title)
    base = variants[-1] if variants else str(title)

    queries = [
        f"{base} {norm_artist}".strip(),
        f"{normalize_title(title)} {norm_artist}".strip(),
        f"{norm_artist} {base}".strip(),
        base.strip(),
    ]

    seen: dict[int, dict] = {}
    for query in dict.fromkeys(q for q in queries if q):
        try:
            resp = get(
                GENIUS_SEARCH,
                namespace="genius_search",
                params={"q": query},
                headers={"Authorization": f"Bearer {token}"},
            )
        except RuntimeError:
            break
        if resp.status != 200:
            if resp.status == 429:
                break
            continue
        hits = [h["result"] for h in resp.json().get("response", {}).get("hits", [])]
        for rank, hit in enumerate(hits):
            if hit["id"] not in seen:
                hit["_rank"] = rank
                seen[hit["id"]] = hit
        # A clean tier-A hit means the remaining query forms cannot improve on it.
        _, _, _, tier = pick_match(list(seen.values()), title, artist)
        if tier == "A":
            break
    return list(seen.values())


def pick_match(
    hits: list[dict], title: str, artist: str
) -> tuple[dict | None, float, float, str | None]:
    """Choose the best Genius hit and the confidence tier it qualified under.

    Returns the best *observed* similarities even when nothing qualifies, so that a
    failure is diagnosable rather than reported as a flat zero.
    """
    want_variants = title_variants(title)
    want_artist = normalize_artist(artist)

    best: dict | None = None
    best_tier: str | None = None
    best_key = -1.0
    seen_title, seen_artist = 0.0, 0.0

    for hit in hits:
        hit_title_raw = hit.get("title") or ""
        hit_artist_raw = hit.get("artist_names") or ""
        if _JUNK_ARTIST.search(hit_artist_raw) or _JUNK_TITLE.search(hit_title_raw):
            continue
        if hit.get("lyrics_state") not in (None, "complete"):
            continue

        ts = _title_score(want_variants, hit_title_raw)
        as_ = _artist_score(want_artist, hit)
        seen_title, seen_artist = max(seen_title, ts), max(seen_artist, as_)

        if ts >= TIER_A["title"] and as_ >= TIER_A["artist"]:
            tier = "A"
        elif ts >= TIER_B["title"] and as_ >= TIER_B["artist"]:
            tier = "B"
        elif (
            ts >= TIER_C["title"]
            and as_ >= TIER_C["artist"]
            and hit.get("_rank", 99) <= TIER_C["max_rank"]
        ):
            tier = "C"
        else:
            continue

        # Prefer the safest tier; break ties on combined similarity.
        key = {"A": 2.0, "B": 1.0, "C": 0.0}[tier] * 10 + ts + as_
        if key > best_key:
            best, best_tier, best_key = hit, tier, key

    if best is None:
        return None, round(seen_title, 3), round(seen_artist, 3), None
    return (
        best,
        round(_title_score(want_variants, best.get("title") or ""), 3),
        round(_artist_score(want_artist, best), 3),
        best_tier,
    )


def scrape_lyrics(url: str, html: str | None = None) -> str | None:
    """Extract the lyric body from a Genius song page.

    ``html`` may be supplied to avoid re-fetching a page we already hold.
    """
    if html is None:
        resp = get(url, namespace="genius_page", headers={"Accept": "text/html"})
        if resp.status != 200:
            return None
        html = resp.text
    soup = BeautifulSoup(html, "lxml")

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


def _slugify(text: str, capitalize_first: bool) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", "and").replace("'", "").replace("\u2019", "")
    s = re.sub(r"[^A-Za-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    parts = s.split(" ")
    if capitalize_first:
        return "-".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])
    return "-".join(p.lower() for p in parts)


def _raw_title_variants(title: str) -> list[str]:
    """Title variants that preserve punctuation, for URL construction.

    :func:`~music_tastes.resolve_songs.title_variants` normalises punctuation to
    spaces, which is right for similarity scoring but wrong for building a slug:
    "That's What I Like" becomes "that s what i like" and then slugifies to
    "that-s-what-i-like", where Genius uses "thats-what-i-like". Feeding normalised
    titles into the slug builder silently broke every title containing an
    apostrophe. These variants apply the same splitting and subtitle-stripping but
    leave the characters alone so :func:`_slugify` can drop apostrophes itself.
    """
    raw = str(title).strip()
    out: list[str] = [raw]
    for side in re.split(r"\s*/\s*", raw):
        side = side.strip()
        for form in (side, _PAREN.sub("", side).strip()):
            if form and form not in out:
                out.append(form)
    stripped = _PAREN.sub("", raw).strip()
    if stripped and stripped not in out:
        out.append(stripped)
    return out


def slug_candidates(title: str, artist: str) -> list[str]:
    """Genius song-page URLs guessed directly from artist and title.

    Genius URLs are deterministic (`/Artist-title-lyrics`), so the page can be
    reached without the search API. This matters because the API enforces a quota
    that a full 32k-song run exhausts, after which every search returns 429 while
    song pages continue to serve normally.

    Guessing is only safe because the result is verified against the page's own
    <title> before being accepted; see :func:`verify_page`.
    """
    primary_raw = re.split(_FEAT_SPLIT, str(artist))[0].strip()
    artists = [a for a in (str(artist).strip(), primary_raw) if a]

    urls: list[str] = []
    for t in _raw_title_variants(title)[:4]:
        for a in artists:
            slug_a, slug_t = _slugify(a, True), _slugify(t, False)
            if not slug_a or not slug_t:
                continue
            url = f"https://genius.com/{slug_a}-{slug_t}-lyrics"
            if url not in urls:
                urls.append(url)
    return urls


_PAREN = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
_FEAT_SPLIT = r"(?i)\s+(?:featuring|feat\.?|ft\.?|with|duet with|introducing)\s+"

_TITLE_TAG = re.compile(r"<title>([^<]*)</title>", re.IGNORECASE)


def verify_page(html: str, title: str, artist: str) -> tuple[float, float, str, str] | None:
    """Check a guessed page really is the song we wanted.

    The page <title> is formatted "Artist – Title Lyrics | Genius Lyrics", which is
    enough to apply the same similarity thresholds used for API matches.
    """
    m = _TITLE_TAG.search(html)
    if not m:
        return None
    raw = m.group(1)
    raw = re.sub(r"\s*\|\s*Genius.*$", "", raw)
    raw = re.sub(r"\s*Lyrics\s*$", "", raw, flags=re.IGNORECASE)
    parts = re.split(r"\s+[\u2013\u2014-]\s+", raw, maxsplit=1)
    if len(parts) != 2:
        return None
    got_artist, got_title = parts[0].strip(), parts[1].strip()

    ts = _title_score(title_variants(title), got_title)
    fake_hit = {"primary_artist": {"name": got_artist}, "artist_names": got_artist}
    as_ = _artist_score(normalize_artist(artist), fake_hit)
    return ts, as_, got_title, got_artist


def match_via_slug(title: str, artist: str) -> dict | None:
    """API-free fallback: guess the URL, fetch it, and verify before accepting."""
    for url in slug_candidates(title, artist):
        resp = get(url, namespace="genius_page", headers={"Accept": "text/html"})
        if resp.status != 200:
            continue
        checked = verify_page(resp.text, title, artist)
        if not checked:
            continue
        ts, as_, got_title, got_artist = checked
        if ts >= TIER_A["title"] and as_ >= TIER_A["artist"]:
            tier = "A-slug"
        elif ts >= TIER_B["title"] and as_ >= TIER_B["artist"]:
            tier = "B-slug"
        else:
            continue
        return {
            "url": url,
            "title": got_title,
            "artist_names": got_artist,
            "html": resp.text,
            "title_similarity": round(ts, 3),
            "artist_similarity": round(as_, 3),
            "tier": tier,
        }
    return None


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


def fetch_one(song, token: str, use_api: bool = True) -> dict:
    cached = _cache_file(song.song_id)
    if cached.exists():
        rec = json.loads(cached.read_text())
        return {k: v for k, v in rec.items() if k != "lyrics"}

    try:
        hits = search_genius(song.title_display, song.artist_display, token, use_api)
    except RuntimeError:
        # Search quota exhausted or the API is unreachable; the slug route below
        # does not touch the API at all, so keep going rather than losing the song.
        hits = []
    match, ts, as_, tier = pick_match(hits, song.title_display, song.artist_display)

    rec = {
        "song_id": song.song_id,
        "matched": match is not None,
        "match_tier": tier,
        "genius_id": match.get("id") if match else None,
        "genius_url": match.get("url") if match else None,
        "genius_title": match.get("title") if match else None,
        "genius_artist": match.get("artist_names") if match else None,
        "title_similarity": ts,
        "artist_similarity": as_,
        "n_hits": len(hits),
        "has_lyrics": False,
        # Missing lyrics must be NaN, not 0. Zero is a *valid* word count, so a
        # sentinel of 0 survives dropna() and enters means as a real observation.
        # Because lyric coverage is strongly year-dependent, those false zeros
        # concentrate in the early decades and manufacture a "lyrics got longer"
        # slope out of missing data.
        "n_chars": float("nan"),
        "n_words": float("nan"),
        "is_instrumental": False,
        "is_english": False,
        "source": "genius",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }

    text = None
    if match:
        text = scrape_lyrics(match["url"])

    if text is None:
        slug = match_via_slug(song.title_display, song.artist_display)
        if slug:
            text = scrape_lyrics(slug["url"], html=slug["html"])
            if text:
                rec.update(
                    matched=True,
                    match_tier=slug["tier"],
                    genius_url=slug["url"],
                    genius_title=slug["title"],
                    genius_artist=slug["artist_names"],
                    title_similarity=slug["title_similarity"],
                    artist_similarity=slug["artist_similarity"],
                    source="genius-slug",
                )

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


def rebuild_index_from_cache() -> pd.DataFrame:
    """Reconstruct the lyrics index from the on-disk cache.

    The main run writes its index only when it finishes, which for a full 32k fetch is
    hours away. The cache is written per song as it completes, so this lets analysis
    stages work against whatever has landed so far, and doubles as crash recovery.
    """
    records = []
    for path in LYRICS_CACHE.glob("*.json"):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        records.append({k: v for k, v in rec.items() if k != "lyrics"})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    out = DERIVED / "lyrics_index.parquet"
    df.to_parquet(out, index=False)
    print(f"Rebuilt index for {len(df):,} songs from cache -> {out}")
    return df


def run(
    limit: int | None = None,
    sample: str = "top",
    workers: int = 4,
    use_api: bool = True,
) -> pd.DataFrame:
    token = env("GENIUS_ACCESS_TOKEN", required=True) if use_api else ""
    if not use_api:
        print("API search disabled; using verified slug URLs only.")
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

    targets = list(songs.itertuples())

    def work(song):
        try:
            return fetch_one(song, token, use_api)
        except Exception as exc:  # noqa: BLE001 - one bad song must not kill a long run
            return {
                "song_id": song.song_id,
                "matched": False,
                "has_lyrics": False,
                "error": str(exc)[:200],
            }

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, s) for s in targets]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="lyrics", unit="song"):
            records.append(fut.result())

    df = pd.DataFrame(records)
    out = DERIVED / "lyrics_index.parquet"
    if out.exists():
        prior = pd.read_parquet(out)
        df = pd.concat([prior[~prior["song_id"].isin(df["song_id"])], df], ignore_index=True)
    df.to_parquet(out, index=False)

    n = len(df)
    print(f"\nLyrics index now covers {n:,} songs")
    print(f"  matched on Genius:  {df['matched'].sum():,} ({df['matched'].mean():.1%})")
    if "match_tier" in df:
        for tier, cnt in df["match_tier"].value_counts().items():
            print(f"    tier {tier}: {cnt:,}")
    if "has_lyrics" in df:
        has = df["has_lyrics"].fillna(False)
        print(f"  lyrics retrieved:   {has.sum():,} ({has.mean():.1%})")
        got = df[has]
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-api", action="store_true",
                    help="Skip the search API entirely and use verified slug URLs. "
                         "Use when the API quota is exhausted (HTTP 429).")
    args = ap.parse_args()
    run(limit=args.limit, sample=args.sample, workers=args.workers,
        use_api=not args.no_api)
