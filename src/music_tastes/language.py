"""Language identification and code-switching measurement for lyrics.

Why this matters to the result
------------------------------
Every lexicon-based sentiment measure in this project is English-only. NRC VAD does
not contain Spanish or Korean words, so a non-English lyric does not score "neutral" --
it scores whatever its handful of accidentally-English-looking tokens happen to say,
over a tiny matched fraction. That is noise presented as a measurement.

This is not a hypothetical problem for a chart trend. The non-English share of Hot 100
hits with lyrics rises from roughly 0% before 2000 to 5.3% in the 2020s (Bad Bunny,
BTS, Karol G, BLACKPINK). It rises over exactly the period in which lyric valence
falls, so it has to be ruled out rather than assumed away.

Two distinct problems, handled separately
-----------------------------------------
1. **Wholly non-English songs.** Detected and excluded. The original filter was an
   ad-hoc stopword-and-ASCII heuristic; this module replaces it with a real language
   identifier and *measures* the heuristic's error rate rather than trusting it.

2. **Code-switching.** A song can be predominantly English yet contain a Spanish verse
   or Korean chorus, pass any language filter, and still drag lexicon coverage down.
   This is the more insidious case because it is invisible to language ID at the song
   level. ``english_token_share`` measures it directly, using word-frequency lists to
   ask what fraction of a lyric's tokens are ordinary English words.

Falling lexicon coverage (0.350 in the 1950s to 0.318 in the 2020s) could be caused by
code-switching, by slang and ad-libs, or by proper nouns. Separating those is what
``coverage_decomposition`` is for.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from .fetch_lyrics import load_lyrics
from .lyrics_features import clean_lyrics, tokenize
from .paths import DERIVED, REPORTS, require

# A token counts as English if the English frequency list knows it at all. The
# threshold is deliberately permissive: we are separating "is this an English word"
# from "is this a rare word", and rare English words are still English.
ENGLISH_MIN_FREQ = 1e-8
MIN_TOKENS_FOR_ID = 20


def _detector():
    from langdetect import DetectorFactory, detect_langs

    # langdetect is randomized; fix the seed so runs are reproducible.
    DetectorFactory.seed = 0
    return detect_langs


def detect_language(text: str) -> tuple[str | None, float]:
    """Return (language code, confidence) for a lyric."""
    detect_langs = _detector()
    try:
        results = detect_langs(text[:4000])
    except Exception:  # noqa: BLE001 - langdetect raises on degenerate input
        return None, 0.0
    if not results:
        return None, 0.0
    best = results[0]
    return best.lang, float(best.prob)


def english_token_share(tokens: list[str]) -> float:
    """Fraction of tokens that are ordinary English words.

    This is the code-switching measure. A monolingual English lyric sits near 1.0; a
    lyric with a Spanish verse drops in proportion to that verse's length, even though
    song-level language ID still says "English".
    """
    from wordfreq import zipf_frequency

    if not tokens:
        return 0.0
    english = sum(1 for t in tokens if zipf_frequency(t, "en") > 0)
    return english / len(tokens)


def analyse(limit: int | None = None) -> pd.DataFrame:
    """Language-profile every song that has lyrics."""
    index = pd.read_parquet(require(DERIVED / "lyrics_index.parquet"))
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")[
        ["song_id", "debut_year", "title_display", "artist_display", "points"]
    ]
    df = index[index["has_lyrics"].fillna(False)].merge(songs, on="song_id")
    if limit:
        df = df.head(limit)

    rows = []
    for r in tqdm(list(df.itertuples()), desc="language", unit="song"):
        raw = load_lyrics(r.song_id)
        if not raw:
            continue
        text = clean_lyrics(raw)
        tokens = tokenize(text)
        if len(tokens) < MIN_TOKENS_FOR_ID:
            continue
        lang, conf = detect_language(text)
        rows.append(
            {
                "song_id": r.song_id,
                "debut_year": r.debut_year,
                "title_display": r.title_display,
                "artist_display": r.artist_display,
                "points": r.points,
                "lang": lang,
                "lang_confidence": conf,
                "english_token_share": english_token_share(tokens),
                "n_tokens": len(tokens),
                "heuristic_is_english": bool(getattr(r, "is_english", False)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_parquet(DERIVED / "language_profile.parquet", index=False)
    return out


def evaluate_heuristic(prof: pd.DataFrame) -> dict:
    """Score the original ASCII/stopword heuristic against real language ID."""
    truth = prof["lang"] == "en"
    heur = prof["heuristic_is_english"]
    tp = int((truth & heur).sum())
    fp = int((~truth & heur).sum())
    fn = int((truth & ~heur).sum())
    tn = int((~truth & ~heur).sum())
    return {
        "n": int(len(prof)),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "heuristic_precision": tp / (tp + fp) if tp + fp else None,
        "heuristic_recall": tp / (tp + fn) if tp + fn else None,
        "non_english_leaking_through": fp,
    }


def coverage_decomposition(prof: pd.DataFrame) -> pd.DataFrame:
    """Is falling lexicon coverage explained by code-switching?

    If it were, English-token share would fall in step with lexicon coverage. If
    English-token share is flat while coverage falls, the cause is something else --
    slang, ad-libs or proper nouns entering English lyrics.
    """
    d = prof[prof["lang"] == "en"].copy()
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    return (
        d.groupby("decade")
        .agg(
            n=("song_id", "size"),
            english_token_share=("english_token_share", "mean"),
            min_share=("english_token_share", lambda s: s.quantile(0.05)),
        )
        .reset_index()
    )


def run(limit: int | None = None) -> dict:
    prof = analyse(limit=limit)
    if prof.empty:
        raise SystemExit("no songs profiled")

    prof["decade"] = (prof["debut_year"] // 10 * 10).astype(int)
    by_decade = (
        prof.assign(non_english=(prof["lang"] != "en"))
        .groupby("decade")
        .agg(n=("song_id", "size"), pct_non_english=("non_english", "mean"))
        .reset_index()
    )

    heur = evaluate_heuristic(prof)
    decomp = coverage_decomposition(prof)

    by_decade.to_csv(REPORTS / "language_by_decade.csv", index=False)
    decomp.to_csv(REPORTS / "code_switching_by_decade.csv", index=False)

    results = {
        "by_decade": by_decade.round(4).to_dict("records"),
        "heuristic_evaluation": heur,
        "code_switching_by_decade": decomp.round(4).to_dict("records"),
        "language_counts": prof["lang"].value_counts().head(12).to_dict(),
    }
    (REPORTS / "language.json").write_text(json.dumps(results, indent=2, default=str))

    print("\n=== Detected language mix by decade ===")
    for r in by_decade.itertuples():
        print(f"  {int(r.decade)}s  n={r.n:5}  non-English {r.pct_non_english:6.1%}")

    print("\n=== Top languages ===")
    for lang, n in list(results["language_counts"].items())[:8]:
        print(f"  {lang}: {n:,}")

    print("\n=== Original heuristic vs real language ID ===")
    print(f"  precision={heur['heuristic_precision']:.3f} "
          f"recall={heur['heuristic_recall']:.3f}")
    print(f"  non-English songs that leaked through the old filter: "
          f"{heur['non_english_leaking_through']:,}")

    print("\n=== Code-switching within English-detected songs ===")
    print("  (if this falls in step with lexicon coverage, code-switching is the cause)")
    for r in decomp.itertuples():
        print(f"  {int(r.decade)}s  n={r.n:5}  English-token share "
              f"mean={r.english_token_share:.3f}  5th pct={r.min_share:.3f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(limit=args.limit)
