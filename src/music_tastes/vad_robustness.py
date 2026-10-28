"""Versioned NRC VAD sensitivity analysis; no new model inference or lyric exports."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from . import analysis_trends as trends
from .cache import cache_matches, fingerprint, mark_cache
from .coverage import load_joined
from .fetch_lyrics import load_lyrics
from .lexicons import LEX_DIR, load_vad_v21
from .lyrics_features import _SECTION, _WORD, clean_lyrics, tokenize
from .paths import DERIVED, INTERIM, LYRICS_CACHE, REPORTS, atomic_json, atomic_parquet

DIMENSIONS = ("valence", "arousal", "dominance")
VARIANTS = (
    "v1_tokens", "v1_shared_tokens", "v21_shared_tokens",
    "v21_tokens", "v21_types", "v21_phrases",
    "v21_polar_0167", "v21_polar_0333",
)
BOUNDARIES = re.compile(r"[\r\n.!?;,:()\[\]{}]+")


def make_lookups(v1: pd.DataFrame, v21: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    baseline = dict(zip(v1["word"], v1[list(DIMENSIONS)].to_numpy(dtype=float)))
    words, trie = {}, {}
    for row in v21.itertuples(index=False):
        term = row.term
        parts = tokenize(term)
        # Do not silently reinterpret hyphenated, accented or numeric entries.
        if not parts or " ".join(parts) != term:
            continue
        score = (np.array([getattr(row, d) for d in DIMENSIONS]) + 1) / 2
        if len(parts) == 1:
            words[term] = score
        node = trie
        for part in parts:
            node = node.setdefault(part, {})
        node[""] = score
    shared = {word: score for word, score in words.items() if word in baseline}
    return baseline, words, shared, trie


def phrase_units(text: str, trie: dict) -> list[tuple[np.ndarray, int]]:
    """Longest nonoverlapping terms; never cross lines, sections or strong punctuation."""
    units = []
    for segment in BOUNDARIES.split(_SECTION.sub("\n", text)):
        tokens = tokenize(segment)
        i = 0
        while i < len(tokens):
            node, j, match = trie, i, None
            while j < len(tokens) and tokens[j] in node:
                node = node[tokens[j]]
                j += 1
                if "" in node:
                    match = (node[""], j - i)
            if match is None:
                i += 1
            else:
                units.append(match)
                i += match[1]
    return units


def score_lyrics(text: str, baseline: dict, words: dict, shared: dict, trie: dict) -> dict:
    tokens = tokenize(clean_lyrics(text))
    counts = Counter(tokens)
    n = len(tokens)
    out = {"n_tokens": n}
    for variant, lexicon in (("v1_tokens", baseline), ("v1_shared_tokens", baseline),
                             ("v21_shared_tokens", shared), ("v21_tokens", words),
                             ("v21_types", words), ("v21_polar_0167", words),
                             ("v21_polar_0333", words)):
        threshold = {"v21_polar_0167": .167, "v21_polar_0333": .333}.get(variant)
        hits = [
            (lexicon[word], count) for word, count in counts.items()
            if word in lexicon
            and (variant != "v1_shared_tokens" or word in shared)
            and (threshold is None or abs(2 * lexicon[word][0] - 1) >= threshold - 1e-12)
        ]
        covered = sum(count for _, count in hits)
        out[f"{variant}_coverage"] = covered / n if n else np.nan
        out[f"{variant}_matched_tokens"] = covered
        weights = [1 if variant == "v21_types" else count for _, count in hits]
        mean = np.average([score for score, _ in hits], axis=0, weights=weights) if hits else [np.nan] * 3
        for dimension, value in zip(DIMENSIONS, mean):
            if threshold is None or dimension == "valence":
                out[f"{variant}_{dimension}"] = float(value)
    units = phrase_units(text, trie)
    covered = sum(length for _, length in units)
    out["v21_phrases_coverage"] = covered / n if n else np.nan
    out["v21_phrases_matched_tokens"] = covered
    out["phrase_matches"] = sum(length > 1 for _, length in units)
    # A phrase is one rated unit, not an additional vote on top of its component words.
    mean = np.mean([score for score, _ in units], axis=0) if units else [np.nan] * 3
    for dimension, value in zip(DIMENSIONS, mean):
        out[f"v21_phrases_{dimension}"] = float(value)
    return out


def _controls(frame: pd.DataFrame, column: str) -> dict:
    import statsmodels.api as sm

    sub = frame[["debut_year", "n_tokens", column, "v21_tokens_coverage"]].dropna()
    sub = sub[sub["n_tokens"] > 0]
    if len(sub) < 200 or sub[column].std() == 0:
        return {"n": len(sub), "note": "insufficient variation or observations"}
    predictors = pd.DataFrame({
        "decade": (sub["debut_year"] - 1990) / 10,
        "log_tokens": np.log(sub["n_tokens"]),
        "coverage": sub["v21_tokens_coverage"],
    }, index=sub.index)
    model = sm.OLS(sub[column] / sub[column].std(), sm.add_constant(predictors)).fit(
        cov_type="cluster", cov_kwds={"groups": sub["debut_year"]},
    )
    strata = []
    for _, part in sub.groupby(pd.qcut(sub["n_tokens"], 5, duplicates="drop"), observed=True):
        rho = stats.spearmanr(part["debut_year"], part[column]).statistic
        strata.append({"n": len(part), "tokens_min": int(part["n_tokens"].min()),
                       "tokens_max": int(part["n_tokens"].max()), "spearman": float(rho)})
    return {
        "n": len(sub), "length_and_coverage_adjusted_sd_per_decade": float(model.params["decade"]),
        "ci95_clustered_by_debut_year": model.conf_int().loc["decade"].tolist(),
        "length_strata": strata,
        "note": "Exploratory association, not causal. Year clustering does not correct serial or artist dependence.",
    }


def _add_series(frame: pd.DataFrame, sample: str, basis: str, variants: tuple,
                rows: list, results: list, cutoff_year: int) -> None:
    for variant in variants:
        dimensions = ("valence",) if "polar" in variant else DIMENSIONS
        for dimension in dimensions:
            column = f"{variant}_{dimension}"
            for weighting, weight in (("exposure", "points"), ("equal_song", None)):
                series = trends.yearly_series(frame, column, weight)
                if series.empty:
                    continue
                series["partial_period"] = series["year"] == cutoff_year
                results.append({
                    "sample": sample, "time_basis": basis, "variant": variant,
                    "dimension": dimension, "weighting": weighting,
                    "trend": trends.trend_test(series),
                    "completed_period_trend": trends.trend_test(series[~series["partial_period"]]),
                })
                rows.append(series.assign(sample=sample, time_basis=basis, variant=variant,
                                          dimension=dimension, weighting=weighting))


def run(archive: Path | None = None, offline: bool = False) -> dict:
    expanded = load_vad_v21(archive, offline)
    songs = load_joined()
    eligible = songs[
        songs["has_lyrics"] & songs["is_english"].fillna(False)
        & ~songs["is_instrumental"].fillna(False)
    ].sort_values("song_id").copy()
    inputs = [
        DERIVED / "songs_weighted.parquet", DERIVED / "lyrics_index.parquet",
        LEX_DIR / "nrc_vad.parquet", LEX_DIR / "nrc_vad_v21.parquet",
        Path(__file__), Path(__file__).with_name("lyrics_features.py"),
        *[LYRICS_CACHE / f"{song}.json" for song in eligible["song_id"]],
    ]
    signature = fingerprint({
        "schema": 2, "tokenizer": _WORD.pattern, "sections": _SECTION.pattern,
        "phrase_boundaries": BOUNDARIES.pattern, "polar_thresholds": [.167, .333],
        "output_scale": [0, 1], "phrase_weighting": "one_per_longest_nonoverlapping_unit",
    }, inputs)
    cache = DERIVED / "nrc_vad_robustness.parquet"
    if cache_matches(cache, signature):
        features = pd.read_parquet(cache)
        print(f"Using cached NRC VAD comparisons for {len(features):,} songs")
    else:
        lookups = make_lookups(pd.read_parquet(LEX_DIR / "nrc_vad.parquet"), expanded)
        records = []
        for song_id in tqdm(eligible["song_id"], desc="NRC VAD v2.1", unit="song"):
            text = load_lyrics(song_id)
            if not text:
                raise ValueError(f"Eligible lyric missing: {song_id}")
            records.append({"song_id": song_id, **score_lyrics(text, *lookups)})
        features = pd.DataFrame(records)
        atomic_parquet(cache, features)
        mark_cache(cache, signature)
    frame = eligible.drop(columns="n_tokens", errors="ignore").merge(
        features, on="song_id", validate="one_to_one",
    )
    # Verify the baseline independently rather than trusting a scale/version label.
    for dimension in DIMENSIONS:
        if not np.allclose(frame[f"v1_tokens_{dimension}"], frame[f"vad_{dimension}"],
                           equal_nan=True, atol=1e-12):
            raise ValueError(f"Recomputed NRC VAD v1 {dimension} differs from production")
    common = frame.dropna(subset=[f"{v}_valence" for v in VARIANTS]).copy()
    if len(common) < 50:
        raise ValueError("Insufficient matched songs for NRC VAD comparison")
    integrity = json.loads((INTERIM / "chart_integrity.json").read_text())
    cutoff = integrity["chart_through"]
    cutoff_year = pd.Timestamp(cutoff).year
    trends._reset_rng()
    rows, results = [], []
    _add_series(common, "matched_all_variants", "debut_year", VARIANTS,
                rows, results, cutoff_year)
    _add_series(frame, "all_eligible", "debut_year", ("v21_tokens",),
                rows, results, cutoff_year)
    _add_series(frame[frame["v21_tokens_matched_tokens"] >= 10], "minimum_10_matches",
                "debut_year", ("v21_tokens",), rows, results, cutoff_year)
    chart = trends.chart_year_frame(frame, pd.read_parquet(DERIVED / "song_year_exposure.parquet"))
    _add_series(chart, "all_eligible", "chart_year", ("v21_tokens",),
                rows, results, cutoff_year)
    followup = trends.fixed_followup_frame(
        frame, pd.read_parquet(INTERIM / "chart_entries_resolved.parquet"),
    )
    # All included fixed-follow-up cohorts are complete, including their latest cohort.
    _add_series(followup, "all_eligible", "first_52_weeks", ("v21_tokens",),
                rows, results, cutoff_year)
    pd.concat(rows, ignore_index=True).to_csv(REPORTS / "nrc_vad_series.csv", index=False)
    coverage_rows = []
    for year, group in frame.groupby("debut_year"):
        for variant in VARIANTS:
            coverage_rows.append({
                "year": int(year), "variant": variant, "n_eligible": len(group),
                "n_scored": int(group[f"{variant}_valence"].notna().sum()),
                "mean_token_coverage": float(group[f"{variant}_coverage"].mean()),
                "exposure_weighted_token_coverage": float(np.average(
                    group[f"{variant}_coverage"], weights=group["points"],
                )),
            })
    pd.DataFrame(coverage_rows).to_csv(REPORTS / "nrc_vad_coverage.csv", index=False)
    provenance = json.loads((LEX_DIR / "nrc_vad_v21.provenance.json").read_text())
    baseline = pd.read_parquet(LEX_DIR / "nrc_vad.parquet")
    shared = baseline.rename(columns={"word": "term"}).merge(
        expanded, on="term", suffixes=("_v1", "_v21"), validate="one_to_one",
    )
    vocabulary = {
        "v1_terms": len(baseline), "v21_terms": len(expanded), "shared_terms": len(shared),
        "v1_terms_not_in_v21": len(baseline) - len(shared),
        "revised_shared_scores": {
            dimension: int((((shared[f"{dimension}_v21"] + 1) / 2
                             - shared[f"{dimension}_v1"]).abs() > .001).sum())
            for dimension in DIMENSIONS
        },
        "revision_tolerance_on_0_1_scale": .001,
    }
    summary = {
        "chart_through": cutoff, "input_signature": signature, "lexicon": provenance,
        "vocabulary_comparison": vocabulary,
        "n_eligible": len(frame), "n_matched_all_variants": len(common),
        "scale": "All dimensions rescaled to [0,1]; v2.1 native [-1,1] maps to (score+1)/2.",
        "matching": "Exact existing ASCII/apostrophe tokenizer; no stemming, lemmatization or synonyms.",
        "phrases": "Longest nonoverlapping match within lines/sections/strong punctuation; "
                   "one vote per matched term, no double-counted component words.",
        "polar": "Filter valence by absolute native score >=0.167 or >=0.333; never tune to trend.",
        "uncertainty": "1000 song bootstrap replicates within each year; intervals do not handle "
                       "serial/artist dependence. Rank-test p-values are exploratory and unadjusted.",
        "interpretation": "VAD are word/phrase norms, not validated song mood, sadness, or listener emotion. "
                          "Arousal and dominance are separate dimensions, not added to a sadness index.",
        "controls": {d: _controls(frame, f"v21_tokens_{d}") for d in DIMENSIONS},
        "n_songs_with_phrase_matches": int(frame["phrase_matches"].gt(0).sum()),
        "results": results,
    }
    atomic_json(REPORTS / "nrc_vad_robustness.json", summary)
    from . import vad_statistics

    vad_statistics.run(REPORTS / "nrc_vad_series.csv", integrity["chart_from"], cutoff, REPORTS)
    print(f"NRC VAD comparison: {len(frame):,} eligible; {len(common):,} matched across all variants")
    return summary
