"""Stage 13: rival explanations for the headline trends.

A falling lyric valence across 68 years has at least three explanations that have
nothing to do with songs becoming less happy:

1. **Genre mix shifted.** Rap and R&B went from absent to dominant on the Hot 100,
   and they have different lyrical conventions (longer lyrics, more profanity, more
   first-person narrative) than the pop and country of the 1960s. If valence differs
   by genre, a change in the *mix* moves the average without any genre changing.
2. **Measurement regime changed.** The Hot 100 formula changed in 1991, 1998, 2005,
   2007 and 2013. A trend spanning those dates crosses measurement regimes.
3. **Lyrics got longer.** Word-level valence is an average over matched tokens, and
   longer lyrics with larger vocabularies regress toward the lexicon mean.

This module tests each. A trend that persists within genre, within the post-1991
era, and after controlling for lyric length is a considerably stronger claim than the
raw series. One that vanishes under any of them is reported as confounded.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .analysis_trends import derive_labels, trend_test, yearly_series
from .coverage import load_joined
from .paths import REPORTS

# The metrics worth stress-testing: the ones with a headline claim attached.
TARGETS = {
    "lyric_valence": "vad_valence",
    "lyric_joy": "emo_joy",
    "independence_share": "is_independent",
    "relationship_share": "is_relationship",
}

GENRE_COL = "ab_genre_dortmund"
MIN_PER_STRATUM = 150


def by_genre(df: pd.DataFrame, metric: str, column: str) -> pd.DataFrame:
    """Re-run the trend separately within each genre stratum."""
    if GENRE_COL not in df.columns:
        return pd.DataFrame()
    rows = []
    for genre, g in df.groupby(GENRE_COL):
        sub = g[g[column].notna()]
        if len(sub) < MIN_PER_STRATUM:
            continue
        series = yearly_series(sub, column, "points", min_n=6)
        if len(series) < 10:
            continue
        t = trend_test(series)
        rows.append(
            {
                "metric": metric,
                "genre": genre,
                "n_songs": len(sub),
                "n_years": t.get("n_years"),
                "kendall_tau": t.get("kendall_tau"),
                "p_value": t.get("p_value"),
                "change_per_decade": t.get("change_per_decade"),
            }
        )
    return pd.DataFrame(rows)


def genre_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Exposure-weighted genre share by decade -- the mechanism being tested."""
    if GENRE_COL not in df.columns:
        return pd.DataFrame()
    d = df[df[GENRE_COL].notna()].copy()
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    tot = d.groupby("decade")["points"].sum()
    share = (
        d.groupby(["decade", GENRE_COL])["points"].sum().unstack(fill_value=0).div(tot, axis=0)
    )
    return share.reset_index()


def within_era(df: pd.DataFrame, metric: str, column: str, start_year: int) -> dict:
    """Re-run the trend using only years from a single measurement regime."""
    sub = df[df["debut_year"] >= start_year]
    series = yearly_series(sub, column, "points", min_n=6)
    if len(series) < 10:
        return {"note": f"too few years from {start_year}"}
    return trend_test(series)


def length_controlled(df: pd.DataFrame, column: str) -> dict:
    """Partial correlation of the metric with year, controlling for lyric length.

    If longer lyrics mechanically pull word-average valence toward the lexicon mean,
    then year and valence would correlate only because lyrics lengthened. Regressing
    both on log word count and correlating the residuals removes that path.
    """
    sub = df[[column, "n_words", "debut_year"]].dropna()
    sub = sub[sub["n_words"] > 0]
    if len(sub) < 200:
        return {"note": "insufficient data"}

    logw = np.log(sub["n_words"].to_numpy(dtype=float))
    year = sub["debut_year"].to_numpy(dtype=float)
    value = sub[column].to_numpy(dtype=float)

    def resid(y):
        slope, intercept = np.polyfit(logw, y, 1)
        return y - (slope * logw + intercept)

    raw_r, raw_p = stats.spearmanr(year, value)
    part_r, part_p = stats.spearmanr(resid(year), resid(value))
    length_r, length_p = stats.spearmanr(year, logw)

    return {
        "n": int(len(sub)),
        "raw_spearman_year_vs_metric": float(raw_r),
        "raw_p": float(raw_p),
        "partial_spearman_controlling_length": float(part_r),
        "partial_p": float(part_p),
        "spearman_year_vs_log_words": float(length_r),
        "length_p": float(length_p),
        "attenuation": float(raw_r - part_r),
    }


def run() -> dict:
    df = derive_labels(load_joined())
    results: dict[str, dict] = {}

    mix = genre_mix(df)
    if not mix.empty:
        mix.to_csv(REPORTS / "genre_mix_by_decade.csv", index=False)

    genre_rows = []
    for metric, column in TARGETS.items():
        if column not in df.columns or df[column].notna().sum() < 200:
            continue
        entry: dict[str, object] = {}

        entry["post_1991_soundscan"] = within_era(df, metric, column, 1991)
        entry["post_2005_digital"] = within_era(df, metric, column, 2005)
        if column in ("vad_valence", "emo_joy"):
            entry["length_control"] = length_controlled(df, column)

        g = by_genre(df, metric, column)
        if not g.empty:
            genre_rows.append(g)
            taus = g["kendall_tau"].dropna()
            entry["genre_strata"] = {
                "n_strata": int(len(g)),
                "taus": {r.genre: round(r.kendall_tau, 3) for r in g.itertuples()},
                "all_same_sign": bool(len(taus) > 0 and (taus > 0).all() or (taus < 0).all()),
                "n_significant": int((g["p_value"] < 0.05).sum()),
            }
        results[metric] = entry

    if genre_rows:
        pd.concat(genre_rows, ignore_index=True).to_csv(
            REPORTS / "trends_by_genre.csv", index=False
        )

    (REPORTS / "confounds.json").write_text(json.dumps(results, indent=2, default=str))

    print("Confound checks\n")
    if not mix.empty:
        print("Exposure-weighted genre mix by decade (Essentia genre_dortmund):")
        cols = [c for c in mix.columns if c != "decade"]
        print("  decade  " + "  ".join(f"{c[:9]:>9}" for c in cols))
        for r in mix.itertuples(index=False):
            vals = "  ".join(f"{getattr(r, c, 0):>8.1%}" for c in cols)
            print(f"  {int(r.decade)}s   {vals}")
        print()

    for metric, entry in results.items():
        print(f"{metric}")
        for era_key, label in [
            ("post_1991_soundscan", "1991+ (SoundScan era only)"),
            ("post_2005_digital", "2005+ (digital/streaming only)"),
        ]:
            t = entry.get(era_key, {})
            if "kendall_tau" in t:
                sig = "significant" if t["significant_at_05"] else "n.s."
                print(f"    {label:32} tau={t['kendall_tau']:+.3f} "
                      f"p={t['p_value']:.2g} ({sig})")
        lc = entry.get("length_control")
        if lc and "partial_spearman_controlling_length" in lc:
            print(f"    {'controlling lyric length':32} "
                  f"raw rho={lc['raw_spearman_year_vs_metric']:+.3f} -> "
                  f"partial={lc['partial_spearman_controlling_length']:+.3f} "
                  f"(year vs log words rho={lc['spearman_year_vs_log_words']:+.3f})")
        gs = entry.get("genre_strata")
        if gs:
            print(f"    {'within genre':32} {gs['n_strata']} strata, "
                  f"{gs['n_significant']} significant, "
                  f"same sign: {gs['all_same_sign']}")
            print(f"      taus: {gs['taus']}")
        print()
    return results


if __name__ == "__main__":
    run()
