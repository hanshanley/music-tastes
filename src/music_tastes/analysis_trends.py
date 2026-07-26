"""Stage 8: year-level trends with uncertainty.

Every series is produced four ways and all four are reported:

  * exposure-weighted and unweighted -- weighting by chart points answers "what did
    America hear", the unweighted version answers "what did the industry release into
    the chart". These can diverge and the difference is itself informative.
  * on all covered songs and on the complete-case subset from
    :mod:`music_tastes.coverage` -- a trend that appears only in the full set is a
    coverage artefact, not a finding.

Weights are normalized *within* each year. Without that, the streaming era would
dominate every pooled statistic: chart tenure has grown from roughly 13 weeks in the
1960s to 90+ weeks today, so raw chart points are an order of magnitude larger now and
would swamp any cross-year comparison.

Trend tests are rank-based (Mann-Kendall, Theil-Sen) rather than least-squares,
because these series are short, autocorrelated and not reliably linear.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from .coverage import complete_case_subset, load_joined
from .paths import DERIVED, REPORTS

RNG = np.random.default_rng(20260725)
N_BOOT = 1000

# metric -> (column, human description, higher_means)
METRICS = {
    "lyric_valence": ("vad_valence", "Lyric valence (NRC VAD, 0=negative 1=positive)", "happier"),
    "lyric_sadness": ("emo_sadness", "Share of words with a sadness association (NRC EmoLex)", "sadder"),
    "lyric_joy": ("emo_joy", "Share of words with a joy association (NRC EmoLex)", "happier"),
    "vader_valence": ("vader_valence", "Lyric valence (VADER)", "happier"),
    "relationship_share": ("is_relationship", "Share of hits that are about a relationship", "more love songs"),
    "independence_share": ("is_independent", "Share of relationship songs taking an 'I don't need you' stance", "more independence"),
    "heartbreak_share": ("is_heartbreak", "Share of relationship songs about heartbreak/wanting an ex back", "more heartbreak"),
    "devotion_share": ("is_devotion", "Share of relationship songs about devotion/commitment", "more devotion"),
}

# Probability thresholds for turning Method B's continuous scores into labels.
# 0.5 is the natural entailment cut; sensitivity to it is reported separately.
P_THRESHOLD = 0.5


def derive_labels(df: pd.DataFrame, threshold: float = P_THRESHOLD) -> pd.DataFrame:
    """Turn Method B probabilities into the indicator columns the metrics need."""
    df = df.copy()
    if "p_relationship_doc" in df.columns:
        df["is_relationship"] = (df["p_relationship_doc"] > threshold).astype(float)
        df.loc[df["p_relationship_doc"].isna(), "is_relationship"] = np.nan

    # Stance shares are conditional on being a relationship song, so they are NaN
    # (not 0) for non-relationship songs and therefore drop out of the denominator.
    rel = df.get("is_relationship")
    for name, col in [
        ("is_independent", "p_independence_max"),
        ("is_heartbreak", "p_heartbreak_max"),
        ("is_devotion", "p_devotion_max"),
        ("is_longing", "p_longing_max"),
        ("is_casual", "p_casual_max"),
        ("is_conflict", "p_conflict_max"),
    ]:
        if col in df.columns:
            val = (df[col] > threshold).astype(float)
            val[df[col].isna()] = np.nan
            if rel is not None:
                val[rel != 1] = np.nan
            df[name] = val
    return df


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else np.nan


def yearly_series(
    df: pd.DataFrame, column: str, weight_col: str | None, min_n: int = 8
) -> pd.DataFrame:
    """Per-year mean with a bootstrap confidence interval.

    The bootstrap resamples songs within each year, so the interval reflects how much
    the estimate depends on which songs happened to be sampled and covered.
    """
    rows = []
    for year, g in df.groupby("debut_year"):
        sub = g[[column] + ([weight_col] if weight_col else [])].dropna()
        n = len(sub)
        if n < min_n:
            continue
        values = sub[column].to_numpy(dtype=float)
        weights = (
            sub[weight_col].to_numpy(dtype=float) if weight_col else np.ones(n, dtype=float)
        )
        point = _weighted_mean(values, weights)

        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boots = np.array([_weighted_mean(values[i], weights[i]) for i in idx])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])

        rows.append(
            {"year": int(year), "n": n, "mean": point, "ci_lo": float(lo), "ci_hi": float(hi)}
        )
    return pd.DataFrame(rows)


def trend_test(series: pd.DataFrame) -> dict:
    """Mann-Kendall significance plus a Theil-Sen slope, both rank-based."""
    if len(series) < 10:
        return {"n_years": len(series), "note": "too few years to test"}

    years = series["year"].to_numpy(dtype=float)
    values = series["mean"].to_numpy(dtype=float)

    tau, p_value = stats.kendalltau(years, values)
    slope, intercept, lo_slope, hi_slope = stats.theilslopes(values, years, 0.95)

    first = float(values[:5].mean())
    last = float(values[-5:].mean())

    return {
        "n_years": int(len(series)),
        "year_min": int(years.min()),
        "year_max": int(years.max()),
        "kendall_tau": float(tau),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "theil_sen_slope_per_year": float(slope),
        "theil_sen_slope_ci": [float(lo_slope), float(hi_slope)],
        "change_per_decade": float(slope * 10),
        "mean_first_5_years": first,
        "mean_last_5_years": last,
        "absolute_change": last - first,
    }


def run_metric(df: pd.DataFrame, name: str, column: str, weight_col: str | None) -> dict | None:
    if column not in df.columns or df[column].notna().sum() < 50:
        return None
    series = yearly_series(df, column, weight_col)
    if series.empty:
        return None
    return {"series": series, "trend": trend_test(series)}


def decade_series(
    df: pd.DataFrame, column: str, weight_col: str | None, min_n: int = 30
) -> pd.DataFrame:
    """Per-decade mean with a bootstrap CI.

    Yearly estimates of a rare binary outcome are dominated by binomial noise: with
    roughly 34 relationship songs in a year and a true rate near 0.1, the standard
    error is about 5 percentage points, which is why the yearly independence series
    swings between 0 and 0.8. Pooling to decades multiplies the sample by ten and
    makes the level -- not just the direction -- readable.
    """
    rows = []
    df = df.copy()
    df["decade"] = (df["debut_year"] // 10 * 10).astype(int)
    for decade, g in df.groupby("decade"):
        sub = g[[column] + ([weight_col] if weight_col else [])].dropna()
        n = len(sub)
        if n < min_n:
            continue
        values = sub[column].to_numpy(dtype=float)
        weights = (
            sub[weight_col].to_numpy(dtype=float) if weight_col else np.ones(n, dtype=float)
        )
        point = _weighted_mean(values, weights)
        idx = RNG.integers(0, n, size=(N_BOOT, n))
        boots = np.array([_weighted_mean(values[i], weights[i]) for i in idx])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(
            {
                "decade": decade,
                "n": n,
                "mean": point,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


def run(threshold: float = P_THRESHOLD) -> dict:
    df = derive_labels(load_joined(), threshold)
    subset = derive_labels(complete_case_subset(load_joined()), threshold)

    results: dict[str, dict] = {}
    all_series = []

    for metric, (column, description, direction) in METRICS.items():
        entry: dict[str, dict] = {"description": description, "higher_means": direction}
        for label, data, weight in [
            ("weighted_all", df, "points"),
            ("unweighted_all", df, None),
            ("weighted_complete_case", subset, "points"),
            ("unweighted_complete_case", subset, None),
        ]:
            res = run_metric(data, metric, column, weight)
            if res is None:
                continue
            entry[label] = res["trend"]
            s = res["series"].copy()
            s["metric"] = metric
            s["variant"] = label
            all_series.append(s)
        if len(entry) > 2:
            results[metric] = entry

    if all_series:
        combined = pd.concat(all_series, ignore_index=True)
        combined.to_parquet(DERIVED / "yearly_series.parquet", index=False)
        combined.to_csv(REPORTS / "yearly_series.csv", index=False)

    # Decade view: the readable version for sparse binary outcomes.
    decade_rows = []
    for metric, (column, _desc, _dir) in METRICS.items():
        if column not in df.columns:
            continue
        for label, data, weight in [
            ("weighted_all", df, "points"),
            ("unweighted_all", df, None),
        ]:
            d = decade_series(data, column, weight)
            if d.empty:
                continue
            d["metric"] = metric
            d["variant"] = label
            decade_rows.append(d)
    if decade_rows:
        decades = pd.concat(decade_rows, ignore_index=True)
        decades.to_csv(REPORTS / "decade_series.csv", index=False)
        decades.to_parquet(DERIVED / "decade_series.parquet", index=False)

    (REPORTS / "trend_results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"Trend results (threshold p>{threshold})\n")
    for metric, entry in results.items():
        print(f"{metric}  --  {entry['description']}")
        for variant in [
            "weighted_all", "unweighted_all",
            "weighted_complete_case", "unweighted_complete_case",
        ]:
            t = entry.get(variant)
            if not t or "kendall_tau" not in t:
                continue
            sig = "significant" if t["significant_at_05"] else "n.s."
            print(
                f"    {variant:26} {t['year_min']}-{t['year_max']}  "
                f"tau={t['kendall_tau']:+.3f} p={t['p_value']:.2g} ({sig})  "
                f"per-decade={t['change_per_decade']:+.4f}  "
                f"first5={t['mean_first_5_years']:.3f} last5={t['mean_last_5_years']:.3f}"
            )
        # A sign disagreement between the full and complete-case views means the
        # coverage audit's warning has bitten; say so rather than picking a winner.
        signs = {
            v: np.sign(entry[v]["kendall_tau"])
            for v in ("weighted_all", "weighted_complete_case")
            if v in entry and "kendall_tau" in entry[v]
        }
        if len(signs) == 2 and len(set(signs.values())) > 1:
            print("    ** direction FLIPS between full and complete-case: unresolved")
        print()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=P_THRESHOLD)
    args = ap.parse_args()
    run(threshold=args.threshold)
