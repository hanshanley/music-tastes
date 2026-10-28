"""Stage 7: coverage audit -- the gate that every trend claim must pass.

The danger this stage exists to catch
-------------------------------------
Lyrics coverage is not random. Genius is a modern, crowd-sourced site: a 2019 hit is
far more likely to have a transcribed lyric than a 1961 hit. If coverage correlates
with year, then any year-level average is computed over a differently-selected
population each year, and a "trend" can be produced entirely by which songs happen to
be present.

This module quantifies that correlation and defines a balanced covered-song
sensitivity subset. Direction changes are reported as unresolved; agreement
does not establish that selection bias is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .paths import DATA, DERIVED, INTERIM, REPORTS, ROOT, atomic_json, require


def load_joined() -> pd.DataFrame:
    """Join songs, exposure weights, and whichever feature tables exist."""
    songs = pd.read_parquet(require(DERIVED / "songs_weighted.parquet"))

    index_path = DERIVED / "lyrics_index.parquet"
    if index_path.exists():
        idx = pd.read_parquet(index_path)
        keep = [
            c
            for c in ["song_id", "matched", "match_tier", "source", "status", "has_lyrics",
                      "is_instrumental", "is_english", "n_words"]
            if c in idx.columns
        ]
        songs = songs.merge(idx[keep], on="song_id", how="left")

    for path, tag in [
        (DERIVED / "lyric_features_method_a.parquet", "a"),
        (DERIVED / "lyric_features_method_b.parquet", "b"),
        (DERIVED / "acoustic_features.parquet", "ac"),
    ]:
        if path.exists():
            feats = pd.read_parquet(path)
            if not feats.empty:
                songs = songs.merge(feats, on="song_id", how="left", suffixes=("", f"_{tag}"))

    if "scale" in songs.columns:
        # Minor-key share is a standard proxy for musical (as opposed to lyrical)
        # sadness, and is a different construct from either lyric valence or
        # Essentia's mood classifier, so it is tracked separately.
        songs["is_minor"] = songs["scale"].map({"minor": 1.0, "major": 0.0})

    songs["has_lyrics"] = songs.get(
        "has_lyrics", pd.Series(False, index=songs.index)
    ).fillna(False).astype(bool)
    songs["scored_b"] = (
        songs["p_independence_max"].notna()
        if "p_independence_max" in songs.columns
        else False
    )
    return songs


def coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year counts and coverage rates for each pipeline stage.

    Reports coverage two ways. Song-count coverage is the share of charting songs we
    have lyrics for. **Exposure coverage** is the share of total chart points those
    songs represent, and it is the relevant figure for every weighted result: missing
    songs can differ systematically in chart exposure; the difference is measured,
    not assumed constant across refreshes.
    """
    g = df.groupby("debut_year")
    covered = df[df["has_lyrics"]].groupby("debut_year")["points"].sum()
    total = g["points"].sum()
    out = pd.DataFrame(
        {
            "n_songs": g.size(),
            "n_matched": g["matched"].sum(min_count=1),
            "n_lyrics": g["has_lyrics"].sum(),
            "total_points": total,
            "covered_points": covered,
        }
    )
    if "is_english" in df:
        eligible = df["has_lyrics"] & df["is_english"].fillna(False)
        if "is_instrumental" in df:
            eligible &= ~df["is_instrumental"].fillna(False)
        out["n_eligible"] = eligible.groupby(df["debut_year"]).sum()
    if "scored_b" in df.columns:
        out["n_scored_b"] = g["scored_b"].sum()
        out["coverage_scored_b"] = out["n_scored_b"] / out["n_songs"]
    out["coverage_lyrics"] = out["n_lyrics"] / out["n_songs"]
    out["coverage_exposure"] = (out["covered_points"] / out["total_points"]).fillna(0.0)
    return out.reset_index()


def coverage_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage by chart success, to detect a popularity bias in the lyric source."""
    bins = [0, 1, 5, 10, 25, 50, 100]
    labels = ["#1", "#2-5", "#6-10", "#11-25", "#26-50", "#51-100"]
    df = df.copy()
    df["peak_band"] = pd.cut(df["peak_rank"], bins=bins, labels=labels, include_lowest=True)
    g = df.groupby("peak_band", observed=True)
    return pd.DataFrame(
        {"n_songs": g.size(), "coverage_lyrics": g["has_lyrics"].mean()}
    ).reset_index()


def audit(df: pd.DataFrame) -> dict:
    """Test whether coverage correlates with year, and size a complete-case subset."""
    by_year = coverage_by_year(df)
    valid = by_year[by_year["n_songs"] >= 20].dropna(subset=["coverage_lyrics"])
    total_pts = float(df["points"].sum())
    covered_pts = float(df.loc[df["has_lyrics"], "points"].sum())
    if len(valid) < 3:
        return {
            "years_examined": int(len(valid)),
            "coverage_lyrics_overall": float(df["has_lyrics"].mean()),
            "coverage_exposure_overall": covered_pts / total_pts if total_pts else None,
            "note": "too few years with enough songs to test coverage dependence",
        }

    # Convert out of pandas nullable dtypes: scipy needs plain float arrays.
    years = valid["debut_year"].to_numpy(dtype=float)
    cov = valid["coverage_lyrics"].to_numpy(dtype=float)
    rho, p_value = stats.spearmanr(years, cov)

    # A complete-case subset holds coverage roughly constant across years by capping
    # each year at the number of songs the worst-covered year can supply.
    min_cov = float(valid["coverage_lyrics"].min())
    worst_year = int(valid.loc[valid["coverage_lyrics"].idxmin(), "debut_year"])

    total_pts = float(df["points"].sum())
    covered_pts = float(df.loc[df["has_lyrics"], "points"].sum())

    report = {
        "years_examined": int(len(valid)),
        "coverage_lyrics_overall": float(df["has_lyrics"].mean()),
        "coverage_exposure_overall": covered_pts / total_pts if total_pts else None,
        "coverage_min": min_cov,
        "coverage_min_year": worst_year,
        "coverage_max": float(valid["coverage_lyrics"].max()),
        "coverage_max_year": int(valid.loc[valid["coverage_lyrics"].idxmax(), "debut_year"]),
        "spearman_rho_year_vs_coverage": float(rho),
        "spearman_p": float(p_value),
        "coverage_is_year_dependent": bool(p_value < 0.05),
    }
    return report


def complete_case_subset(
    df: pd.DataFrame, per_year: int | None = None, column: str | None = None,
) -> pd.DataFrame:
    """Equal-N-per-year sample of lyric-covered songs, ranked by chart exposure.

    When supplied, ``column`` must be observed before selecting equal year counts.
    This balances sample sizes, not selection bias or population representativeness.
    """
    covered = df[df["has_lyrics"]].copy()
    if column is not None:
        covered = covered.dropna(subset=[column])
    counts = covered.groupby("debut_year").size()
    counts = counts[counts >= 10]
    if per_year is None:
        per_year = int(counts.min()) if len(counts) else 0
    covered = covered[covered["debut_year"].isin(counts.index)]
    return (
        covered.sort_values("points", ascending=False)
        .groupby("debut_year", group_keys=False)
        .head(per_year)
        .reset_index(drop=True)
    )


def coverage_changes(previous: pd.DataFrame, current: pd.DataFrame, key: str) -> pd.DataFrame:
    columns = [key, "n_songs", "n_lyrics", "coverage_lyrics", "coverage_exposure"]
    result = previous[columns].merge(
        current[columns], on=key, how="outer", suffixes=("_previous", "_current"),
        validate="one_to_one",
    )
    for metric in ("coverage_lyrics", "coverage_exposure"):
        result[f"{metric}_change_pp"] = 100 * (
            result[f"{metric}_current"] - result[f"{metric}_previous"]
        )
    return result.sort_values(key)


def write_change_audit(by_year: pd.DataFrame, report: dict) -> None:
    old_reports = ROOT / "reports"
    previous_path = old_reports / "coverage_by_year.csv"
    baseline = None
    previous = by_year.iloc[:0].copy()
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        manifest_path = old_reports / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        baseline = {
            "n_songs": int(previous["n_songs"].sum()),
            "n_lyrics": int(previous["n_lyrics"].sum()),
            "chart_through": manifest.get("chart_through"),
        }
    coverage_changes(previous, by_year, "debut_year").to_csv(
        REPORTS / "coverage_change_by_year.csv", index=False
    )

    def by_decade(frame):
        totals = frame.assign(decade=frame["debut_year"] // 10 * 10).groupby("decade")[
            ["n_songs", "n_lyrics", "total_points", "covered_points"]
        ].sum()
        totals["coverage_lyrics"] = totals["n_lyrics"] / totals["n_songs"]
        totals["coverage_exposure"] = totals["covered_points"] / totals["total_points"]
        return totals.reset_index()

    coverage_changes(by_decade(previous), by_decade(by_year), "decade").to_csv(
        REPORTS / "coverage_change_by_decade.csv", index=False
    )
    integrity = json.loads((INTERIM / "chart_integrity.json").read_text())
    result = {
        "baseline": baseline, "current": report,
        "chart_through": integrity["chart_through"],
        "chart_integrity": integrity,
        "comparison_note": (
            "Changes describe different chart/lyric snapshots, not matched-song recovery counts. "
            "A legacy snapshot without a manifest has no verified chart cutoff."
        ),
    }
    recovery_path = DATA / "lyric_recovery_audit.json"
    if recovery_path.exists():
        from .cache import fingerprint

        recovery = json.loads(recovery_path.read_text())
        if recovery.get("lyrics_index_signature") == fingerprint(
            {}, [DERIVED / "lyrics_index.parquet"],
        ):
            result["matched_recovery"] = {
                key: recovery[key] for key in (
                    "observed_at", "additional_lyrics",
                    "eligibility_reclassifications", "lyrics_before_merge",
                    "lyrics_after_merge", "recovery_states", "retry_passes",
                ) if key in recovery
            }
            if "removed_misattributed_lyrics" in recovery:
                result["matched_recovery"]["removed_recording_transcriptions"] = recovery[
                    "removed_misattributed_lyrics"
                ]
                result["matched_recovery"]["removal_note"] = (
                    "Excluded instrumental-recording transcriptions; this does not establish "
                    "that every attached word was fabricated or that the recordings lack speech."
                )
    atomic_json(REPORTS / "gap_audit.json", result)


def run() -> dict:
    df = load_joined()
    by_year = coverage_by_year(df)
    by_rank = coverage_by_rank(df)
    report = audit(df)
    report["n_songs"] = len(df)
    report["n_lyrics"] = int(df["has_lyrics"].sum())
    report["n_scored_b"] = int(df["scored_b"].sum())
    if "status" in df:
        report["lyric_outcomes"] = df["status"].fillna("never_attempted").value_counts().to_dict()
    report["subset_note"] = (
        "Equal-N high-exposure covered songs, not a fully observed or unbiased population."
    )

    subset = complete_case_subset(df)
    report["complete_case_per_year"] = (
        int(subset.groupby("debut_year").size().max()) if len(subset) else 0
    )
    report["complete_case_n"] = int(len(subset))

    write_change_audit(by_year, report)
    by_year.to_csv(REPORTS / "coverage_by_year.csv", index=False)
    by_rank.to_csv(REPORTS / "coverage_by_rank.csv", index=False)
    queue_columns = [
        column for column in (
            "song_id", "title_display", "artist_display", "debut_year", "points",
            "peak_rank", "status", "source", "match_tier",
        ) if column in df
    ]
    df.loc[~df["has_lyrics"], queue_columns].to_csv(
        REPORTS / "lyrics_unresolved.csv", index=False
    )
    from .analysis_trends import METRICS, derive_labels

    labelled = derive_labels(df)
    metric_counts = []
    for metric, (column, _, _) in METRICS.items():
        if column in labelled:
            counts = labelled.groupby("debut_year")[column].count().reset_index(name="n_observed")
            counts["metric"] = metric
            metric_counts.append(counts)
    if metric_counts:
        pd.concat(metric_counts, ignore_index=True).to_csv(
            REPORTS / "metric_coverage_by_year.csv", index=False
        )
    (REPORTS / "coverage_audit.json").write_text(json.dumps(report, indent=2))

    print("Coverage audit")
    print(f"  songs total:            {len(df):,}")
    print(f"  with lyrics:            {df['has_lyrics'].sum():,} "
          f"({report['coverage_lyrics_overall']:.1%} of songs, "
          f"{report['coverage_exposure_overall']:.1%} of chart exposure)")
    if "scored_b" in df:
        print(f"  scored by Method B:     {df['scored_b'].sum():,}")
    if "coverage_min" in report:
        print(f"  coverage range: {report['coverage_min']:.1%} to {report['coverage_max']:.1%}")
        print(f"  Spearman(year, coverage) = {report['spearman_rho_year_vs_coverage']:+.3f}")
    if report.get("coverage_is_year_dependent"):
        print("  ** coverage IS year-dependent: every headline result must be")
        print("     re-run on the complete-case subset before it can be reported.")
    print(f"  complete-case subset:   {report['complete_case_n']:,} songs "
          f"({report['complete_case_per_year']}/year)")
    print("\n  Coverage by peak chart position:")
    for r in by_rank.itertuples():
        print(f"    {str(r.peak_band):8} n={r.n_songs:6,}  coverage={r.coverage_lyrics:5.1%}")
    return report


if __name__ == "__main__":
    run()
