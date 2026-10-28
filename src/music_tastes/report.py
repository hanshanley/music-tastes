"""Render figures and findings from the current analysis, without fixed conclusions."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis_trends import METRICS
from .page_plots import export as export_page_plots
from .paths import DERIVED, FIGURES, INTERIM, REPORTS, atomic_text
from .vizstyle import (
    ACCENT, BLUE, CHARTS, MUTED, chart_filename, chart_heading, house_style,
    mark_partial_years, period_labels, save_fig, source_note,
)

house_style()

SOURCE_LYRICS = "Source: US Billboard Hot 100; locally scored lyrics."
MODEL_NOTE = "Relationship labels are model classifications; independent human validation is pending."


def _generated_stamp() -> str:
    if os.environ.get("MUSIC_TASTES_STABLE_REPORT"):
        return "_Generated from the committed pipeline outputs._"
    return f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._"


class _Unit:
    def __init__(self, header: str, kind: str) -> None:
        self.header, self.kind = header, kind

    def fmt(self, value) -> str:
        if value is None or pd.isna(value):
            return "\u2014"
        if self.kind == "percent":
            return f"{value:.1%}"
        if self.kind == "count":
            return f"{value:,.0f}"
        if self.kind == "decimal3":
            return f"{value:.3f}"
        return f"{value:.1f}"


_PERCENT = _Unit("", "percent")
_UNITS = {
    "bpm": _Unit(" (BPM)", "decimal"),
    "lyric_length": _Unit(" (words)", "count"),
    "vader_valence": _Unit(" (VADER score)", "decimal"),
    "lyric_valence": _Unit(" (0\u20131 valence)", "decimal3"),
    "acoustic_mood_happy": _Unit(" (P(happy))", "percent"),
    "acoustic_mood_sad": _Unit(" (P(sad))", "percent"),
    "lyric_sadness": _Unit(" (share of words)", "percent"),
    "lyric_joy": _Unit(" (share of words)", "percent"),
}


def _metric_unit(metric: str) -> _Unit:
    return _UNITS.get(metric, _PERCENT)


def _read_json(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _missed_median_peak() -> int:
    idx = pd.read_parquet(DERIVED / "lyrics_index.parquet")[["song_id", "has_lyrics"]]
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")[["song_id", "peak_rank"]]
    joined = songs.merge(idx, on="song_id", how="left", validate="one_to_one")
    missed = joined.loc[~joined["has_lyrics"].fillna(False), "peak_rank"].dropna()
    return int(missed.median()) if len(missed) else 0


def _share_range_text() -> str:
    path = REPORTS / "decade_series.csv"
    if not path.exists():
        return "not yet measured by decade"
    df = pd.read_csv(path)
    sub = df[(df["metric"] == "relationship_share") & (df["variant"] == "weighted_all")]
    if sub.empty:
        return "not yet measured by decade"
    return f"{sub['mean'].min():.0%}\u2013{sub['mean'].max():.0%} across {len(sub)} decades"


def summary_for(metric: str, trends: dict) -> str:
    result = trends.get(metric, {})
    primary = result.get("weighted_all", {})
    balanced = result.get("weighted_complete_case", {})
    tau, p = primary.get("kendall_tau"), primary.get("p_value")
    if tau is None or p is None or not math.isfinite(p):
        return "Insufficient evidence to estimate a trend."
    if tau * balanced.get("kendall_tau", tau) < 0:
        return "Unresolved: direction changes in the balanced covered-song subset."
    direction = "Increasing" if tau > 0 else "Decreasing"
    if not primary.get("significant_at_05"):
        direction = "No statistically clear monotonic trend"
    return f"{direction} (tau={tau:+.2f}, p={p:.2g})."


def _chart_subtitle(metric: str) -> str:
    if metric in {"independence_share", "heartbreak_share", "devotion_share"}:
        return "Among covered relationship songs, weighted by chart exposure"
    if metric == "relationship_share":
        return "Model-classified romantic or sexual topics, weighted by chart exposure"
    return "Covered songs, weighted by chart exposure"


def _plot_metric(series: pd.DataFrame, metric: str, description: str, path) -> bool:
    sub = series[
        (series["metric"] == metric) & (series["variant"] == "weighted_all")
    ].sort_values("year")
    if sub.empty:
        return False
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sub["year"], sub["mean"], color=ACCENT, lw=2.4, label="Covered songs")
    ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"],
                    color=ACCENT, alpha=0.10, linewidth=0)
    if _metric_unit(metric).kind == "percent":
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0 if metric.endswith("_share") else 1))
        ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.margins(x=.015)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    mark_partial_years(ax, sub["year"])
    chart_heading(ax, CHARTS[metric][1], _chart_subtitle(metric))
    ax.set_ylabel(CHARTS[metric][2])
    ax.set_xlabel("Song debut year")
    note = (SOURCE_LYRICS + "\nShading: 95% song-bootstrap intervals. "
            "*Partial years; recent cohorts have shorter follow-up.")
    if metric in {"relationship_share", "independence_share", "heartbreak_share", "devotion_share"}:
        note += "\n" + MODEL_NOTE
    source_note(fig, note)
    save_fig(fig, path, bottom=.20)
    return True


def _plot_decades(decades: pd.DataFrame, metric: str, description: str, path) -> bool:
    sub = decades[
        (decades["metric"] == metric) & (decades["variant"] == "weighted_all")
    ].sort_values("decade")
    if sub.empty:
        return False
    unit = _metric_unit(metric)
    factor = 100 if unit.kind == "percent" else 1
    error = [
        (sub["mean"] - sub["ci_lo"]).clip(lower=0).to_numpy() * factor,
        (sub["ci_hi"] - sub["mean"]).clip(lower=0).to_numpy() * factor,
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(period_labels(sub["decade"], decade=True), sub["mean"] * factor,
           yerr=error, color=ACCENT, capsize=3, width=0.68,
           error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.set_ylabel(CHARTS[metric][2])
    if factor == 100:
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0 if metric.endswith("_share") else 1))
    chart_heading(ax, CHARTS[metric][1] + " by decade", _chart_subtitle(metric))
    ax.set_xlabel("Song debut decade")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    note = (SOURCE_LYRICS + "\nError bars: 95% song-bootstrap intervals. "
            "*Partial decades; recent cohorts have shorter follow-up.")
    if metric in {"relationship_share", "independence_share", "heartbreak_share", "devotion_share"}:
        note += "\n" + MODEL_NOTE
    source_note(fig, note)
    save_fig(fig, path, bottom=.20)
    return True


def _plot_coverage(path) -> None:
    cov = pd.read_csv(REPORTS / "coverage_by_year.csv")
    fig, ax = plt.subplots(figsize=(11, 6))
    for column, label, color, linestyle in [
        ("coverage_lyrics", "Songs", ACCENT, "-"),
        ("coverage_exposure", "Chart exposure", BLUE, "--"),
    ]:
        if column in cov:
            ax.plot(cov["debut_year"], cov[column] * 100, color=color,
                    lw=2.2, linestyle=linestyle, label=label)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylabel(CHARTS["coverage_by_year"][2])
    ax.set_xlabel("Song debut year")
    mark_partial_years(ax, cov["debut_year"])
    chart_heading(ax, CHARTS["coverage_by_year"][1], "Retrieved lyrics, not necessarily eligible for analysis")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", frameon=False)
    source_note(fig, SOURCE_LYRICS + "\n*Partial calendar years. Retrieval does not imply analysis eligibility.")
    save_fig(fig, path, bottom=.17)


def _vad_section() -> list[str]:
    result = _read_json(REPORTS / "nrc_vad_robustness.json")
    series_path = REPORTS / "nrc_vad_series.csv"
    if not result or not series_path.exists():
        return []
    series = pd.read_csv(series_path)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    labels = {"valence": "Pleasantness\n(valence, 0-1)", "arousal": "Activation\n(arousal, 0-1)",
              "dominance": "Sense of control\n(dominance, 0-1)"}
    for ax, dimension in zip(axes, ("valence", "arousal", "dominance")):
        for variant, label, color, linestyle in [
            ("v1_tokens", "NRC VAD v1", BLUE, "--"),
            ("v21_tokens", "NRC VAD v2.1", ACCENT, "-"),
        ]:
            sub = series[
                (series["sample"] == "matched_all_variants")
                & (series["time_basis"] == "debut_year")
                & (series["weighting"] == "exposure")
                & (series["dimension"] == dimension) & (series["variant"] == variant)
            ].sort_values("year")
            ax.plot(sub["year"], sub["mean"], lw=2, color=color, label=label, linestyle=linestyle)
            ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"],
                            color=color, alpha=.08, linewidth=0)
        ax.set_ylabel(labels[dimension], fontsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.margins(x=.015)
        ax.grid(True, axis="y")
    mark_partial_years(axes[-1], series["year"])
    chart_heading(axes[0], CHARTS["nrc_vad_robustness"][1],
                  "Same covered songs in both versions, weighted by chart exposure")
    axes[0].legend(ncol=2, fontsize=10, frameon=False)
    axes[-1].set_xlabel("Song debut year")
    source_note(fig, "Sources: US Billboard Hot 100; NRC VAD, Mohammad / NRC Canada (2018, 2025)."
                "\nShading: 95% song-bootstrap intervals. *Partial years; recent cohorts have shorter follow-up."
                "\nWord ratings are on 0-1 scales, with separate panel ranges; these are not song-sadness scores.")
    save_fig(fig, FIGURES / chart_filename("nrc_vad_robustness"), bottom=.15)
    lines = [
        "", "## NRC VAD version and scoring sensitivity", "",
        f"Compared **{result['n_matched_all_variants']:,} identical songs** across all scoring variants,",
        f"from **{result['n_eligible']:,} eligible lyrics**.",
        "NRC VAD v2.1 is rescaled from its native -1 to +1 to 0-1 before comparison",
        "with the unchanged v1 baseline. These are word/phrase ratings, not percentages",
        "of happiness, a validated sadness classifier, or listener-emotion measurements.", "",
        "| Matched-song valence measure | Change per decade (0-1 units) | Kendall tau |",
        "|---|---:|---:|",
    ]
    for row in result["results"]:
        if (row["sample"] == "matched_all_variants" and row["time_basis"] == "debut_year"
                and row["dimension"] == "valence" and row["weighting"] == "exposure"):
            estimate = row["trend"]
            if "change_per_decade" in estimate:
                lines.append(
                    f"| {row['variant']} | {estimate['change_per_decade']:+.4f} | "
                    f"{estimate['kendall_tau']:+.2f} |"
                )
    lines += [
        "", "The shared-vocabulary variants hold matched words constant while changing",
        "their version-specific ratings; expanded-vocabulary variants also change coverage.",
        "", "Unique-word scoring removes repetition. Phrase scoring uses longest",
        "nonoverlapping terms within lines/sections, without also counting their component words.",
        "Polar-word sensitivities use fixed native valence thresholds of +/-0.167 and +/-0.333.",
        "Arousal and dominance remain separate dimensions, not an invented sadness index.",
        "Completed-period, minimum-match, chart-year, fixed-follow-up, and length/coverage",
        "checks are exploratory; agreement does not establish historical invariance or causality.", "",
        f"![Three dimensions of lyrical emotion](figures/{chart_filename('nrc_vad_robustness')})", "",
        "[Complete annual series and intervals](nrc_vad_series.csv) |",
        "[Annual token coverage](nrc_vad_coverage.csv) |",
        "[Methods, provenance, controls and trend results](nrc_vad_robustness.json)", "",
    ]
    statistics = _read_json(REPORTS / "nrc_vad_statistics.json")
    if statistics:
        lines += [
            "### Dependence-aware statistical checks", "",
            f"Completed years **{statistics['first_complete_year']}-"
            f"{statistics['last_complete_year']}**; partial boundary years excluded.",
            "Linear slopes use Newey-West serial-correlation/heteroskedasticity-robust",
            "inference. The table uses five annual lags and Holm correction across the",
            "three primary v2.1 dimensions. Intervals are pointwise 95%, not simultaneous.", "",
            "| Dimension | OLS change/decade | 95% HAC interval | Holm-adjusted p |",
            "|---|---:|---|---:|",
        ]
        for row in statistics["tests"]:
            if row["is_primary"] and row["method"] == "newey_west" and row["window_years"] == 5:
                lines.append(
                    f"| {row['dimension'].capitalize()} | {row['slope_per_decade']:+.5f} | "
                    f"[{row['ci95_lo']:+.5f}, {row['ci95_hi']:+.5f}] | "
                    f"{row['p_holm_primary']:.3g} |"
                )
        lines += [
            "", "All three-, five- and ten-year HAC windows and circular residual-block",
            "bootstrap checks are available below, including multiplicity correction",
            f"over all {statistics['n_comparisons']} comparisons and paired version contrasts.",
            "These are retrospective checks, not preregistered confirmation. Failure",
            "to reject zero does not establish a flat trend or equivalence. Tests do not",
            "validate song sadness, remove coverage bias, or establish causality.", "",
            "[Complete statistical tests](nrc_vad_statistics.csv) |",
            "[Settings, assumptions and results](nrc_vad_statistics.json)", "",
        ]
    return lines


def run() -> None:
    series_path = DERIVED / "yearly_series.parquet"
    if not series_path.exists():
        raise SystemExit("Run music-tastes trends first.")
    series = pd.read_parquet(series_path)
    trends = _read_json(REPORTS / "trend_results.json")
    coverage = _read_json(REPORTS / "coverage_audit.json")
    integrity = _read_json(INTERIM / "chart_integrity.json")
    decades = pd.read_csv(REPORTS / "decade_series.csv")
    figures = []
    for metric, (_, description, _) in METRICS.items():
        if _plot_metric(series, metric, description, FIGURES / chart_filename(metric)):
            figures.append(metric)
        _plot_decades(decades, metric, description, FIGURES / chart_filename(metric, decade=True))
    _plot_coverage(FIGURES / chart_filename("coverage_by_year"))
    lines = [
        "# Relationships and mood in US hit songs", "", _generated_stamp(), "",
        f"Chart issues through **{integrity.get('chart_through', 'unverified')}**.",
        "This is a retrospective snapshot, not an automatically updated series.",
        "Issue dates can be later than publication dates. The latest year/decade is incomplete.",
        "", "## What is being measured", "",
        "**Relationship means romantic or sexual relationship**, not every form of love.",
        "A local NLI model tests this topic on the first 380 cleaned words; scores >0.5",
        "count as positive. Scores are not calibrated probabilities. Stances are scored",
        "separately on chunks, then reported only within relationship-positive songs.",
        "See [methods and review rubric](../METHODS.md).", "",
        "Debut-year results describe cohorts using observed lifetime chart exposure.",
        "They do not measure precisely what listeners heard in that calendar year.",
        "See [chart-year comparison](chart_year_series.csv) for that alternative.",
        "Recent cohorts have shorter observed chart lifetimes.", "",
        "[Fixed 52-week follow-up](followup_52_week_series.csv) compares entirely",
        "observed debut-year cohorts using only their first 52 calendar weeks of chart points.", "",
        "## Coverage", "",
        f"Lyrics: **{coverage.get('coverage_lyrics_overall', 0):.1%}** of songs; "
        f"**{coverage.get('coverage_exposure_overall', 0):.1%}** of exposure.",
        "A balanced high-exposure covered-song subset is a sensitivity check, not",
        "proof that missingness bias has been removed. Metric-specific missingness remains.",
        "See the [gap/change audit](gap_audit.json) and",
        "[coverage changes by decade](coverage_change_by_decade.csv).",
        "", f"![Lyric coverage](figures/{chart_filename('coverage_by_year')})", "",
        "## Results", "", "| Measure | Exposure-weighted debut-year trend |",
        "|---|---|",
    ]
    for metric, (_, description, _) in METRICS.items():
        lines.append(f"| {CHARTS[metric][1]} | {summary_for(metric, trends)} |")
    lines += _vad_section()
    lines += ["", "## Classifier validation", ""]
    gold = _read_json(REPORTS / "gold" / "validation.json")
    independent = gold.get("independent_review", {"status": "pending"})
    lines.append(f"Independent relationship review: **{independent.get('status', 'pending')}**.")
    lines.append(f"Conditional stance review: **{independent.get('stance_review_status', 'pending')}**.")
    for name in ("random_sample", "purposive_sample"):
        result = gold.get(name, {})
        for task, stats in result.items():
            if isinstance(stats, dict) and "accuracy" in stats:
                lines.append(
                    f"- Legacy {name}, {task}: n={stats['n']}, "
                    f"accuracy={stats['accuracy']:.2f}. Selection/annotation provenance "
                    "is not independently audited; purposive results are not population accuracy."
                )
    validity = _read_json(REPORTS / "validity.json")
    agg = validity.get("aggregation_bias", {})
    if agg.get("chunk_adjusted_p") is not None:
        lines += ["", f"Chunk-count-adjusted independence estimate: "
                  f"{agg.get('chunk_adjusted_per_decade', float('nan')):+.4f}/decade "
                  f"(p={agg['chunk_adjusted_p']:.2g})."]
    lines += [
        "", "## Robustness and limitations", "",
        "Primary tests are exploratory associations, not causal estimates. Annual series",
        "can be autocorrelated; reported rank-test p-values are not a correction for all",
        "tested hypotheses. Coverage agreement alone does not establish representativeness.",
        "English/non-instrumental eligibility and source/recording matching restrict the sample.",
        "AcousticBrainz stopped accepting submissions in 2022: missing recent audio is",
        "not evidence of no musical change. Audio mood scores are classifier outputs,",
        "not direct measurements of human emotion.", "",
        "[Confounds](confounds.json) | [Construct validity](validity.json) |",
        "[Wording sensitivity](prompt_robustness.json) |",
        "[Relationship wording/input sensitivity](relationship_sensitivity.json) |",
        "[Threshold sensitivity](relationship_threshold_sensitivity.json) |",
        "[Gold validation](gold/validation.json)", "",
    ]
    for metric in figures:
        lines += [f"### {CHARTS[metric][1]}", "",
                  f"![{CHARTS[metric][1]}](figures/{chart_filename(metric)})", ""]
    atomic_text(REPORTS / "findings.md", "\n".join(lines))
    export_page_plots(FIGURES)
    print(f"Wrote {REPORTS / 'findings.md'} and {len(figures)} metric figures")


if __name__ == "__main__":
    run()
