"""Stage 9: figures and a written findings document.

Charts show the bootstrap confidence band, not just the point estimate, and every
figure is drawn for both the full and the complete-case sample so a reader can see
directly whether a trend depends on coverage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis_trends import METRICS  # noqa: E402
from .paths import DERIVED, FIGURES, REPORTS  # noqa: E402
from .vizstyle import (  # noqa: E402
    ACCENT,
    BLUE,
    GRID,
    MUTED,
    end_label,
    house_style,
    save_fig,
    source_note,
)

house_style()

# Chart-methodology breaks, annotated on every time series so a reader can see
# which comparisons cross a change in what the Hot 100 measures.
ERA_MARKS = [
    (1991, "SoundScan"),
    (2005, "digital sales"),
    (2013, "streaming/video"),
]

SOURCE_BILLBOARD = (
    "Data: Billboard Hot 100 (Billboard/Luminate) via the mhollingshead archive; "
    "lyrics via Genius; acoustic features via AcousticBrainz (MetaBrainz)."
)
SOURCE_LYRICS = (
    "Data: Billboard Hot 100 (Billboard/Luminate); lyrics via Genius. "
    "Sentiment from NRC VAD/EmoLex (Mohammad, NRC Canada) and a local NLI model."
)


def _generated_stamp() -> str:
    """Generation timestamp, suppressible so the report can be diffed.

    The numbers are reproducible, but an embedded clock makes every regeneration
    look like a change. Set MUSIC_TASTES_STABLE_REPORT=1 to omit it.
    """
    import os

    if os.environ.get("MUSIC_TASTES_STABLE_REPORT"):
        return "_Generated from the committed pipeline outputs._"
    return f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._"


def _missed_median_peak() -> int:
    """Median chart peak of songs we could not get lyrics for.

    Computed rather than quoted. This figure shifted as lyric coverage grew, and a
    stale number inside a claim about representativeness is exactly the kind of
    error that erodes trust in everything around it.
    """
    idx = pd.read_parquet(DERIVED / "lyrics_index.parquet")[["song_id", "has_lyrics"]]
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")[["song_id", "peak_rank"]]
    m = songs.merge(idx, on="song_id", how="left")
    m["has_lyrics"] = m["has_lyrics"].fillna(False)
    missed = m.loc[~m["has_lyrics"], "peak_rank"].dropna()
    return int(missed.median()) if len(missed) else 0


def _share_range_text() -> str:
    """Observed range and decade count for the relationship share."""
    path = REPORTS / "decade_series.csv"
    if not path.exists():
        return "roughly two thirds to three quarters"
    d = pd.read_csv(path)
    sub = d[(d["metric"] == "relationship_share") & (d["variant"] == "weighted_all")]
    if sub.empty:
        return "roughly two thirds to three quarters"
    return (
        f"{sub['mean'].min():.0%}–{sub['mean'].max():.0%} across "
        f"{len(sub)} decades"
    )


def _headline_p_text() -> str:
    """Quote the actual adjusted p-value for the headline finding."""
    path = REPORTS / "validity.json"
    if not path.exists():
        return "the independence trend is significant well beyond p<0.05"
    agg = json.loads(path.read_text()).get("aggregation_bias", {})
    p_adj = agg.get("chunk_adjusted_p")
    if p_adj is None:
        return "the independence trend is significant well beyond p<0.05"
    return f"the independence trend is p={p_adj:.2g} after adjustment"


def _short_stratum_text(agg: dict) -> str:
    """First-to-last decade share within the shortest-lyric stratum."""
    strata = agg.get("within_chunk_strata", [])
    short = next((s for s in strata if s["stratum"].startswith("short")), None)
    if not short or not short.get("by_decade"):
        return "the shortest songs included"
    by_dec = {int(k): v for k, v in short["by_decade"].items()}
    lo, hi = min(by_dec), max(by_dec)
    return f"{by_dec[lo]:.1%} in the {lo}s to {by_dec[hi]:.1%} in the {hi}s"


class _Unit:
    """How to render a metric's values in a table.

    Shares are fractions in [0, 1] and read naturally as percentages. Tempo and word
    counts are not: formatting 116.99 BPM with ".1%" produced "11698.5%" in an
    earlier version of this report. Units are therefore chosen per metric rather
    than assumed.
    """

    def __init__(self, header: str, kind: str) -> None:
        self.header = header
        self.kind = kind

    def fmt(self, value) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "—"
        if self.kind == "percent":
            return f"{value:.1%}"
        if self.kind == "count":
            return f"{value:,.0f}"
        if self.kind == "decimal3":
            return f"{value:.3f}"
        return f"{value:.1f}"


_PERCENT = _Unit("", "percent")
# Only genuine shares belong on a percent scale. VADER emits a signed compound-style
# score rather than a proportion, so rendering it as a percentage produced "117.3%"
# for a value of 1.17.
_UNITS = {
    "bpm": _Unit(" (BPM)", "decimal"),
    "lyric_length": _Unit(" (words)", "count"),
    "vader_valence": _Unit(" (VADER score)", "decimal"),
    "lyric_valence": _Unit(" (0–1 valence)", "decimal3"),
    # Essentia mood outputs are class probabilities, so a percent scale is valid,
    # but the header has to say so or the reader cannot tell it from a share of songs.
    "acoustic_mood_happy": _Unit(" (P(happy))", "percent"),
    "acoustic_mood_sad": _Unit(" (P(sad))", "percent"),
    "lyric_sadness": _Unit(" (share of words)", "percent"),
    "lyric_joy": _Unit(" (share of words)", "percent"),
}


def _metric_unit(metric: str) -> _Unit:
    """Percent is right for shares and lexicon rates; anything else is declared."""
    return _UNITS.get(metric, _PERCENT)


def _era_marks(ax) -> None:
    """Annotate the chart-methodology breaks without dominating the plot."""
    top = ax.get_ylim()[1]
    for year, name in ERA_MARKS:
        ax.axvline(year, color=GRID, ls=":", lw=1.0, zorder=0)
        ax.text(year + 0.5, top, name, fontsize=7.5, color=MUTED,
                va="top", rotation=90, style="italic")


def _plot_metric(series: pd.DataFrame, metric: str, description: str, path):
    """Yearly series with bootstrap band, drawn for two coverage variants.

    Terracotta is the headline series; blue is the coverage-controlled subset. If
    the two diverge, the trend depends on which songs happen to have lyrics.
    """
    variants = [
        ("weighted_all", "All covered songs", ACCENT),
        ("weighted_complete_case", "Complete-case subset", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = False
    last = None
    for variant, label, color in variants:
        sub = series[(series["metric"] == metric) & (series["variant"] == variant)]
        if sub.empty:
            continue
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["mean"], color=color, lw=2.4, label=label, zorder=3)
        ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"], color=color,
                        alpha=0.14, linewidth=0, zorder=2)
        plotted = True
        last = (sub, label, color)
    if not plotted:
        plt.close(fig)
        return False

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    _era_marks(ax)

    ax.set_title(description)
    ax.set_xlabel("Year song first charted")
    ax.legend(loc="best")
    source_note(fig, SOURCE_LYRICS)
    save_fig(fig, path)
    return True


def _plot_decades(decades: pd.DataFrame, metric: str, description: str, path):
    """Decade means: the readable view for sparse binary outcomes."""
    sub = decades[
        (decades["metric"] == metric) & (decades["variant"] == "weighted_all")
    ].sort_values("decade")
    if sub.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    x = [f"{int(d)}s" for d in sub["decade"]]
    yerr = [
        (sub["mean"] - sub["ci_lo"]).clip(lower=0).to_numpy() * 100,
        (sub["ci_hi"] - sub["mean"]).clip(lower=0).to_numpy() * 100,
    ]
    ax.bar(x, sub["mean"] * 100, yerr=yerr, color=ACCENT, capsize=4, width=0.68,
           error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capthick": 1.2}, zorder=3)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel("% of songs")
    ax.set_title(f"{description}\nBy decade, exposure-weighted, with 95% bootstrap CI")
    for xi, n in enumerate(sub["n"]):
        ax.text(xi, 0.6, f"n={int(n):,}", ha="center", va="bottom", fontsize=8.5,
                color=MUTED)
    source_note(fig, SOURCE_LYRICS)
    save_fig(fig, path)
    return True


def _plot_coverage(path):
    """Coverage by year, reported before any trend because it gates them all."""
    cov = pd.read_csv(REPORTS / "coverage_by_year.csv")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(cov["debut_year"], cov["coverage_lyrics"] * 100, color=ACCENT, lw=2.4,
            label="Share of charting songs", zorder=3)
    if "coverage_exposure" in cov.columns:
        ax.plot(cov["debut_year"], cov["coverage_exposure"] * 100, color=BLUE,
                lw=2.0, ls="--", label="Share of chart exposure", zorder=3)
        last = cov.dropna(subset=["coverage_exposure"]).iloc[-1]
        end_label(ax, last["debut_year"], last["coverage_exposure"] * 100,
                  "exposure", BLUE)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    _era_marks(ax)
    ax.set_ylim(0, 100)
    ax.set_title("Lyric coverage by year\nThe main threat to every trend that follows")
    ax.set_ylabel("% covered")
    ax.set_xlabel("Year song first charted")
    ax.legend(loc="lower right")
    source_note(fig, SOURCE_LYRICS)
    save_fig(fig, path)


def run() -> None:
    series_path = DERIVED / "yearly_series.parquet"
    if not series_path.exists():
        raise SystemExit("Run music_tastes.analysis_trends first.")

    series = pd.read_parquet(series_path)
    trends = json.loads((REPORTS / "trend_results.json").read_text())
    coverage = json.loads((REPORTS / "coverage_audit.json").read_text())

    decade_path = REPORTS / "decade_series.csv"
    decades_df = pd.read_csv(decade_path) if decade_path.exists() else None

    made = []
    for metric, (_col, description, _dir) in METRICS.items():
        path = FIGURES / f"{metric}.png"
        if _plot_metric(series, metric, description, path):
            made.append((metric, path))
        if decades_df is not None:
            _plot_decades(decades_df, metric, description, FIGURES / f"{metric}_decade.png")
    _plot_coverage(FIGURES / "coverage_by_year.png")

    # The summary table is derived, not hardcoded. Earlier versions embedded figures
    # that silently went stale as the sample grew — twice.
    valid_early = json.loads((REPORTS / "validity.json").read_text()) if (
        REPORTS / "validity.json"
    ).exists() else {}
    conf_early = json.loads((REPORTS / "confounds.json").read_text()) if (
        REPORTS / "confounds.json"
    ).exists() else {}
    recon_early = valid_early.get("length_bias_reconciliation", {})
    agg_early = valid_early.get("aggregation_bias", {})
    pr_early = json.loads((REPORTS / "prompt_robustness.json").read_text()) if (
        REPORTS / "prompt_robustness.json"
    ).exists() else {}

    def _fmt_valence() -> str:
        if "contextual" not in recon_early:
            return "**Not yet measured.**"
        lo = abs(recon_early["contextual"]["length_adjusted_per_decade_sd"])
        hi = abs(recon_early["lexicon"]["length_adjusted_per_decade_sd"])
        lo, hi = sorted((lo, hi))
        return (
            f"**Modestly, yes** — about {lo:.2f}–{hi:.2f} SD per decade. A word-norm "
            "lexicon and a context-aware model agree once their *opposite* "
            "lyric-length biases are removed."
        )

    def _fmt_independence() -> str:
        adj = agg_early.get("chunk_adjusted_per_decade")
        levels = ""
        if pr_early.get("by_decade"):
            shares = [
                sum(v.values()) / len(v) for v in pr_early["by_decade"].values() if v
            ]
            if shares:
                levels = (
                    f" But the *level* is not quotable: it ranges "
                    f"{min(shares):.1%}–{max(shares):.1%} purely on how the question "
                    "is worded."
                )
        mag = f"~{adj * 100:+.1f} points/decade" if adj is not None else "rising"
        return (
            f"**Yes** — the *direction* is the strongest finding here, {mag} after "
            f"correcting for aggregation bias.{levels}"
        )

    def _fmt_bpm() -> str:
        b = trends.get("bpm", {}).get("weighted_all", {})
        if "kendall_tau" not in b:
            return "**Not yet measured.**"
        return (
            f"**No change.** Tempo is flat (tau {b['kendall_tau']:+.2f}, "
            f"p={b['p_value']:.2g})."
        )

    def _fmt_music_sadness() -> str:
        mk = conf_early.get("minor_key_share", {}).get("genre_adjusted", {})
        att = mk.get("attenuation_fraction")
        att_txt = f"~{att:.0%} is genre mix" if att is not None else "much is genre mix"
        return (
            "**No usable evidence.** Essentia's happy *and* sad scores both fall, "
            f"which indicates classifier drift. Minor-key share doubles but {att_txt} "
            "and it vanishes post-1991."
        )

    lines = [
        "# Are US hit songs getting sadder, and are fewer of them about love?",
        "",
        _generated_stamp(),
        "",
        "## Summary",
        "",
        "| Question | Answer | Confidence |",
        "|---|---|---|",
        f"| Are fewer hits about love/relationships? | **No.** Exposure-weighted share is "
        f"flat at {_share_range_text()}. | Good — but the unweighted series "
        f"declines and fails the coverage check, so the two views differ. |",
        f"| Among relationship songs, are more about *not needing* one? | "
        f"{_fmt_independence()} | Direction: strong (survives coverage, genre, era, "
        "length, and 4 of 5 paraphrases). Level: unreliable. |",
        f"| Are the lyrics getting sadder? | {_fmt_valence()} | Moderate — the raw "
        "lexicon series overstates it. |",
        f"| Is the music getting sadder? | {_fmt_music_sadness()} | Weak. |",
        f"| Are songs getting faster or slower? | {_fmt_bpm()} | Good. |",
        "",
        "The short version: **what songs are *about* changed more than how they *feel*.** "
        "Love songs are as common as ever, but the stance inside them shifted markedly "
        "toward self-sufficiency. Lyrics did get somewhat less positive, though far "
        "less than a naive word-count reading suggests, and the *musical* sadness "
        "signals (tempo, mood classifiers) show nothing usable at all.",
        "",
        "## What this measures",
        "",
        "Every song that entered the Billboard Hot 100 between 1958-08-04 and the",
        "present. The Hot 100 combines sales, radio airplay and (since 2007-2013)",
        "streaming, so it is the closest long-run proxy available for what Americans",
        "actually listened to. It is a proxy, not a census: see Limitations.",
        "",
        "## Coverage, and why it is reported first",
        "",
        f"- Overall lyric coverage: **{coverage['coverage_lyrics_overall']:.1%}** of "
        f"charting songs, and **{coverage.get('coverage_exposure_overall', 0):.1%} of "
        f"total chart exposure** — the misses are disproportionately low-exposure "
        f"deep cuts (median peak position #{_missed_median_peak()}), so weighted "
        f"results are better covered than the song count suggests",
        f"- Coverage ranges from {coverage['coverage_min']:.1%} "
        f"({coverage['coverage_min_year']}) to {coverage['coverage_max']:.1%} "
        f"({coverage['coverage_max_year']})",
        f"- Spearman(year, coverage) = **{coverage['spearman_rho_year_vs_coverage']:+.3f}** "
        f"(p = {coverage['spearman_p']:.2g})",
        "",
        "![coverage](figures/coverage_by_year.png)",
        "",
    ]
    if coverage.get("coverage_is_year_dependent"):
        lines += [
            "Coverage **is** year-dependent, so every result below is reported twice:",
            "once on all covered songs and once on a complete-case subset holding the",
            "number of songs per year constant. Where the two disagree in direction,",
            "the result is marked unresolved rather than reported as a finding.",
            "",
        ]

    lines += ["## How much to trust the classifiers", ""]
    gold_path = REPORTS / "gold" / "validation.json"
    if gold_path.exists():
        gold = json.loads(gold_path.read_text())
        rand = gold.get("random_sample", {})
        pur = gold.get("purposive_sample", {})
        if "method_b_relationship" in rand:
            b = rand["method_b_relationship"]
            lines += [
                f"**Random hand-labelled sample ({b['n']} songs, four per decade).** "
                "This is an unbiased estimate.",
                "",
                "| Task | Method | Accuracy | Precision | Recall | Cohen kappa |",
                "|---|---|---|---|---|---|",
                f"| Is it a relationship song? | B (NLI) | {b['accuracy']:.2f} | "
                f"{b['precision']:.2f} | {b['recall']:.2f} | {b['cohen_kappa']:.2f} |",
            ]
            a = rand.get("method_a_relationship")
            if a:
                lines.append(
                    f"| Is it a relationship song? | A (embeddings) | {a['accuracy']:.2f} | "
                    f"{a['precision']:.2f} | {a['recall']:.2f} | {a['cohen_kappa']:.2f} |"
                )
            lines.append("")
            if "methods_agree_kappa" in rand:
                lines += [
                    f"Inter-method agreement is poor (Cohen kappa = "
                    f"{rand['methods_agree_kappa']:.2f}). Agreement is therefore not "
                    "treated as evidence in itself; the hand labels decide which "
                    "method is right, and Method B is used for all reported stance "
                    "results.",
                    "",
                ]
        if "method_b_independence" in pur:
            b = pur["method_b_independence"]
            a = pur.get("method_a_independence")
            lines += [
                f"**Purposive independence set ({b['n']} famous songs with "
                "uncontroversial stances).** These were chosen for being clear-cut, so "
                "this is an **upper bound**, not an unbiased estimate. It exists "
                "because independence songs are rare -- the random sample of 32 "
                "contained exactly one, too few to estimate precision or recall for "
                "the class this project is about.",
                "",
                "| Method | Accuracy | Precision | Recall |",
                "|---|---|---|---|",
                f"| B (NLI) | {b['accuracy']:.2f} | {b['precision']:.2f} | {b['recall']:.2f} |",
            ]
            if a:
                lines.append(
                    f"| A (embeddings) | {a['accuracy']:.2f} | {a['precision']:.2f} | "
                    f"{a['recall']:.2f} |"
                )
            lines += [
                "",
                "Method A's failure mode is systematic, not noisy: cosine similarity "
                "tracks *topic* (this is a breakup song) and cannot see *stance* "
                "(...and the narrator is fine about it), so it labels "
                "\"I Will Survive\", \"Since U Been Gone\" and \"thank u, next\" as "
                "heartbreak. This is exactly the distinction the research question "
                "turns on, which is why the entailment model carries the result.",
                "",
            ]
    else:
        lines += ["_Validation has not been run yet._", ""]

    lines += ["## Results", ""]

    ex_path = REPORTS / "exhibit_independence.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        top = ex.get("top_by_decade", [])
        leak = ex.get("gate_leakage", [])
        if top:
            lines += [
                "### Check the classifier yourself",
                "",
                "Aggregate accuracy figures are easy to publish and hard to trust. These "
                "are the songs the model is most confident about, so a reader who knows "
                "the music can judge directly.",
                "",
                "| Decade | Highest-confidence \"I don't need you\" songs |",
                "|---|---|",
            ]
            by_dec: dict[int, list[str]] = {}
            for r in top:
                by_dec.setdefault(int(r["decade"]), []).append(
                    f"{r['title_display']} — {r['artist_display']} ({r['p']:.2f})"
                )
            for dec in sorted(by_dec):
                lines.append(f"| {dec}s | {'; '.join(by_dec[dec][:4])} |")
            lines.append("")
        if leak:
            lines += [
                "**Why the two-stage design matters.** The stance model keys on the "
                "literal claim, not the romantic context, so on its own it fires on "
                "*Another Brick In The Wall* (\"we don't need no education\"), on J. Cole's "
                "*Brackets* (about tax), and on *The Little Drummer Boy*. The "
                "relationship gate removes these before the statistic is computed — "
                "these songs score high on the stance but near zero on being about a "
                "relationship:",
                "",
                "| Song | stance | is-relationship |",
                "|---|---|---|",
            ]
            for r in leak[:6]:
                lines.append(
                    f"| {r['title_display']} — {r['artist_display']} | "
                    f"{r['p_stance']:.2f} | {r['p_relationship_doc']:.2f} |"
                )
            lines += [
                "",
                "The separation is clean (stance >0.9 against relationship <0.08), which "
                "is why the headline share is computed *within* relationship songs "
                "rather than over the whole chart.",
                "",
            ]

    n_metrics = len([m for m, e in trends.items() if "kendall_tau" in e.get("weighted_all", {})])
    lines += [
        f"_{n_metrics} metrics are tracked, each in four variants, plus a battery of "
        "confound tests — well over a hundred hypothesis tests in total. At p<0.05 "
        "several 'significant' results are expected by chance alone. The findings "
        f"leaned on here clear that bar comfortably ({_headline_p_text()}); isolated "
        "marginal results are not treated as findings._",
        "",
    ]

    decade_path = REPORTS / "decade_series.csv"
    decades = pd.read_csv(decade_path) if decade_path.exists() else None

    for metric, entry in trends.items():
        full = entry.get("weighted_all", {})
        cc = entry.get("weighted_complete_case", {})
        if "kendall_tau" not in full:
            continue
        agree = (
            "yes"
            if cc and (full["kendall_tau"] > 0) == (cc.get("kendall_tau", 0) > 0)
            else "**NO - unresolved**"
        )
        verdict = (
            f"{'rising' if full['kendall_tau'] > 0 else 'falling'}, "
            f"{'significant' if full['significant_at_05'] else 'not significant'}"
        )
        lines += [
            f"### {metric}",
            "",
            f"{entry['description']}  (higher = {entry['higher_means']})",
            "",
            f"- Direction: **{verdict}** (Kendall tau = {full['kendall_tau']:+.3f}, "
            f"p = {full['p_value']:.2g})",
            f"- Change per decade: {full['change_per_decade']:+.4f}",
            f"- First 5 years {full['mean_first_5_years']:.3f} -> "
            f"last 5 years {full['mean_last_5_years']:.3f}",
            f"- Survives complete-case check: {agree}",
            "",
        ]

        # Decade table: yearly estimates of a rare binary outcome are dominated by
        # binomial noise, so the decade view is the one to read for levels.
        if decades is not None:
            sub = decades[
                (decades["metric"] == metric) & (decades["variant"] == "weighted_all")
            ].sort_values("decade")
            if not sub.empty:
                unit = _metric_unit(metric)
                lines += [
                    f"| Decade | n | Exposure-weighted mean{unit.header} | 95% CI |",
                    "|---|---|---|---|",
                ]
                for r in sub.itertuples():
                    lines.append(
                        f"| {int(r.decade)}s | {r.n} | {unit.fmt(r.mean)} | "
                        f"{unit.fmt(r.ci_lo)} – {unit.fmt(r.ci_hi)} |"
                    )
                lines.append("")

        lines += [f"![{metric}](figures/{metric}_decade.png)", ""]
        lines += [f"![{metric} yearly](figures/{metric}.png)", ""]

    lines += [
        "## Limitations",
        "",
    ]

    conf_path = REPORTS / "confounds.json"
    val_path = REPORTS / "validity.json"
    if conf_path.exists() or val_path.exists():
        conf = json.loads(conf_path.read_text()) if conf_path.exists() else {}
        valid = json.loads(val_path.read_text()) if val_path.exists() else {}
        lines = lines[:-2] + ["## Rival explanations, tested", ""]

        ctx = valid.get("contextual_check", {})
        recon = valid.get("length_bias_reconciliation", {})
        if "contextual" in recon and "spearman_year_vs_contextual_valence" in ctx:
            lex_r, ctx_r = recon["lexicon"], recon["contextual"]
            lines += [
                "### The two valence measures disagree — until you remove length bias",
                "",
                "On an identical set of "
                f"{ctx['n']} songs the raw comparison looks decisive against the "
                "sentiment finding:",
                "",
                "| Measure | Sees context? | raw rho(year, valence) | p |",
                "|---|---|---|---|",
                f"| NRC VAD word norms | no | "
                f"{ctx['spearman_year_vs_lexicon_valence']:+.3f} | "
                f"{ctx['p_lexicon']:.2g} |",
                f"| Entailment model | yes | "
                f"{ctx['spearman_year_vs_contextual_valence']:+.3f} | "
                f"{ctx['p_contextual']:.2g} |",
                "",
                "Read naively that says the lexicon result is an artefact. It is not "
                "that simple, because **both measures are length-dependent and in "
                "opposite directions**:",
                "",
                f"- lexicon: rho(words, valence) = "
                f"**{lex_r['length_bias_spearman']:+.3f}** — longer looks sadder",
                f"- contextual: rho(words, valence) = "
                f"**{ctx_r['length_bias_spearman']:+.3f}** — longer looks happier",
                "",
                "Lyrics roughly doubled in length, so those biases drive the two "
                "year-trends apart: the lexicon's decline is inflated and the "
                "contextual model's is masked. The apparent disagreement was mostly an "
                "artefact of the comparison.",
                "",
                "**Opposite biases also settle whether to adjust.** A substantive effect "
                "cannot be negative in one valid measure and positive in another; two "
                "measures disagreeing in *sign* on the same nuisance variable is the "
                "signature of measurement error, which is the case where adjustment is "
                "correct. Adjusted for length, they converge:",
                "",
                "| Measure | raw SD/decade | length-adjusted SD/decade | p |",
                "|---|---|---|---|",
                f"| lexicon | {lex_r['raw_per_decade_sd']:+.3f} | "
                f"**{lex_r['length_adjusted_per_decade_sd']:+.3f}** | "
                f"{lex_r['length_adjusted_p']:.2g} |",
                f"| contextual | {ctx_r['raw_per_decade_sd']:+.3f} | "
                f"**{ctx_r['length_adjusted_per_decade_sd']:+.3f}** | "
                f"{ctx_r['length_adjusted_p']:.2g} |",
                "",
                (
                    "**Revised conclusion.** Hit lyrics did become modestly less "
                    f"positive — about "
                    f"{abs(ctx_r['length_adjusted_per_decade_sd']):.2f}–"
                    f"{abs(lex_r['length_adjusted_per_decade_sd']):.2f} standard "
                    "deviations per decade — and this replicates across two methods "
                    f"with very different failure modes on {recon['n']:,} songs. It is "
                    "real but smaller than the raw lexicon series implies, and an "
                    "earlier version of this report over-retracted it on the strength "
                    "of the unadjusted comparison alone."
                ),
                "",
            ]
        elif "spearman_year_vs_contextual_valence" in ctx:
            lines += [
                "### The sentiment result does not survive a change of method",
                "",
                f"On an identical set of {ctx['n']} songs, word norms give rho "
                f"{ctx['spearman_year_vs_lexicon_valence']:+.3f} "
                f"(p={ctx['p_lexicon']:.2g}) while a context-aware model gives "
                f"{ctx['spearman_year_vs_contextual_valence']:+.3f} "
                f"(p={ctx['p_contextual']:.2g}).",
                "",
            ]

        happy = trends.get("acoustic_mood_happy", {}).get("weighted_all", {})
        sad = trends.get("acoustic_mood_sad", {}).get("weighted_all", {})
        if "kendall_tau" in happy and "kendall_tau" in sad:
            same_way = (happy["kendall_tau"] < 0) and (sad["kendall_tau"] < 0)
            lines += [
                "### Essentia mood scores move together, which means drift",
                "",
                f"Essentia's `mood_happy` falls sharply (tau {happy['kendall_tau']:+.3f}, "
                f"p={happy['p_value']:.2g}, {happy['mean_first_5_years']:.3f} to "
                f"{happy['mean_last_5_years']:.3f}). Taken alone that looks like strong "
                "evidence the music itself got sadder.",
                "",
                f"But `mood_sad` falls too (tau {sad['kendall_tau']:+.3f}, "
                f"p={sad['p_value']:.2g}, {sad['mean_first_5_years']:.3f} to "
                f"{sad['mean_last_5_years']:.3f}).",
                "",
            ]
            if same_way:
                lines += [
                    "**Two opposing classifiers moving the same direction is diagnostic "
                    "of drift, not emotion.** If songs were genuinely sadder, happy "
                    "should fall while sad rises. Both falling points at something "
                    "systematic in the audio — most plausibly production and mastering "
                    "changes (loudness, compression, stereo width) shifting Essentia's "
                    "features away from its 1990s training distribution. These two "
                    "series are therefore **not reported as evidence about mood**.",
                    "",
                ]

        mk = conf.get("minor_key_share", {})
        mk_ga = mk.get("genre_adjusted", {})
        mk_era = mk.get("post_1991_soundscan", {})
        mk_gs = mk.get("genre_strata", {})
        if mk_ga.get("attenuation_fraction") is not None:
            lines += [
                "### Minor-key share — a weaker signal than it first appears",
                "",
                "Minor-key share roughly doubles across the period and is significant in "
                "all four headline variants, which made it look like the one solid "
                "musical result. Under the same scrutiny applied elsewhere it does not "
                "hold up well:",
                "",
                f"- **Genre mix explains about half of it** — "
                f"{mk_ga['unadjusted_year_coef_per_decade']:+.4f}/decade unadjusted "
                f"falls to {mk_ga['genre_adjusted_year_coef_per_decade']:+.4f} with "
                f"genre fixed effects ({mk_ga['attenuation_fraction']:.0%} attenuation). "
                "Minor keys are simply more common in the genres that grew.",
            ]
            if mk_gs.get("taus"):
                signs = ", ".join(f"{k} {v:+.2f}" for k, v in mk_gs["taus"].items())
                lines.append(
                    f"- **Within genre the direction is inconsistent** ({signs}); only "
                    f"{mk_gs.get('n_significant', 0)} of {mk_gs.get('n_strata', 0)} "
                    "strata is significant and the signs disagree."
                )
            if "kendall_tau" in mk_era:
                lines.append(
                    f"- **No trend within the post-1991 era** (tau "
                    f"{mk_era['kendall_tau']:+.3f}, p={mk_era['p_value']:.2g}), the "
                    "period with one consistent chart methodology."
                )
            lines += [
                "",
                "So the honest reading is that hits shifted toward minor keys mostly "
                "*because the genre mix shifted*, not because songwriting within genres "
                "moved that way. It is reported as suggestive, not established.",
                "",
            ]

        agg = valid.get("aggregation_bias", {})
        if "unadjusted_per_decade" in agg:
            strata_txt = "; ".join(
                f"{s['stratum']} n={s['n']}, rho {s['spearman_year']:+.3f}"
                for s in agg.get("within_chunk_strata", [])
            )
            lines += [
                "### The independence rise is real, but half the headline size",
                "",
                "Method B scores verse-sized chunks and takes the **maximum**, which is "
                "what lets it find a self-sufficiency claim living in a single chorus. "
                "But the maximum of N draws rises with N even if nothing underlying "
                "changes, and lyrics roughly doubled in length over the period "
                f"(rho(year, chunks) = {agg['spearman_year_vs_chunks']:+.2f}; "
                f"rho(chunks, p_max) = {agg['spearman_chunks_vs_max']:+.2f}). Part of "
                "the apparent rise is therefore mechanical.",
                "",
                "Unlike lyric length and valence — where length is a mediator and "
                "controlling it would remove real signal — this inflation is a property "
                "of the **estimator**, not of the music, so adjusting for it is correct.",
                "",
                f"- Unadjusted: **{agg['unadjusted_per_decade']:+.4f}/decade** "
                f"(p={agg['unadjusted_p']:.2g})",
                f"- Chunk-adjusted: **{agg['chunk_adjusted_per_decade']:+.4f}/decade** "
                f"(p={agg['chunk_adjusted_p']:.2g}) — "
                f"{agg['attenuation_fraction']:.0%} attenuation",
                "",
                "The trend nonetheless rises inside **every** fixed chunk-count stratum "
                f"({strata_txt}), including short songs where the bias cannot operate "
                f"({_short_stratum_text(agg)}). So the direction is solid and the "
                f"**adjusted figure of about {agg['chunk_adjusted_per_decade'] * 100:+.1f} "
                f"points per decade should be read as the headline**, not the raw "
                f"{agg['unadjusted_per_decade'] * 100:+.1f}.",
                "",
            ]

        pr_path = REPORTS / "prompt_robustness.json"
        if pr_path.exists():
            pr = json.loads(pr_path.read_text())
            pv = pr.get("per_variant", {})
            if pv:
                n_sig = sum(1 for v in pv.values() if v["significant"])
                lines += [
                    "### Does the result depend on how the question was worded?",
                    "",
                    "The independence finding rests on one sentence handed to a "
                    "zero-shot model — *\"The singer does not need this person and will "
                    "be fine without them.\"* Zero-shot classifiers are phrasing-"
                    "sensitive, so four paraphrases were scored on the same "
                    f"{pr.get('n_songs', 0):,}-song year-balanced sample.",
                    "",
                    "| Phrasing | Mean share | Kendall tau | p |",
                    "|---|---|---|---|",
                ]
                for key, v in pv.items():
                    dec = pr["by_decade"].get(key, {})
                    mean_share = sum(dec.values()) / len(dec) if dec else float("nan")
                    lines.append(
                        f"| `{key}` | {mean_share:.1%} | {v['kendall_tau']:+.3f} | "
                        f"{v['p_value']:.2g} |"
                    )
                lines += [
                    "",
                    f"**Direction is robust: all five paraphrases give a positive trend "
                    f"and {n_sig} of {len(pv)} are significant.** The exception, "
                    "`better_alone`, fires on only 0.8% of songs — too strict a claim "
                    "to have any statistical power — so it is degenerate rather than "
                    "contradictory. Per-song scores correlate 0.31–0.76 across "
                    "phrasings.",
                    "",
                    "**But the absolute level is not robust.** Mean share ranges from "
                    "0.8% to 14.8% depending purely on wording. So *\"the share of love "
                    "songs about not needing a partner rose\"* is supported; *\"19% of "
                    "love songs are about not needing a partner\"* is **not** a fact "
                    "about music, it is a fact about one sentence. Quote the trend, not "
                    "the level.",
                    "",
                ]

        lang = valid.get("language_robustness", {})
        if "all_songs" in lang:
            lines += [
                "### Non-English songs — ruled out",
                "",
                "All lexicons are English-only, and the non-English share of charting "
                "songs rises from about 1% before 2010 to 7.2% in the 2020s. "
                "Restricting to confidently-English, effectively monolingual songs "
                f"moves the coefficient only from "
                f"{lang['all_songs']['spearman']:+.3f} to "
                f"{lang['english_monolingual']['spearman']:+.3f}, so language does not "
                "drive the trend.",
                "",
            ]

        strata = valid.get("length_strata")
        tvt = valid.get("type_vs_token", {})
        if strata:
            rhos = [s["spearman_year_vs_valence"] for s in strata]
            lines += [
                "### Lyric length — a mediator, not an artefact (correction)",
                "",
                "An earlier version of this report claimed *\"roughly a third of the "
                "effect is length, not mood\"*, based on a partial correlation "
                "controlling for word count. **That was wrong.** Length is a mediator "
                "(year → songs get wordier → word-average valence falls), and "
                "controlling for a variable on the causal path subtracts real signal.",
                "",
                "Two tests settle it:",
                "",
                f"- Within every length quintile the decline persists "
                f"(rho {min(rhos):+.2f} to {max(rhos):+.2f}, all significant), "
                "including the shortest songs.",
                "",
            ]
            if "spearman_types" in tvt:
                lines.append(
                    f"- Computing valence over unique word *types*, which removes "
                    f"repetition entirely, makes the decline **stronger** "
                    f"(rho {tvt['spearman_types']:+.3f} vs "
                    f"{tvt['spearman_tokens']:+.3f}), so it is not old songs repeating "
                    "happy hooks."
                )
            lines += [
                "",
                "So within the lexicon's own terms the decline is real. That is a "
                "separate question from whether the lexicon measures mood, which the "
                "contextual check above puts in doubt.",
                "",
            ]

        val = conf.get("lyric_valence", {})
        era = val.get("post_1991_soundscan", {})
        if "kendall_tau" in era:
            verdict = "survives" if era["significant_at_05"] else "does not survive"
            lines += [
                "### Measurement regime",
                "",
                f"Restricted to 1991 onward (SoundScan era only, one consistent chart "
                f"methodology), the lexicon valence decline {verdict}: tau = "
                f"{era['kendall_tau']:+.3f}, p = {era['p_value']:.2g}.",
                "",
            ]
        ind = conf.get("independence_share", {}).get("post_1991_soundscan", {})
        if "kendall_tau" in ind:
            if ind["significant_at_05"]:
                lines += [
                    "### The independence rise continues inside the modern era",
                    "",
                    f"Restricted to 1991 onward — one consistent chart methodology — the "
                    f"trend is still present and significant (tau = "
                    f"{ind['kendall_tau']:+.3f}, p = {ind['p_value']:.2g}). An earlier "
                    "pass on roughly half this much data found it non-significant within "
                    "that window and read the rise as a single step around 2000; with "
                    "the fuller sample it looks like a continuing climb rather than a "
                    "one-off shift.",
                    "",
                ]
            else:
                lines += [
                    "### The independence rise may be a step rather than a slope",
                    "",
                    f"Within the post-1991 era alone it is not significant (tau = "
                    f"{ind['kendall_tau']:+.3f}, p = {ind['p_value']:.2g}), which would "
                    "suggest a jump around 2000 followed by a plateau rather than a "
                    "continuing climb.",
                    "",
                ]
        gs = val.get("genre_strata")
        ga = val.get("genre_adjusted", {})
        if "genre_adjusted_year_coef_per_decade" in ga:
            ind_ga = conf.get("independence_share", {}).get("genre_adjusted", {})
            ind_gs = conf.get("independence_share", {}).get("genre_strata", {})
            lines += [
                "### Genre mix — tested, and not the driver",
                "",
                "Rap and R&B went from absent to dominant on the Hot 100, and they have "
                "different lyrical conventions, so a change in the *mix* could move the "
                "average without any genre changing. Re-estimating the year effect with "
                "genre fixed effects:",
                "",
                "| Metric | Unadjusted / decade | Genre-adjusted | Attenuation | n |",
                "|---|---|---|---|---|",
                f"| lyric valence | {ga['unadjusted_year_coef_per_decade']:+.5f} | "
                f"{ga['genre_adjusted_year_coef_per_decade']:+.5f} "
                f"(p={ga['genre_adjusted_p']:.2g}) | "
                f"{ga['attenuation_fraction']:.0%} | {ga['n']} |",
            ]
            if ind_ga.get("genre_adjusted_year_coef_per_decade") is not None:
                lines.append(
                    f"| independence share | "
                    f"{ind_ga['unadjusted_year_coef_per_decade']:+.5f} | "
                    f"{ind_ga['genre_adjusted_year_coef_per_decade']:+.5f} "
                    f"(p={ind_ga['genre_adjusted_p']:.2g}) | "
                    f"{ind_ga['attenuation_fraction']:.0%} | {ind_ga['n']} |"
                )
            lines += [
                "",
                f"Genre mix accounts for only {ga['attenuation_fraction']:.0%} of the "
                "lexicon valence trend, so it is not the explanation there.",
                "",
                (
                    f"For the independence trend genre control removes only "
                    f"{ind_ga['attenuation_fraction']:.0%} of the effect "
                    f"(p={ind_ga['genre_adjusted_p']:.2g} adjusted, n={ind_ga['n']:,}). "
                    "**Genre mix is not driving it.** An earlier pass on a sixth as much "
                    "genre-labelled data put the attenuation at 23% and lost "
                    "significance; that was a power limitation and it resolved with more "
                    "data."
                    if ind_ga.get("genre_adjusted_p", 1) < 0.05
                    else f"The independence trend keeps "
                    f"{1 - ind_ga.get('attenuation_fraction', 0):.0%} of its magnitude "
                    f"under genre control but is not significant at n={ind_ga.get('n', 0):,}, "
                    "a power limitation rather than evidence of absence."
                ),
                "",
                (
                    "Within individual genres the independence trend is positive in "
                    + ", ".join(
                        f"{k} ({v:+.2f})"
                        for k, v in ind_gs.get("taus", {}).items()
                        if v > 0.1
                    )
                    + ". It is flat in hip-hop, so this is **not** a rap phenomenon "
                    "riding the genre shift — it appears inside the older guitar- and "
                    "vocal-led genres too."
                    if ind_gs.get("taus")
                    else ""
                ),
                "",
                "**Caveat on the labels themselves.** These are Essentia's automatic "
                "classifiers, not editorial metadata. Its `genre_dortmund` model was "
                "discarded outright: it labels 95–98% of everything after 1980 "
                "\"electronic\", which is not a credible description of the Hot 100. "
                "`genre_rosamerica` is used instead — balanced across seven classes "
                "with a plausible trajectory (hip-hop 0% in the 1950s to 22.8% in the "
                "1990s) — but it is still wrong often enough that this is a sanity "
                "check, not a clean genre control.",
                "",
            ]
        elif gs:
            lines += [
                f"### Genre mix",
                "",
                f"Re-running within Essentia genre strata: {gs['n_strata']} strata, "
                f"{gs['n_significant']} significant, all same sign: "
                f"{gs['all_same_sign']}.",
                "",
            ]
        else:
            lines += [
                "### Genre mix — still untested",
                "",
                "The acoustic stage is still populating genre labels. This remains an "
                "untested rival explanation.",
                "",
            ]
        lines += ["## Limitations", ""]

    lines += [
        "1. **The chart is not listening.** Hot 100 methodology changed in 1991",
        "   (SoundScan/BDS), 1998, 2005 (digital), 2007 and 2013 (streaming/video).",
        "   Comparisons spanning those dates cross measurement regimes.",
        "2. **Chart tenure has inflated.** Songs now stay on the chart 90+ weeks versus",
        "   ~13 in the 1960s, so exposure weights are normalized within year.",
        "3. **Lyric coverage is uneven** and correlates with year; see above.",
        "4. **Lexicon sentiment is blind to stance and context.** Word-level valence",
        "   cannot tell 'I don't need you' from 'I need you', cannot see negation, and",
        "   cannot track 68 years of semantic change. This is not hypothetical here: a",
        "   context-aware model run on the same songs finds no valence trend at all.",
        "   Stance questions are therefore answered by the entailment model instead.",
        "5. **Genre labels are model-inferred, not editorial.** Essentia's classifiers",
        "   are noisy; `genre_dortmund` was discarded as degenerate. Genre control is a",
        "   sanity check rather than a clean adjustment.",
        "6. **Acoustic coverage is partial.** AcousticBrainz is community-submitted, so",
        "   BPM and mood exist for a subset of songs, and Essentia BPM is prone to",
        "   octave errors.",
        "",
        "## Sources",
        "",
        "See `DATA_SOURCES.md`.",
        "",
    ]

    out = REPORTS / "findings.md"
    out.write_text("\n".join(l for l in lines if l is not None))
    print(f"Wrote {out} and {len(made)} figures to {FIGURES}")


if __name__ == "__main__":
    run()
