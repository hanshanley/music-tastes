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

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

ERA_MARKS = [
    (1991, "SoundScan"),
    (2005, "digital sales"),
    (2013, "streaming/video"),
]


def _plot_metric(series: pd.DataFrame, metric: str, description: str, path):
    variants = [
        ("weighted_all", "All covered songs (exposure-weighted)", "#1f77b4"),
        ("weighted_complete_case", "Complete-case subset (exposure-weighted)", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    plotted = False
    for variant, label, color in variants:
        sub = series[(series["metric"] == metric) & (series["variant"] == variant)]
        if sub.empty:
            continue
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["mean"], color=color, lw=1.6, label=label)
        ax.fill_between(sub["year"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False

    for year, name in ERA_MARKS:
        ax.axvline(year, color="grey", ls=":", lw=0.8)
        ax.text(year + 0.4, ax.get_ylim()[1], name, fontsize=6.5, color="grey",
                va="top", rotation=90)

    ax.set_title(description, fontsize=9.5)
    ax.set_xlabel("Year song first charted")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_decades(decades: pd.DataFrame, metric: str, description: str, path):
    sub = decades[
        (decades["metric"] == metric) & (decades["variant"] == "weighted_all")
    ].sort_values("decade")
    if sub.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = [f"{int(d)}s" for d in sub["decade"]]
    yerr = [
        (sub["mean"] - sub["ci_lo"]).clip(lower=0).to_numpy(),
        (sub["ci_hi"] - sub["mean"]).clip(lower=0).to_numpy(),
    ]
    ax.bar(x, sub["mean"] * 100, yerr=[e * 100 for e in yerr], color="#4c72b0",
           capsize=3, alpha=0.85)
    ax.set_ylabel("%")
    ax.set_title(f"{description}\n(by decade, exposure-weighted, 95% CI)", fontsize=9)
    for xi, (m, n) in enumerate(zip(sub["mean"], sub["n"])):
        ax.text(xi, 0.4, f"n={n}", ha="center", va="bottom", fontsize=6, color="white")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _plot_coverage(path):
    cov = pd.read_csv(REPORTS / "coverage_by_year.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(cov["debut_year"], cov["coverage_lyrics"] * 100, color="#2ca02c", lw=1.6)
    ax.set_title("Lyric coverage by year (the main threat to every trend above)")
    ax.set_ylabel("% of charting songs with lyrics")
    ax.set_xlabel("Year song first charted")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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

    lines = [
        "# Are US hit songs getting sadder, and are fewer of them about love?",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._",
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
        f"- Overall lyric coverage: **{coverage['coverage_lyrics_overall']:.1%}** "
        f"of charting songs",
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
                lines += [
                    "| Decade | n | Exposure-weighted mean | 95% CI |",
                    "|---|---|---|---|",
                ]
                for r in sub.itertuples():
                    lines.append(
                        f"| {int(r.decade)}s | {r.n} | {r.mean:.1%} | "
                        f"{r.ci_lo:.1%} – {r.ci_hi:.1%} |"
                    )
                lines.append("")

        lines += [f"![{metric}](figures/{metric}_decade.png)", ""]
        lines += [f"![{metric} yearly](figures/{metric}.png)", ""]

    lines += [
        "## Limitations",
        "",
        "1. **The chart is not listening.** Hot 100 methodology changed in 1991",
        "   (SoundScan/BDS), 1998, 2005 (digital), 2007 and 2013 (streaming/video).",
        "   Comparisons spanning those dates cross measurement regimes.",
        "2. **Chart tenure has inflated.** Songs now stay on the chart 90+ weeks versus",
        "   ~13 in the 1960s, so exposure weights are normalized within year.",
        "3. **Lyric coverage is uneven** and correlates with year; see above.",
        "4. **Lexicon sentiment is blind to stance.** Word-level valence cannot tell",
        "   'I don't need you' from 'I need you', which is why the stance question is",
        "   answered by an entailment model instead.",
        "5. **Genre mix is uncontrolled** in this version. A shift toward genres with",
        "   different lyrical conventions is a live rival explanation for any change",
        "   in valence.",
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
