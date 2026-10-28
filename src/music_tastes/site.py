"""Build a portable, static evidence page. No deployment or raw-lyric exports."""

from __future__ import annotations

import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .cache import fingerprint
from .paths import REPORTS, ROOT, atomic_json, atomic_text
from .report import summary_for


def run(output: Path | None = None) -> Path:
    output = Path(output) if output else ROOT / "site"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    generated_path = assets / ".generated.json"
    try:
        old_generated = json.loads(generated_path.read_text())
    except (OSError, ValueError):
        old_generated = []
    generated = set()

    def copy_asset(path, name=None):
        name = name or path.name
        shutil.copy2(path, assets / name)
        generated.add(name)

    manifest_path = REPORTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    verified = (
        manifest.get("status") == "complete" and bool(manifest.get("chart_through"))
        and manifest.get("output_signature") == fingerprint(
            {}, sorted(path for path in REPORTS.rglob("*")
                       if path.is_file() and path != manifest_path)
        )
    )
    findings = (REPORTS / "findings.md").read_text() if (REPORTS / "findings.md").exists() else ""
    date = re.search(r"Generated (\d{4}-\d{2}-\d{2})", findings)
    status = (
        f"Complete analysis through chart issue {manifest['chart_through']}"
        if verified else "Unverified historical snapshot. A full refresh is not verified."
    )
    if not verified and date:
        status += f" Findings dated {date.group(1)}; newer chart ingestion is not a newer analysis."
    trends_path = REPORTS / "trend_results.json"
    trends = json.loads(trends_path.read_text()) if trends_path.exists() else {}
    selected = [
        ("relationship_share", "How many hits concern relationships?",
         "Romantic or sexual relationships, including breakups and casual sex."),
        ("independence_share", "What changes within relationship songs?",
         "Independence is measured only among relationship-positive songs."),
        ("lyric_valence", "Do the words feel less positive?",
         "A lexical measure with length, language, and coverage limitations."),
        ("nrc_vad_robustness", "Does a larger emotion lexicon change the picture?",
         "NRC VAD v1 and v2.1 on identical songs. Valence, arousal and dominance "
         "are separate word-rating dimensions, not a sadness classifier."),
        ("coverage_by_year", "What is missing?",
         "Song-count and chart-exposure coverage by debut year."),
    ]
    sections = []
    for metric, title, caption in selected:
        image = REPORTS / "figures" / f"{metric}.png"
        if not image.exists():
            continue
        copy_asset(image)
        claim = summary_for(metric, trends) if verified and metric in trends else ""
        if verified and metric == "nrc_vad_robustness":
            statistics_path = REPORTS / "nrc_vad_statistics.json"
            statistics = json.loads(statistics_path.read_text()) if statistics_path.exists() else {}
            parts = []
            for row in statistics.get("tests", []):
                if row["is_primary"] and row["method"] == "newey_west" and row["window_years"] == 5:
                    direction = "increases" if row["slope_per_decade"] > 0 else "decreases"
                    verdict = direction if row["reject_primary_05"] else "no clear linear trend"
                    parts.append(f"{row['dimension']}: {verdict} (p={row['p_holm_primary']:.2g})")
            if parts:
                claim = "Completed-year Newey-West tests, Holm-adjusted: " + "; ".join(parts) + "."
        figcaption = f"<figcaption>{html.escape(claim)}</figcaption>" if claim else ""
        sections.append(
            f'<section class="evidence" id="{metric}"><div class="section-head">'
            f'<h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p></div>'
            f'<figure><a href="assets/{image.name}" '
            f'aria-label="View full-size chart: {html.escape(title)}">'
            f'<img src="assets/{image.name}" alt="{html.escape(title)} '
            f'Historical debut-year series; see downloadable data." loading="lazy"></a>'
            f'{figcaption}</figure></section>'
        )
    downloads = []
    for path in sorted(REPORTS.glob("*.csv")):
        copy_asset(path)
        downloads.append(
            f'<li><a href="assets/{path.name}" download>{html.escape(path.stem.replace("_", " "))}</a></li>'
        )
    public_json = [(path, path.name) for path in sorted(REPORTS.glob("*.json"))]
    public_json.append((REPORTS / "gold" / "validation.json", "gold_validation.json"))
    for path, name in public_json:
        if path.exists():
            copy_asset(path, name)
            downloads.append(f'<li><a href="assets/{name}" download>{name}</a></li>')
    for name in ("METHODS.md", "DATA_SOURCES.md"):
        path = ROOT / name
        if path.exists():
            copy_asset(path)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    freshness = f"Preview built {built}."
    if verified:
        freshness = f"Analysis completed {manifest.get('completed_at', 'unrecorded')}. " + freshness
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A reproducible retrospective of relationships and mood in US hit songs.">
  <title>Music tastes | Relationships and mood in US hits</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
<a class="skip" href="#main">Skip to evidence</a>
<header><a class="wordmark" href="#main">Music tastes</a>
<nav aria-label="Main"><a href="#evidence">Evidence</a><a href="#methods">Methods</a>
<a href="#downloads">Data</a></nav></header>
<main id="main">
<section class="intro">
<p class="eyebrow">Billboard Hot 100 / Retrospective</p>
<h1>Have hit songs changed<br>how they talk about relationships?</h1>
<p class="lede">A retrospective study of what US chart songs are about, how their lyrics feel,
and how much the answer depends on what we can measure.</p>
<p class="status" role="note">{html.escape(status)}</p>
<p class="preview">{html.escape(freshness)}</p>
<p class="preview">A fixed historical snapshot, not an automatically updated service.</p>
<p class="preview">Local design preview. Main-site integration and deployment are separate work.</p>
</section>
<div id="evidence">{"".join(sections)}</div>
<section id="methods" class="methods"><h2>What counts as a relationship song?</h2>
<p>A local natural-language inference model tests whether a song concerns a
<strong>romantic or sexual relationship</strong>. It reads the first 380 cleaned words;
scores strictly above 0.5 count as positive. This is not a keyword count, a calibrated
probability, or a measure of family, friendship, spiritual love, or self-love.</p>
<p>Stance is a separate test over lyric chunks. Independence and heartbreak shares
are conditional on a positive relationship label. Missing lyrics and failed scores
are not negative labels.</p>
<p>Legacy validation was small and its annotation provenance was unaudited. A larger,
independently reviewed sample is required before stronger accuracy claims.
The historical trend groups songs by debut year, not the year audiences heard them.
Coverage and partial-period checks are essential; associations are not causal.</p>
<p><a href="assets/METHODS.md">Full method and annotation rubric</a> /
<a href="assets/DATA_SOURCES.md">Source notes</a></p></section>
<section id="downloads"><h2>Follow the evidence</h2>
<p>Complete available aggregate series, with estimates, sample sizes, and confidence
intervals where produced. No raw lyrics are distributed.</p><ul class="downloads">{"".join(downloads)}</ul></section>
</main>
<footer><a href="https://github.com/hanshanley/music-tastes">Source code</a> /
Billboard / Luminate chart data via independent public archives.
Lyrics and model features are processed locally. No website deployment is performed by this build.</footer>
</body></html>
"""
    stylesheet = ROOT / "site" / "style.css"
    if output.resolve() != stylesheet.parent.resolve():
        shutil.copy2(stylesheet, output / "style.css")
    atomic_text(output / "index.html", document)
    if isinstance(old_generated, list):
        for name in old_generated:
            if (
                isinstance(name, str) and Path(name).name == name and name not in generated
                and (assets / name).is_file()
            ):
                (assets / name).unlink()
    atomic_json(generated_path, sorted(generated))
    print(f"Static preview: {output / 'index.html'}")
    return output / "index.html"
