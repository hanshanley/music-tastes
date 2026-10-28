"""Command line entry point tying the pipeline stages together.

Stages are ordered and resumable. Every stage caches its network and model work on
disk, so re-running a stage that has already completed costs nothing and a long run
can be interrupted and restarted freely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STAGES = [
    ("ingest-charts", "Download and normalize weekly Hot 100 charts, with cross-check"),
    ("resolve-songs", "Collapse chart rows into unique songs"),
    ("exposure", "Compute exposure weights and the chart-methodology era table"),
    ("lexicons", "Download NRC VAD, NRC EmoLex and VADER"),
    ("fetch-lyrics", "Fetch lyrics into the gitignored local cache"),
    ("enrich-acoustic", "MusicBrainz -> AcousticBrainz BPM, key and Essentia mood"),
    ("features-a", "Method A: lexicon and embedding-anchor features"),
    ("stance-b", "Method B: local zero-shot NLI stance classification"),
    ("language", "Language ID and code-switching measurement"),
    ("validity", "Construct-validity checks on the sentiment measures"),
    ("prompt-robustness", "Does the stance result survive rewording the hypothesis?"),
    ("stance-composition", "Full stance mix within relationship songs over time"),
    ("confounds", "Rival explanations: genre, era, lyric length"),
    ("coverage", "Coverage audit (gates every trend claim)"),
    ("trends", "Year-level trends with bootstrap CIs and rank-based tests"),
    ("vad-robustness", "Compare NRC VAD v1/v2.1, phrases, repetition and polar-word scoring"),
    ("gold-set", "Validate both classifiers against hand-labelled songs"),
    ("exhibit", "Face-validity exhibit: songs driving the stance result"),
    ("report", "Figures and the written findings document"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-tastes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {n:15} {d}" for n, d in STAGES),
    )
    parser.add_argument(
        "stage", choices=[n for n, _ in STAGES] + [
            "all", "update", "status", "review-packet", "site", "page-plots",
        ]
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--as-of")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-after-days", type=float, default=7)
    parser.add_argument("--per-decade", type=int, default=20)
    parser.add_argument("--lrclib", action="store_true")
    parser.add_argument("--vad-archive", type=Path, help="Local NRC VAD v2.1 ZIP (vad-robustness)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.retry_after_days < 0:
        parser.error("--workers must be positive and --retry-after-days nonnegative")
    if args.offline and args.stage not in {"ingest-charts", "update", "status", "site", "vad-robustness"}:
        parser.error("--offline is supported by ingest-charts, update and vad-robustness")
    if args.vad_archive and args.stage != "vad-robustness":
        parser.error("--vad-archive is supported by vad-robustness")
    if args.stage in {"update", "status"}:
        from . import refresh

        if args.stage == "status":
            refresh.status()
        else:
            refresh.run(
                limit=args.limit, workers=args.workers, offline=args.offline,
                use_api=not args.no_api, as_of=args.as_of,
                retry_after_days=args.retry_after_days,
            )
        return 0
    if args.stage == "review-packet":
        from .gold_set import prepare_review

        prepare_review(per_decade=args.per_decade)
        return 0
    if args.stage == "site":
        from .site import run

        run()
        return 0
    if args.stage == "page-plots":
        from .page_plots import export

        export()
        return 0

    stages = [n for n, _ in STAGES] if args.stage == "all" else [args.stage]

    for stage in stages:
        print(f"\n{'=' * 70}\n{stage}\n{'=' * 70}")
        if stage == "ingest-charts":
            from . import ingest_charts

            ingest_charts.run(refresh=args.refresh, offline=args.offline, as_of=args.as_of)
        elif stage == "resolve-songs":
            from . import resolve_songs

            resolve_songs.run()
        elif stage == "exposure":
            from . import exposure

            exposure.run()
        elif stage == "lexicons":
            from . import lexicons

            lexicons.run()
        elif stage == "fetch-lyrics":
            from . import fetch_lyrics

            fetch_lyrics.run(
                limit=args.limit, workers=args.workers, use_api=not args.no_api,
                retry_failed=args.retry_failed, retry_after_days=args.retry_after_days,
                use_lrclib=args.lrclib,
            )
        elif stage == "enrich-acoustic":
            from . import enrich_acoustic

            enrich_acoustic.run(limit=args.limit)
        elif stage == "features-a":
            from . import lyrics_features

            lyrics_features.run(limit=args.limit)
        elif stage == "stance-b":
            from . import stance_nli

            stance_nli.run(limit=args.limit)
        elif stage == "language":
            from . import language

            language.run(limit=args.limit)
        elif stage == "validity":
            from . import validity

            validity.run()
        elif stage == "prompt-robustness":
            from . import prompt_robustness

            prompt_robustness.run()
        elif stage == "stance-composition":
            from . import stance_composition

            stance_composition.run()
        elif stage == "confounds":
            from . import confounds

            confounds.run()
        elif stage == "coverage":
            from . import coverage

            coverage.run()
        elif stage == "trends":
            from . import analysis_trends

            analysis_trends.run()
        elif stage == "vad-robustness":
            from . import vad_robustness

            vad_robustness.run(archive=args.vad_archive, offline=args.offline)
        elif stage == "gold-set":
            from . import gold_set

            gold_set.run()
        elif stage == "exhibit":
            from . import exhibit

            exhibit.run()
        elif stage == "report":
            from . import report

            report.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
