"""Command line entry point tying the pipeline stages together.

Stages are ordered and resumable. Every stage caches its network and model work on
disk, so re-running a stage that has already completed costs nothing and a long run
can be interrupted and restarted freely.
"""

from __future__ import annotations

import argparse

STAGES = [
    ("ingest-charts", "Download and normalize weekly Hot 100 charts, with cross-check"),
    ("resolve-songs", "Collapse chart rows into unique songs"),
    ("exposure", "Compute exposure weights and the chart-methodology era table"),
    ("lexicons", "Download NRC VAD, NRC EmoLex and VADER"),
    ("fetch-lyrics", "Fetch lyrics into the gitignored local cache"),
    ("features-a", "Method A: lexicon and embedding-anchor features"),
    ("stance-b", "Method B: local zero-shot NLI stance classification"),
    ("coverage", "Coverage audit (gates every trend claim)"),
    ("trends", "Year-level trends with bootstrap CIs and rank-based tests"),
    ("report", "Figures and the written findings document"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="music-tastes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages:\n" + "\n".join(f"  {n:15} {d}" for n, d in STAGES),
    )
    parser.add_argument("stage", choices=[n for n, _ in STAGES] + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    stages = [n for n, _ in STAGES] if args.stage == "all" else [args.stage]

    for stage in stages:
        print(f"\n{'=' * 70}\n{stage}\n{'=' * 70}")
        if stage == "ingest-charts":
            from . import ingest_charts

            ingest_charts.run()
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

            fetch_lyrics.run(limit=args.limit, workers=args.workers)
        elif stage == "features-a":
            from . import lyrics_features

            lyrics_features.run(limit=args.limit)
        elif stage == "stance-b":
            from . import stance_nli

            stance_nli.run(limit=args.limit)
        elif stage == "coverage":
            from . import coverage

            coverage.run()
        elif stage == "trends":
            from . import analysis_trends

            analysis_trends.run()
        elif stage == "report":
            from . import report

            report.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
