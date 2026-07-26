"""Stage 3: exposure weights and Hot 100 methodology eras.

A song that sat at #1 for ten weeks was heard far more than one that scraped #98 for a
single week, so unweighted song averages answer a different question than "what did
America actually hear". We compute several weightings and carry all of them through the
analysis, because the choice of weight is itself an analytical assumption.

Weights (all per song):
  points        sum over charted weeks of (101 - rank). The standard Billboard-style
                inverse-rank score; a #1 week is worth 100, a #100 week is worth 1.
  log_points    sum over charted weeks of log2(101 - rank + 1), which compresses the
                gap between the very top and the middle of the chart.
  weeks         number of charted weeks, ignoring position.
  unweighted    1 per song, regardless of performance.

The era table records the dates on which Billboard changed what the Hot 100 measures.
Any comparison spanning these dates is a comparison across measurement regimes, not a
clean like-for-like, and every chart in the final report cites it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import DERIVED, INTERIM

# Documented changes to Hot 100 methodology. Dates are the week the change took effect.
ERAS = [
    {
        "start": "1958-08-04",
        "end": "1991-11-29",
        "name": "Survey era",
        "measurement": "Store and radio surveys reported by hand; positions subject to "
                       "label influence and reporting bias.",
    },
    {
        "start": "1991-11-30",
        "end": "1998-12-04",
        "name": "SoundScan/BDS era",
        "measurement": "Point-of-sale scanning (Nielsen SoundScan) and monitored airplay "
                       "(Broadcast Data Systems) replace surveys.",
    },
    {
        "start": "1998-12-05",
        "end": "2005-02-11",
        "name": "Airplay-eligible era",
        "measurement": "Songs no longer need a commercial single release to chart; "
                       "airplay-only tracks become eligible.",
    },
    {
        "start": "2005-02-12",
        "end": "2007-08-10",
        "name": "Digital sales era",
        "measurement": "Paid digital downloads incorporated.",
    },
    {
        "start": "2007-08-11",
        "end": "2012-03-02",
        "name": "Early streaming era",
        "measurement": "Online streaming begins to count toward the chart.",
    },
    {
        "start": "2012-03-03",
        "end": "2013-02-19",
        "name": "Expanded streaming era",
        "measurement": "On-demand audio streaming weighted more heavily.",
    },
    {
        "start": "2013-02-20",
        "end": "2099-12-31",
        "name": "Video/streaming era",
        "measurement": "YouTube and video streams incorporated; streaming dominates the "
                       "formula for most genres.",
    },
]


def label_era(chart_week: pd.Series) -> pd.Series:
    d = pd.to_datetime(chart_week)
    out = pd.Series(pd.NA, index=d.index, dtype="object")
    for era in ERAS:
        mask = (d >= pd.Timestamp(era["start"])) & (d <= pd.Timestamp(era["end"]))
        out[mask] = era["name"]
    return out


def run() -> pd.DataFrame:
    entries = pd.read_parquet(INTERIM / "chart_entries_resolved.parquet")
    songs = pd.read_parquet(INTERIM / "songs.parquet")

    entries["chart_year"] = pd.to_datetime(entries["chart_week"]).dt.year
    entries["era"] = label_era(entries["chart_week"])
    entries["points"] = 101 - entries["rank"].astype(int)
    entries["log_points"] = np.log2(entries["points"] + 1)

    # Song-level totals across the song's entire chart life.
    song_weights = (
        entries.groupby("song_id")
        .agg(
            points=("points", "sum"),
            log_points=("log_points", "sum"),
            weeks=("chart_week", "nunique"),
            peak_rank=("rank", "min"),
            n_chart_runs=("chart_year", "nunique"),
        )
        .reset_index()
    )
    song_weights["unweighted"] = 1.0

    # Song-year totals, so a song charting across a year boundary contributes its
    # exposure to the correct year rather than only to its debut year.
    song_year = (
        entries.groupby(["song_id", "chart_year"])
        .agg(
            points=("points", "sum"),
            log_points=("log_points", "sum"),
            weeks=("chart_week", "nunique"),
            best_rank=("rank", "min"),
        )
        .reset_index()
    )
    song_year["unweighted"] = 1.0

    songs = songs.merge(song_weights, on="song_id", how="left", suffixes=("", "_w"))

    songs.to_parquet(DERIVED / "songs_weighted.parquet", index=False)
    song_year.to_parquet(DERIVED / "song_year_exposure.parquet", index=False)
    (DERIVED / "chart_eras.json").write_text(json.dumps(ERAS, indent=2))

    print(f"\nComputed exposure weights for {len(songs):,} songs")
    print(f"  song-year exposure rows: {len(song_year):,}")
    print("\n  Top songs by total chart points:")
    top = songs.nlargest(8, "points")
    for r in top.itertuples():
        print(f"    {r.points:>6,} pts  {r.weeks:>3}wk  "
              f"{r.title_display[:38]:38} — {r.artist_display[:26]}")

    print("\n  Songs debuting and total chart points by era:")
    era_counts = entries.groupby("era").agg(
        weeks=("chart_week", "nunique"), points=("points", "sum")
    )
    for era in ERAS:
        if era["name"] in era_counts.index:
            row = era_counts.loc[era["name"]]
            print(f"    {era['name']:24} {era['start']}→{era['end'][:4]}  "
                  f"{row['weeks']:>5,} weeks")
    return songs


if __name__ == "__main__":
    run()
