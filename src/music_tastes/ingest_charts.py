"""Stage 1: ingest weekly Billboard Hot 100 charts, 1958-08-04 to present.

Primary source is the mhollingshead/billboard-hot-100 archive, which publishes one
JSON document per chart week. We cross-check it against an independent archive
(utdata/rwd-billboard-data, maintained at UT Austin) and record every disagreement
rather than silently preferring one source.

Billboard's Hot 100 is compiled by Billboard from Luminate (formerly Nielsen
SoundScan/BDS) sales, streaming and radio airplay data. Neither archive is an official
Billboard product; both are third-party mirrors, which is exactly why we cross-check.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import pandas as pd

from .http import get
from .paths import INTERIM, RAW

PRIMARY_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"
SECONDARY_URL = (
    "https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv"
)


def _fetch_primary() -> list[dict]:
    """Download the full weekly archive, caching the raw payload on disk."""
    local = RAW / "billboard_hot100_all.json"
    if local.exists():
        print(f"Using cached primary archive: {local}")
        return json.loads(local.read_text())

    print(f"Downloading primary archive from {PRIMARY_URL} ...")
    # This payload is ~90 MB, so it bypasses the small-response JSON cache.
    resp = get(PRIMARY_URL, namespace="billboard", timeout=300, use_cache=False)
    if resp.status != 200:
        raise SystemExit(f"Primary archive fetch failed with HTTP {resp.status}")
    local.write_text(resp.text)
    (RAW / "billboard_hot100_all.provenance.json").write_text(
        json.dumps(
            {
                "url": PRIMARY_URL,
                "retrieved_at": resp.retrieved_at,
                "source": "mhollingshead/billboard-hot-100 (GitHub)",
                "upstream": "Billboard Hot 100, compiled by Billboard from Luminate data",
            },
            indent=2,
        )
    )
    return json.loads(resp.text)


def _normalize_primary(weeks: list[dict]) -> pd.DataFrame:
    """Flatten the per-week documents into one row per (chart_week, rank)."""
    rows = []
    for week in weeks:
        chart_week = week["date"]
        for entry in week.get("data", []):
            rows.append(
                {
                    "chart_week": chart_week,
                    "rank": entry.get("this_week"),
                    "title": entry.get("song"),
                    "artist": entry.get("artist"),
                    "last_week": entry.get("last_week"),
                    "peak_rank": entry.get("peak_position"),
                    "weeks_on_chart": entry.get("weeks_on_chart"),
                }
            )

    df = pd.DataFrame(rows)
    df["chart_week"] = pd.to_datetime(df["chart_week"]).dt.date
    for col in ("rank", "last_week", "peak_rank", "weeks_on_chart"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.dropna(subset=["title", "artist", "rank"])
    df["title"] = df["title"].str.strip()
    df["artist"] = df["artist"].str.strip()
    return df.sort_values(["chart_week", "rank"]).reset_index(drop=True)


def _fetch_secondary() -> pd.DataFrame | None:
    """Fetch the independent archive used for cross-validation. Optional."""
    try:
        resp = get(SECONDARY_URL, namespace="billboard", timeout=300, use_cache=False)
    except RuntimeError as exc:
        print(f"  secondary archive unavailable ({exc}); skipping cross-check")
        return None
    if resp.status != 200:
        print(f"  secondary archive returned HTTP {resp.status}; skipping cross-check")
        return None

    # Record the cross-check source with the same rigour as the primary archive, so
    # a disagreement between them can be traced to two dated retrievals.
    (RAW / "billboard_hot100_secondary.provenance.json").write_text(
        json.dumps(
            {
                "url": SECONDARY_URL,
                "retrieved_at": resp.retrieved_at,
                "source": "utdata/rwd-billboard-data (UT Austin, GitHub)",
                "upstream": "Billboard Hot 100, compiled by Billboard from Luminate data",
                "role": "independent cross-check of the primary archive",
            },
            indent=2,
        )
    )

    df = pd.read_csv(io.StringIO(resp.text), quoting=csv.QUOTE_MINIMAL)
    rename = {
        "chart_week": "chart_week",
        "chart_date": "chart_week",
        "current_week": "rank",
        "this_week": "rank",
        "title": "title",
        "performer": "artist",
        "artist": "artist",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    missing = {"chart_week", "rank", "title", "artist"} - set(df.columns)
    if missing:
        print(f"  secondary archive has unexpected columns (missing {missing}); skipping")
        return None

    df["chart_week"] = pd.to_datetime(df["chart_week"], errors="coerce").dt.date
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    return df[["chart_week", "rank", "title", "artist"]].dropna()


def _cross_check(primary: pd.DataFrame, secondary: pd.DataFrame) -> dict:
    """Compare the two archives on the weeks they share.

    We compare titles case-insensitively with whitespace collapsed, because the two
    archives differ in punctuation and casing conventions. Anything still mismatched
    is reported, not resolved.
    """

    def key(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["title_key"] = out["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        return out.set_index(["chart_week", "rank"])["title_key"]

    shared_weeks = sorted(set(primary["chart_week"]) & set(secondary["chart_week"]))
    p = key(primary[primary["chart_week"].isin(shared_weeks)])
    s = key(secondary[secondary["chart_week"].isin(shared_weeks)])

    p = p[~p.index.duplicated()]
    s = s[~s.index.duplicated()]
    joined = pd.concat([p.rename("primary"), s.rename("secondary")], axis=1, join="inner")
    mismatches = joined[joined["primary"] != joined["secondary"]]

    report = {
        "shared_weeks": len(shared_weeks),
        "primary_only_weeks": len(set(primary["chart_week"]) - set(secondary["chart_week"])),
        "secondary_only_weeks": len(set(secondary["chart_week"]) - set(primary["chart_week"])),
        "compared_positions": int(len(joined)),
        "mismatched_positions": int(len(mismatches)),
        "mismatch_rate": round(len(mismatches) / len(joined), 6) if len(joined) else None,
        "examples": mismatches.head(25).reset_index().to_dict("records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for row in report["examples"]:
        row["chart_week"] = str(row["chart_week"])
        row["rank"] = int(row["rank"])
    return report


def run() -> pd.DataFrame:
    weeks = _fetch_primary()
    df = _normalize_primary(weeks)

    out = INTERIM / "chart_entries.parquet"
    df.to_parquet(out, index=False)

    print(f"\nIngested {len(df):,} chart positions")
    print(f"  weeks:  {df['chart_week'].nunique():,} "
          f"({df['chart_week'].min()} → {df['chart_week'].max()})")
    print(f"  unique (title, artist) pairs: {df.groupby(['title', 'artist']).ngroups:,}")
    print(f"  wrote {out}")

    print("\nCross-checking against the independent UT Austin archive ...")
    secondary = _fetch_secondary()
    if secondary is not None:
        report = _cross_check(df, secondary)
        path = INTERIM / "chart_crosscheck.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  compared {report['compared_positions']:,} shared positions")
        print(f"  mismatches: {report['mismatched_positions']:,} "
              f"({(report['mismatch_rate'] or 0) * 100:.4f}%)")
        print(f"  wrote {path}")

    return df


if __name__ == "__main__":
    run()
