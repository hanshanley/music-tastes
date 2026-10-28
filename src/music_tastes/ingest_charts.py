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
from .paths import INTERIM, RAW, atomic_json, atomic_text, atomic_parquet

PRIMARY_URL = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json"
SECONDARY_URL = (
    "https://raw.githubusercontent.com/utdata/rwd-billboard-data/main/data-out/hot-100-current.csv"
)


def _fetch_primary(refresh: bool = False, offline: bool = False) -> list[dict]:
    """Download the full weekly archive, caching the raw payload on disk."""
    local = RAW / "billboard_hot100_all.json"
    if local.exists() and (not refresh or offline):
        print(f"Using cached primary archive: {local}")
        return json.loads(local.read_text())
    if offline:
        raise SystemExit("No cached chart archive. Run ingest-charts online first.")

    print(f"Downloading primary archive from {PRIMARY_URL} ...")
    # This payload is ~90 MB, so it bypasses the small-response JSON cache.
    resp = get(PRIMARY_URL, namespace="billboard", timeout=300, use_cache=False)
    if resp.status != 200:
        raise SystemExit(f"Primary archive fetch failed with HTTP {resp.status}")
    weeks = json.loads(resp.text)
    validate_charts(_normalize_primary(weeks))
    atomic_text(local, resp.text)
    atomic_json(RAW / "billboard_hot100_all.provenance.json", {
                "url": PRIMARY_URL,
                "retrieved_at": resp.retrieved_at,
                "source": "mhollingshead/billboard-hot-100 (GitHub)",
                "upstream": "Billboard Hot 100, compiled by Billboard from Luminate data",
                "chart_through": max(w["date"] for w in weeks),
            })
    return weeks


def validate_charts(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("Chart archive is empty")
    if df.duplicated(["chart_week", "rank"]).any():
        raise ValueError("Duplicate chart-week/rank positions")
    if not df["rank"].between(1, 100).all():
        raise ValueError("Chart ranks must be between 1 and 100")
    dates = pd.Series(sorted(df["chart_week"].unique()))
    gaps = dates.diff().dt.days
    # Both archives switch from Monday issue dates to Saturday at this boundary.
    missing = [
        str(dates.iloc[i]) for i in gaps[gaps > 10].index
        if not (str(dates.iloc[i - 1]) == "1961-12-25" and str(dates.iloc[i]) == "1962-01-06")
    ]
    sizes = df.groupby("chart_week").size()
    incomplete = {str(k): int(v) for k, v in sizes[sizes != 100].items()}
    return {
        "chart_from": str(dates.iloc[0]),
        "chart_through": str(dates.iloc[-1]),
        "weeks": len(dates),
        "positions": len(df),
        "gaps_before": missing,
        "non_100_position_weeks": incomplete,
    }


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
        for column in ("title", "artist"):
            out[column] = out[column].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        return out.set_index(["chart_week", "rank"])[["title", "artist"]]

    shared_weeks = sorted(set(primary["chart_week"]) & set(secondary["chart_week"]))
    p = key(primary[primary["chart_week"].isin(shared_weeks)])
    s = key(secondary[secondary["chart_week"].isin(shared_weeks)])

    p = p[~p.index.duplicated()]
    s = s[~s.index.duplicated()]
    joined = p.join(s, how="outer", lsuffix="_primary", rsuffix="_secondary")
    present = joined.dropna()
    mismatches = present[
        (present["title_primary"] != present["title_secondary"])
        | (present["artist_primary"] != present["artist_secondary"])
    ]

    report = {
        "shared_weeks": len(shared_weeks),
        "primary_only_weeks": len(set(primary["chart_week"]) - set(secondary["chart_week"])),
        "secondary_only_weeks": len(set(secondary["chart_week"]) - set(primary["chart_week"])),
        "compared_positions": int(len(present)),
        "missing_primary_positions": int(joined["title_primary"].isna().sum()),
        "missing_secondary_positions": int(joined["title_secondary"].isna().sum()),
        "mismatched_positions": int(len(mismatches)),
        "mismatch_rate": round(len(mismatches) / len(present), 6) if len(present) else None,
        "examples": mismatches.head(25).reset_index().to_dict("records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for row in report["examples"]:
        row["chart_week"] = str(row["chart_week"])
        row["rank"] = int(row["rank"])
    return report


def run(refresh: bool = False, offline: bool = False, as_of: str | None = None) -> pd.DataFrame:
    weeks = _fetch_primary(refresh=refresh, offline=offline)
    df = _normalize_primary(weeks)
    print("\nCross-checking against the independent UT Austin archive ...")
    secondary_path = RAW / "billboard_hot100_secondary.parquet"
    secondary = (
        pd.read_parquet(secondary_path) if offline and secondary_path.exists()
        else None if offline else _fetch_secondary()
    )
    df["chart_source"] = "mhollingshead"
    if secondary is not None:
        atomic_parquet(secondary_path, secondary)
        report = _cross_check(df, secondary)
        path = INTERIM / "chart_crosscheck.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  compared {report['compared_positions']:,} shared positions")
        print(f"  mismatches: {report['mismatched_positions']:,} "
              f"({(report['mismatch_rate'] or 0) * 100:.4f}%)")
        print(f"  wrote {path}")
        extra = secondary[~secondary["chart_week"].isin(df["chart_week"])].copy()
        if not extra.empty:
            validate_charts(extra)
            sizes = extra.groupby("chart_week").size()
            if (sizes != 100).any():
                raise SystemExit("Secondary-only chart weeks are incomplete; refusing to append")
            extra["chart_source"] = "utdata"
            df = pd.concat([df, extra], ignore_index=True)
    else:
        atomic_json(INTERIM / "chart_crosscheck.json", {
            "status": "not_checked", "reason": "offline" if offline else "source_unavailable"
        })
    if as_of:
        df = df[df["chart_week"] <= pd.Timestamp(as_of).date()].copy()
    df = df.sort_values(["chart_week", "rank"]).reset_index(drop=True)
    audit = validate_charts(df)
    audit["sources"] = df.groupby("chart_source").size().astype(int).to_dict()
    atomic_json(INTERIM / "chart_integrity.json", audit)
    if audit["gaps_before"]:
        raise SystemExit(f"Missing chart weeks before: {audit['gaps_before']}")
    out = INTERIM / "chart_entries.parquet"
    atomic_parquet(out, df)
    print(f"\nIngested {len(df):,} positions, {df['chart_week'].nunique():,} weeks")
    print(f"  {audit['chart_from']} through {audit['chart_through']} -> {out}")

    return df


if __name__ == "__main__":
    run()
