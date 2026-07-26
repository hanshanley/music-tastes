"""Face-validity exhibit: which songs actually drive the independence trend.

Every number in this project depends on a model deciding what a song is about. That
decision should be inspectable, not taken on trust. This module lists the songs the
classifier is most confident about in each decade, so a reader who knows the music can
check the result directly rather than relying on aggregate accuracy figures.

It also lists the classifier's least confident calls near the decision boundary, which
is where errors concentrate and where a reader is most likely to disagree.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from music_tastes.analysis_trends import P_THRESHOLD, derive_labels
from music_tastes.coverage import load_joined
from music_tastes.paths import REPORTS, require

STANCE_COLS = {
    "independence": "p_independence_max",
    "heartbreak": "p_heartbreak_max",
    "devotion": "p_devotion_max",
    "longing": "p_longing_max",
    "casual": "p_casual_max",
    "conflict": "p_conflict_max",
}


def top_by_decade(
    df: pd.DataFrame, stance: str, per_decade: int = 6, relationship_only: bool = True
) -> pd.DataFrame:
    """Highest-scoring songs per decade.

    ``relationship_only`` must default to True because that is what the headline
    number measures: the independence share is computed *within* relationship songs.
    Listing raw stance scores without the gate is misleading — the classifier fires
    on "we don't need no education" and on songs about tax brackets, and those are
    excluded from the statistic by the relationship condition, not by the stance
    score.
    """
    col = STANCE_COLS[stance]
    if col not in df.columns:
        return pd.DataFrame()
    d = df[df[col].notna()].copy()
    if relationship_only and "is_relationship" in d.columns:
        d = d[d["is_relationship"] == 1]
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    # Rank by the model score, then by chart exposure, so the exhibit shows songs a
    # reader is likely to recognise rather than obscure deep cuts.
    d = d.sort_values([col, "points"], ascending=False)
    return (
        d.groupby("decade", group_keys=False)
        .head(per_decade)[
            ["decade", "debut_year", "title_display", "artist_display", col, "points"]
        ]
        .rename(columns={col: "p"})
        .sort_values(["decade", "p"], ascending=[True, False])
    )


def gate_leakage(df: pd.DataFrame, stance: str = "independence", n: int = 12) -> pd.DataFrame:
    """Songs scoring high on the stance that the relationship gate rejects.

    These are the classifier's characteristic false positives — it keys on the
    literal claim rather than the romantic context. Showing them demonstrates that
    the relationship condition is doing real work, and makes the failure mode
    explicit rather than hidden.
    """
    col = STANCE_COLS[stance]
    if col not in df.columns or "is_relationship" not in df.columns:
        return pd.DataFrame()
    d = df[(df[col] > 0.9) & (df["is_relationship"] != 1)]
    return d.nlargest(n, "points")[
        ["debut_year", "title_display", "artist_display", col, "p_relationship_doc"]
    ].rename(columns={col: "p_stance"})


def borderline(df: pd.DataFrame, stance: str, n: int = 15, window: float = 0.06) -> pd.DataFrame:
    """Songs sitting closest to the decision threshold, where errors concentrate."""
    col = STANCE_COLS[stance]
    if col not in df.columns:
        return pd.DataFrame()
    d = df[df[col].notna()].copy()
    d["distance"] = (d[col] - P_THRESHOLD).abs()
    d = d[d["distance"] <= window].nlargest(n, "points")
    return d[["debut_year", "title_display", "artist_display", col, "points"]].rename(
        columns={col: "p"}
    )


def run(stance: str = "independence", per_decade: int = 6) -> dict:
    df = derive_labels(load_joined())
    top = top_by_decade(df, stance, per_decade)
    edge = borderline(df, stance)

    if top.empty:
        raise SystemExit("no stance scores available yet")

    top.to_csv(REPORTS / f"exhibit_top_{stance}.csv", index=False)
    edge.to_csv(REPORTS / f"exhibit_borderline_{stance}.csv", index=False)

    print(f"Highest-confidence '{stance}' songs by decade")
    print("(a reader who knows the music can check these directly)\n")
    for decade, g in top.groupby("decade"):
        print(f"  {int(decade)}s")
        for r in g.itertuples():
            print(f"    {r.p:.2f}  {r.title_display[:40]:40} — {r.artist_display[:28]}")
        print()

    leak = gate_leakage(df, stance)
    if not leak.empty:
        leak.to_csv(REPORTS / f"exhibit_gate_leakage_{stance}.csv", index=False)
        print(f"High '{stance}' scores REJECTED by the relationship gate")
        print("(the classifier's characteristic false positives — it keys on the")
        print(" literal claim, not the romantic context):\n")
        for r in leak.itertuples():
            print(f"    p_stance={r.p_stance:.2f} p_rel={r.p_relationship_doc:.2f}  "
                  f"{r.debut_year}  {r.title_display[:34]:34} — {r.artist_display[:22]}")
        print()

    if not edge.empty:
        print(f"Closest calls (within {0.06} of the {P_THRESHOLD} threshold) —")
        print("where the classifier is least reliable:\n")
        for r in edge.itertuples():
            print(f"    {r.p:.2f}  {r.debut_year}  {r.title_display[:38]:38} — "
                  f"{r.artist_display[:24]}")

    out = {
        "stance": stance,
        "top_by_decade": top.to_dict("records"),
        "borderline": edge.to_dict("records"),
        "gate_leakage": leak.to_dict("records") if not leak.empty else [],
    }
    (REPORTS / f"exhibit_{stance}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stance", default="independence", choices=list(STANCE_COLS))
    ap.add_argument("--per-decade", type=int, default=6)
    args = ap.parse_args()
    run(stance=args.stance, per_decade=args.per_decade)
