"""Stance composition of relationship songs over time.

The headline number answers "what share of relationship songs say I don't need you".
This answers the fuller question behind it: what did the *rest* of the space look
like, and what did the independence stance grow at the expense of?

Stances are scored independently rather than as a forced choice, because a song can
be both heartbroken and angry, or both devoted and longing. The composition is
therefore reported two ways:

* **Independent shares** -- for each stance, the share of relationship songs whose
  probability exceeds the threshold. These need not sum to 1, and that is correct:
  overlap between stances is real.
* **Dominant stance** -- each song assigned to its single highest-scoring stance, so
  the shares do sum to 1 and can be read as a composition.

Both are given because they answer different questions and can disagree in
interesting ways.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis_trends import P_THRESHOLD, derive_labels  # noqa: E402
from .coverage import load_joined  # noqa: E402
from .paths import FIGURES, REPORTS  # noqa: E402

STANCES = {
    "independence": "Doesn't need them / better off alone",
    "heartbreak": "Heartbroken / wants an ex back",
    "devotion": "Devoted / committed",
    "longing": "Longing for someone unattainable",
    "casual": "Casual / physical, no commitment",
    "conflict": "Angry at a partner",
}

PALETTE = {
    "independence": "#d62728",
    "heartbreak": "#1f77b4",
    "devotion": "#2ca02c",
    "longing": "#9467bd",
    "casual": "#ff7f0e",
    "conflict": "#8c564b",
}


def _relationship_songs(df: pd.DataFrame) -> pd.DataFrame:
    d = derive_labels(df, P_THRESHOLD)
    return d[d.get("is_relationship") == 1].copy()


def independent_shares(df: pd.DataFrame, weight: str | None = "points") -> pd.DataFrame:
    """Share of relationship songs above threshold on each stance, by decade."""
    d = _relationship_songs(df)
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    rows = []
    for decade, g in d.groupby("decade"):
        row = {"decade": decade, "n": len(g)}
        for stance in STANCES:
            col = f"p_{stance}_max"
            if col not in g.columns:
                continue
            sub = g[g[col].notna()]
            if len(sub) < 25:
                continue
            hit = (sub[col] > P_THRESHOLD).astype(float).to_numpy()
            w = sub[weight].to_numpy(dtype=float) if weight else np.ones(len(sub))
            row[stance] = float((hit * w).sum() / w.sum()) if w.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dominant_stance(df: pd.DataFrame, weight: str | None = "points") -> pd.DataFrame:
    """Assign each relationship song its highest-scoring stance, then compose."""
    d = _relationship_songs(df)
    cols = [f"p_{s}_max" for s in STANCES if f"p_{s}_max" in d.columns]
    if not cols:
        return pd.DataFrame()
    d = d[d[cols].notna().all(axis=1)].copy()
    if d.empty:
        return pd.DataFrame()

    d["dominant"] = d[cols].idxmax(axis=1).str.replace("p_", "", regex=False).str.replace(
        "_max", "", regex=False
    )
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    w = d[weight] if weight else pd.Series(1.0, index=d.index)
    d["_w"] = w.astype(float)

    tot = d.groupby("decade")["_w"].sum()
    comp = d.groupby(["decade", "dominant"])["_w"].sum().unstack(fill_value=0.0)
    comp = comp.div(tot, axis=0)
    comp["n"] = d.groupby("decade").size()
    return comp.reset_index()


def _plot_composition(comp: pd.DataFrame, path) -> bool:
    stances = [s for s in STANCES if s in comp.columns]
    if not stances or comp.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = [f"{int(d)}s" for d in comp["decade"]]
    bottom = np.zeros(len(comp))
    for stance in stances:
        vals = comp[stance].to_numpy(dtype=float) * 100
        ax.bar(x, vals, bottom=bottom, label=STANCES[stance],
               color=PALETTE.get(stance), width=0.72)
        bottom += vals
    ax.set_ylabel("% of relationship songs")
    ax.set_title("What relationship songs are about, by decade\n"
                 "(dominant stance, exposure-weighted)", fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left",
              bbox_to_anchor=(0, -0.12))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_independent(shares: pd.DataFrame, path) -> bool:
    stances = [s for s in STANCES if s in shares.columns]
    if not stances or shares.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = [int(d) for d in shares["decade"]]
    for stance in stances:
        ax.plot(x, shares[stance] * 100, marker="o", ms=3.5, lw=1.7,
                color=PALETTE.get(stance), label=STANCES[stance])
    ax.set_ylabel("% of relationship songs above threshold")
    ax.set_xlabel("Decade")
    ax.set_title("Stances within relationship songs (independent, may overlap)",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def run() -> dict:
    df = load_joined()
    shares = independent_shares(df)
    comp = dominant_stance(df)

    shares.to_csv(REPORTS / "stance_shares_by_decade.csv", index=False)
    if not comp.empty:
        comp.to_csv(REPORTS / "stance_composition_by_decade.csv", index=False)

    _plot_independent(shares, FIGURES / "stance_shares.png")
    _plot_composition(comp, FIGURES / "stance_composition.png")

    print("Stances within relationship songs (exposure-weighted, independent)\n")
    stances = [s for s in STANCES if s in shares.columns]
    header = "  decade    n  " + "".join(f"{s[:11]:>13}" for s in stances)
    print(header)
    for r in shares.itertuples():
        cells = "".join(
            f"{getattr(r, s, float('nan')) * 100:12.1f}%" for s in stances
        )
        print(f"  {int(r.decade)}s {r.n:5}  {cells}")

    if not comp.empty:
        print("\nDominant stance (sums to 100%)\n")
        dstances = [s for s in STANCES if s in comp.columns]
        print("  decade    n  " + "".join(f"{s[:11]:>13}" for s in dstances))
        for r in comp.itertuples():
            cells = "".join(
                f"{getattr(r, s, float('nan')) * 100:12.1f}%" for s in dstances
            )
            print(f"  {int(r.decade)}s {int(r.n):5}  {cells}")

    out = {
        "independent_shares": shares.round(4).to_dict("records"),
        "dominant_composition": comp.round(4).to_dict("records") if not comp.empty else [],
    }
    (REPORTS / "stance_composition.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run()
