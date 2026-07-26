"""Is the independence trend an artefact of how the hypothesis was worded?

The "I don't need you" result is the only headline finding that survived every other
check, so it carries the weight of the project's conclusion. That makes it worth
attacking directly. A zero-shot entailment model is sensitive to phrasing, and the
whole result rests on one sentence:

    "The singer does not need this person and will be fine without them."

If the measured trend is a property of the music, rewording the claim should reproduce
it. If it is a property of that sentence, it will not.

Four paraphrases are scored on the same year-balanced sample and the decade trend is
compared across all of them. Agreement is reported as the correlation between per-song
probabilities and as the Kendall tau of each variant's yearly series.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .coverage import load_joined
from .fetch_lyrics import load_lyrics
from .lyrics_features import clean_lyrics
from .paths import DERIVED, REPORTS

# The original wording plus four deliberately varied paraphrases: different verbs,
# different framings (need / better off / self-sufficiency / rejection), and one
# phrased from the partner's perspective rather than the narrator's.
VARIANTS = {
    "original": "The singer does not need this person and will be fine without them.",
    "better_alone": "The singer is better off alone than with this person.",
    "self_sufficient": "The singer is independent and does not need anyone else.",
    "rejecting": "The singer is rejecting this person and moving on without regret.",
    "no_need_love": "The singer says they do not need this relationship to be happy.",
}


def score_variants(
    per_year: int = 22, device: str | None = None, batch_size: int = 32
) -> pd.DataFrame:
    """Score a year-balanced sample against every phrasing of the claim."""
    from .stance_nli import _build_pipeline, chunk_lyrics

    # Gate on relationship songs, matching the headline metric. The published
    # independence share is conditional on being a relationship song, so scoring
    # paraphrases over *all* English-lyric songs would compare two different
    # populations and make the level comparison meaningless.
    from .analysis_trends import derive_labels

    df = derive_labels(load_joined())
    d = df[
        df["has_lyrics"]
        & df["is_english"].fillna(False)
        & (df.get("is_relationship") == 1)
    ].copy()
    sample = (
        d.sort_values("points", ascending=False)
        .groupby("debut_year", group_keys=False)
        .head(per_year)
    )
    print(f"Scoring {len(sample):,} songs ({per_year}/year) against "
          f"{len(VARIANTS)} phrasings")

    clf = _build_pipeline(device, fp16=False)
    hyps = list(VARIANTS.values())
    keys = list(VARIANTS.keys())

    rows = []
    targets = list(sample.itertuples())
    for i in tqdm(range(0, len(targets), 16), desc="phrasings", unit="batch"):
        group = targets[i : i + 16]
        flat, owner = [], []
        for song in group:
            raw = load_lyrics(song.song_id)
            if not raw:
                continue
            chunks = chunk_lyrics(clean_lyrics(raw))
            flat.extend(chunks)
            owner.extend([song.song_id] * len(chunks))
        if not flat:
            continue

        res = clf(flat, hyps, multi_label=True, batch_size=batch_size)
        if isinstance(res, dict):
            res = [res]

        acc: dict[str, dict[str, list[float]]] = {}
        for song_id, r in zip(owner, res):
            bucket = acc.setdefault(song_id, {h: [] for h in hyps})
            for label, score in zip(r["labels"], r["scores"]):
                bucket[label].append(score)

        meta = {s.song_id: s for s in group}
        for song_id, bucket in acc.items():
            song = meta[song_id]
            rec = {
                "song_id": song_id,
                "debut_year": song.debut_year,
                "points": song.points,
            }
            for key, hyp in zip(keys, hyps):
                # Chunk maximum, matching how the headline measure is built.
                rec[key] = float(np.max(bucket[hyp])) if bucket[hyp] else np.nan
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_parquet(DERIVED / "prompt_robustness.parquet", index=False)
    return out


def analyse(t: pd.DataFrame, threshold: float = 0.5) -> dict:
    keys = [k for k in VARIANTS if k in t.columns]
    t = t.copy()
    t["decade"] = (t["debut_year"] // 10 * 10).astype(int)

    trends, decade_tbl = {}, {}
    for key in keys:
        sub = t[t[key].notna()]
        yearly = (
            sub.assign(hit=(sub[key] > threshold).astype(float))
            .groupby("debut_year")
            .agg(n=("hit", "size"), mean=("hit", "mean"))
        )
        yearly = yearly[yearly["n"] >= 6]
        tau, p = stats.kendalltau(
            yearly.index.to_numpy(dtype=float), yearly["mean"].to_numpy(dtype=float)
        )
        trends[key] = {
            "n": int(len(sub)),
            "kendall_tau": round(float(tau), 4),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "positive": bool(tau > 0),
        }
        dec = (
            sub.assign(hit=(sub[key] > threshold).astype(float))
            .groupby("decade")["hit"]
            .mean()
        )
        decade_tbl[key] = {int(k): round(float(v), 4) for k, v in dec.items()}

    corr = t[keys].corr(method="spearman").round(3)

    taus = [v["kendall_tau"] for v in trends.values()]
    return {
        "threshold": threshold,
        "n_songs": int(len(t)),
        "per_variant": trends,
        "by_decade": decade_tbl,
        "inter_variant_spearman": corr.to_dict(),
        "all_positive": bool(all(x > 0 for x in taus)),
        "all_significant": bool(all(v["significant"] for v in trends.values())),
        "tau_range": [round(min(taus), 4), round(max(taus), 4)],
    }


def run(per_year: int = 22, threshold: float = 0.5) -> dict:
    path = DERIVED / "prompt_robustness.parquet"
    if path.exists():
        t = pd.read_parquet(path)
        print(f"Using cached scores for {len(t):,} songs")
    else:
        t = score_variants(per_year=per_year)

    res = analyse(t, threshold)
    (REPORTS / "prompt_robustness.json").write_text(json.dumps(res, indent=2, default=str))

    print(f"\nDoes the independence trend survive rewording? (n={res['n_songs']:,})\n")
    print(f"  {'phrasing':18} {'tau':>8} {'p':>10}   decade shares")
    for key, v in res["per_variant"].items():
        dec = res["by_decade"][key]
        trail = " ".join(f"{d % 100:02d}s:{s:.0%}" for d, s in sorted(dec.items()))
        print(f"  {key:18} {v['kendall_tau']:+8.3f} {v['p_value']:10.2g}   {trail}")

    print(f"\n  all positive: {res['all_positive']}   "
          f"all significant: {res['all_significant']}   "
          f"tau range: {res['tau_range']}")
    print(f"\n  wrote {REPORTS / 'prompt_robustness.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year", type=int, default=22)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    run(per_year=args.per_year, threshold=args.threshold)
