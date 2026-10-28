"""Stage 10: gold-set validation.

Two label sets, kept separate because they answer different questions and have very
different sampling properties.

**Legacy random set**. 32 nominally random songs, four per decade, with two
uncertain cases excluded. Selection and annotation provenance have not been
independently audited; equal-decade accuracy is not population-weighted accuracy.

**Purposive set** (`gold_purposive.csv`). 13 well-known songs chosen precisely because
their stance is uncontroversial, and deliberately balanced between "I don't need you"
and "I want you back". It exists because independence songs are rare: the random
sample of 32 contained exactly one, which is far too few to estimate precision or
recall for the class the research question is about. Because these songs were chosen
for being clear-cut and famous, accuracy on this set may be optimistic and is not an unbiased population estimate.

Songs whose label was genuinely ambiguous on reading are marked `uncertain` and
excluded from scoring rather than being forced into a class.
"""

from __future__ import annotations

import json
import math

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .paths import DERIVED, REPORTS, require

GOLD_DIR = REPORTS / "gold"

# song_id -> (is_relationship, stance, title for readability)
# stance vocabulary matches taxonomy.STANCE_ANCHORS keys, plus "uncertain".
RANDOM_GOLD = {
    "sac67cfc406e73fa9": (1, "casual_physical", "Chantilly Lace"),
    "s5ba9b4e2f3706c2c": (1, "devotion_commitment", "Come Into My Heart"),
    "s4a57ba2bac02c0d8": (0, None, "Along Came Jones"),
    "s64d0e98bfb0519db": (1, "heartbreak_loss", "City Lights"),
    "s4805c9d6ecfc8fb8": (1, "devotion_commitment", "Bad Boy"),
    "s23a7ef1292fc63ef": (0, None, "Baby's First Christmas"),
    "sdb1da756e08185b0": (1, "casual_physical", "Chills And Fever"),
    "sf1f86ef33460d65c": (1, "casual_physical", "California Girl"),
    "sda9507db3abfe8bc": (0, None, "Bennie And The Jets"),
    "sd39f1753732c4a5f": (0, None, "Crackerbox Palace"),
    "s548cccf9c6272e8a": (0, None, "Dancin' Fool"),
    "s6295a67905f49e37": (0, None, "Animal Zoo"),
    "sbe032ad95ca7a17b": (1, "devotion_commitment", "Affair Of The Heart"),
    "s12a80e399d7f7ead": (0, None, "Breakdance"),
    "s0dbb2d54ea8767f0": (1, "heartbreak_loss", "Be Mine Tonight"),
    "sf0f94f93f517867e": (None, "uncertain", "All The Right Moves"),
    "sc821671701178111": (1, "devotion_commitment", "Breathe"),
    "sfe21b576a9977e28": (1, "devotion_commitment", "Broken Arrow"),
    "sacbf9309db408829": (0, None, "Boom! Shake The Room"),
    "saad3726d83ffd826": (0, None, "9th Wonder"),
    "s91bda29400925401": (1, "heartbreak_loss", "Angel"),
    "s3515057df497c771": (0, None, "Born To Fly"),
    "scc5ba23160c782bd": (0, None, "Californication"),
    "s3d3280c8b4346688": (0, None, "Bet On It"),
    "s2a3878207f0cdb8a": (1, "devotion_commitment", "Controlla"),
    "s25679bae865c5e4d": (1, "independence_self_sufficiency", "Chainsaw"),
    "s0225016e675d2aea": (1, "devotion_commitment", "Cornelia Street"),
    "s4077ecb8adb85c17": (0, None, "Blow A Bag"),
    "s452577256cf1f6ae": (None, "uncertain", "Brazzier"),
    "s7183c02d51f55390": (0, None, "Carolina"),
    "s695512cb360b3db6": (1, "casual_physical", "5 Dollar Pony Rides"),
    "sc3593d643efb2447": (0, None, "Blow For Blow"),
}

# (title, artist prefix, is_independence_stance)
PURPOSIVE_GOLD = [
    ("I Will Survive", "Gloria Gaynor", 1),
    ("Survivor", "Destiny", 1),
    ("Independent Women Part I", "Destiny", 1),
    ("Since U Been Gone", "Kelly", 1),
    ("Irreplaceable", "Beyonce", 1),
    ("Thank U, Next", "Ariana", 1),
    ("Truth Hurts", "Lizzo", 1),
    ("Stronger (What Doesn't Kill You)", "Kelly", 1),
    ("Nothing Compares 2 U", "Sinead", 0),
    ("Un-Break My Heart", "Toni", 0),
    ("I Want It That Way", "Backstreet", 0),
    ("Endless Love", "Diana", 0),
    ("Baby", "Justin Bieber", 0),
]


def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")

    def wilson(success, total):
        if not total:
            return None
        z = 1.959963984540054
        p = success / total
        center = p + z * z / (2 * total)
        half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        scale = 1 + z * z / total
        return [max(0.0, (center - half) / scale), min(1.0, (center + half) / scale)]

    return {
        "n": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(y_true) if y_true else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ci95_wilson": {
            "accuracy": wilson(tp + tn, len(y_true)),
            "precision": wilson(tp, tp + fp),
            "recall": wilson(tp, tp + fn),
        },
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
    }


def evaluate_random(threshold: float = 0.5) -> dict:
    feats = pd.read_parquet(require(DERIVED / "lyric_features_method_b.parquet"))
    feats = feats.set_index("song_id")
    a_path = DERIVED / "lyric_features_method_a.parquet"
    feats_a = pd.read_parquet(a_path).set_index("song_id") if a_path.exists() else None

    rows = []
    missing = 0
    for song_id, (is_rel, stance, title) in RANDOM_GOLD.items():
        if is_rel is None:
            continue
        if song_id not in feats.index:
            missing += 1
            continue
        f = feats.loc[song_id]
        if pd.isna(f.get("p_relationship_doc")):
            missing += 1
            continue
        row = {
            "song_id": song_id,
            "title": title,
            "gold_relationship": is_rel,
            "gold_stance": stance,
            "b_p_relationship": f.get("p_relationship_doc"),
            "b_pred_relationship": int(f.get("p_relationship_doc", 0) > threshold),
            "b_p_independence": f.get("p_independence_max"),
        }
        if feats_a is not None and song_id in feats_a.index:
            fa = feats_a.loc[song_id]
            row["a_relationship_margin"] = fa.get("emb_relationship_margin")
            margin = fa.get("emb_relationship_margin")
            row["a_pred_relationship"] = int(margin > 0) if pd.notna(margin) else float("nan")
            row["a_stance"] = fa.get("emb_stance")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"note": "no gold songs scored yet"}

    out = {
        "n_scored": len(df), "n_missing_score": missing,
        "sampling_note": "Legacy labels; selection/annotation provenance not independently audited.",
    }
    out["method_b_relationship"] = _metrics(
        df["gold_relationship"].tolist(), df["b_pred_relationship"].tolist()
    )
    if "a_pred_relationship" in df.columns:
        available = df.dropna(subset=["a_pred_relationship"])
        out["method_a_relationship"] = _metrics(
            available["gold_relationship"].tolist(),
            available["a_pred_relationship"].astype(int).tolist(),
        )
        out["methods_agree_kappa"] = float(
            cohen_kappa_score(
                available["b_pred_relationship"].tolist(),
                available["a_pred_relationship"].astype(int).tolist(),
            )
        ) if len(set(available["b_pred_relationship"]) | set(available["a_pred_relationship"])) > 1 else None

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_DIR / "gold_random_scored.csv", index=False)
    return out


def evaluate_purposive(threshold: float = 0.5) -> dict:
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")
    feats = pd.read_parquet(DERIVED / "lyric_features_method_b.parquet").set_index("song_id")
    a_path = DERIVED / "lyric_features_method_a.parquet"
    feats_a = pd.read_parquet(a_path).set_index("song_id") if a_path.exists() else None

    rows = []
    for title, artist, gold in PURPOSIVE_GOLD:
        m = songs[(songs["title_display"] == title)
                  & (songs["artist_display"].str.startswith(artist))]
        if m.empty:
            continue
        song_id = m.iloc[0]["song_id"]
        if song_id not in feats.index:
            continue
        f = feats.loc[song_id]
        if pd.isna(f.get("p_independence_max")):
            continue
        row = {
            "title": title,
            "gold_independence": gold,
            "b_p_independence": f.get("p_independence_max"),
            "b_pred": int(f.get("p_independence_max", 0) > threshold),
        }
        if feats_a is not None and song_id in feats_a.index:
            stance = feats_a.loc[song_id].get("emb_stance")
            row["a_stance"] = stance
            row["a_pred"] = (
                int(stance == "independence_self_sufficiency") if pd.notna(stance)
                else float("nan")
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"note": "no purposive gold songs scored yet"}

    out = {
        "n_scored": len(df),
        "caveat": "songs chosen for being unambiguous; accuracy is an upper bound",
        "method_b_independence": _metrics(
            df["gold_independence"].tolist(), df["b_pred"].tolist()
        ),
    }
    if "a_pred" in df.columns:
        available = df.dropna(subset=["a_pred"])
        out["method_a_independence"] = _metrics(
            available["gold_independence"].tolist(), available["a_pred"].astype(int).tolist()
        )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_DIR / "gold_purposive_scored.csv", index=False)
    return out


def run(threshold: float = 0.5) -> dict:
    results = {
        "threshold": threshold,
        "random_sample": evaluate_random(threshold),
        "purposive_sample": evaluate_purposive(threshold),
        "independent_review": evaluate_review(threshold),
    }
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (GOLD_DIR / "validation.json").write_text(json.dumps(results, indent=2, default=str))

    print("Gold-set validation\n")
    rand = results["random_sample"]
    if "method_b_relationship" in rand:
        print(f"Legacy sample ({rand['n_scored']} scored; annotation provenance unaudited)")
        for method in ("method_b_relationship", "method_a_relationship"):
            m = rand.get(method)
            if not m:
                continue
            print(f"  {method:26} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
                  f"rec={m['recall']:.2f} f1={m['f1']:.2f} kappa={m['cohen_kappa']}")
        if rand.get("methods_agree_kappa") is not None:
            print(f"  A vs B agreement (kappa): {rand['methods_agree_kappa']:.3f}")
    else:
        print(f"Random sample: {rand.get('note')}")

    pur = results["purposive_sample"]
    print()
    if "method_b_independence" in pur:
        print(f"Purposive independence set ({pur['n_scored']} songs) "
              f"-- UPPER BOUND, not an unbiased estimate")
        for method in ("method_b_independence", "method_a_independence"):
            m = pur.get(method)
            if not m:
                continue
            print(f"  {method:26} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
                  f"rec={m['recall']:.2f} f1={m['f1']:.2f}")
    else:
        print(f"Purposive set: {pur.get('note')}")
    return results


def prepare_review(per_decade: int = 20, seed: int = 20261027) -> pd.DataFrame:
    from .coverage import load_joined
    from .stance_nli import CHUNK_KEYS

    if per_decade < 1:
        raise ValueError("per_decade must be positive")
    df = load_joined()
    df = df[df["has_lyrics"] & df.get(
        "p_relationship_doc", pd.Series(float("nan"), index=df.index)
    ).notna()].copy()
    df["decade"] = df["debut_year"] // 10 * 10
    rows = []
    for _, group in df.groupby("decade"):
        rows.append(group.sample(min(per_decade, len(group)), random_state=seed))
    if not rows:
        raise SystemExit("No scored songs available for review. Run stance-b first.")
    sample = pd.concat(rows)
    columns = ["song_id", "title_display", "artist_display", "debut_year", "decade"]
    population_counts = df.groupby("decade").size()
    sample = sample[columns].copy()
    sample["sample_probability"] = sample["decade"].map(
        (per_decade / population_counts).clip(upper=1)
    )
    label_columns = ["gold_relationship"] + [f"gold_{key}" for key in CHUNK_KEYS]
    for column in label_columns + ["reviewer_id", "rationale"]:
        sample[column] = ""
    out = DERIVED / "review_packet.csv"
    challenge_path = DERIVED / "review_challenge.csv"
    if out.exists() or challenge_path.exists():
        raise SystemExit(f"Refusing to overwrite existing review labels in {DERIVED}")

    pool = df[~df["song_id"].isin(sample["song_id"])].copy()
    reasons = pd.Series("", index=pool.index)
    criteria = {"borderline_topic": pool["p_relationship_doc"].between(.35, .65)}
    if "emb_relationship_margin" in pool:
        criteria["model_disagreement"] = pool["emb_relationship_margin"].notna() & (
            (pool["emb_relationship_margin"] > 0) != (pool["p_relationship_doc"] > .5)
        )
    if "n_words" in pool:
        criteria["short_or_truncated"] = (pool["n_words"] < 80) | (pool["n_words"] > 380)
    if "source" in pool:
        criteria["historical_fallback"] = (pool["debut_year"] < 1980) & pool["source"].eq("lrclib")
    for reason, mask in criteria.items():
        reasons.loc[mask] += reason + ";"
    pool["selection_reasons"] = reasons.str.rstrip(";")
    candidates = pool[pool["selection_reasons"].ne("")]
    challenge_rows = [
        group.sample(min(10, per_decade, len(group)), random_state=seed + 1)
        for _, group in candidates.groupby("decade")
    ]
    challenge = (
        pd.concat(challenge_rows)[columns + ["selection_reasons"]].copy()
        if challenge_rows else pd.DataFrame(columns=columns + ["selection_reasons"])
    )
    for column in label_columns + ["reviewer_id", "rationale"]:
        challenge[column] = ""
    sample.to_csv(out, index=False)
    challenge.to_csv(challenge_path, index=False)
    from .paths import atomic_json

    atomic_json(out.with_suffix(".provenance.json"), {
        "seed": seed, "per_decade": per_decade, "n": len(sample),
        "population": "scored songs with available lyrics",
        "population_by_decade": population_counts.to_dict(),
        "challenge_n": len(challenge),
        "note": "Equal-decade sample; pooled accuracy is not population-weighted.",
    })
    print(f"Blind review packet: {out}. See METHODS.md for the rubric.")
    return sample


def evaluate_review(threshold: float = 0.5) -> dict:
    from .stance_nli import CHUNK_KEYS

    path = DERIVED / "review_packet.csv"
    if not path.exists():
        return {"status": "pending", "note": "No independent review packet completed."}
    review = pd.read_csv(path, dtype=str).fillna("")
    if review["song_id"].duplicated().any():
        raise ValueError("Review packet must contain unique song IDs")
    labels = review["gold_relationship"]
    if not labels.isin(["", "0", "1", "uncertain"]).all():
        raise ValueError("Review labels must be 0, 1, uncertain, or blank")
    reviewed = labels.ne("") & review["reviewer_id"].str.strip().ne("")
    valid = labels.isin(["0", "1"]) & reviewed
    feats = pd.read_parquet(DERIVED / "lyric_features_method_b.parquet")
    scored = review[valid].merge(
        feats[["song_id", "p_relationship_doc"]], on="song_id", validate="one_to_one"
    )
    scored = scored.dropna(subset=["p_relationship_doc"])
    result = {
        "status": (
            "unscorable" if reviewed.any() and not valid.any()
            else "complete" if len(review) and reviewed.all() and len(scored) == valid.sum()
            else "partial"
        ),
        "n_reviewed": int(reviewed.sum()), "n_total": len(review),
        "n_uncertain": int((reviewed & labels.eq("uncertain")).sum()),
        "n_scored": len(scored),
        "note": "Equal-decade review; report per-decade metrics, not unbiased population accuracy.",
        "by_decade": {},
    }
    for decade, group in scored.groupby("decade"):
        result["by_decade"][str(decade)] = _metrics(
            group["gold_relationship"].astype(int).tolist(),
            (group["p_relationship_doc"] > threshold).astype(int).tolist(),
        )
    joined = review.merge(feats, on="song_id", how="left", validate="one_to_one")
    result["conditional_stances"] = {}
    annotated = 0
    expected = int((valid & labels.eq("1")).sum()) * len(CHUNK_KEYS)
    for key in CHUNK_KEYS:
        gold, score = f"gold_{key}", f"p_{key}_max"
        if gold not in joined:
            continue
        if not joined[gold].isin(["", "0", "1", "uncertain"]).all():
            raise ValueError(f"{gold} labels must be 0, 1, uncertain, or blank")
        known = joined[gold].isin(["0", "1"]) & joined["reviewer_id"].str.strip().ne("")
        conditional = known & joined["gold_relationship"].eq("1")
        annotated += int(conditional.sum())
        if score not in joined:
            continue
        eligible = conditional & joined[score].notna() & joined["p_relationship_doc"].notna()
        group = joined[eligible]
        result["conditional_stances"][key] = {
            "scope": "Human relationship-positive songs; predictions include the production topic gate.",
            **_metrics(
                group[gold].astype(int).tolist(),
                ((group["p_relationship_doc"] > threshold) & (group[score] > threshold))
                .astype(int).tolist(),
            ),
        }
        if key == "independence":
            group = joined[known & joined[score].notna()]
            result["ungated_independence"] = _metrics(
                group[gold].astype(int).tolist(),
                (group[score] > threshold).astype(int).tolist(),
            )
    n_scored = sum(m["n"] for m in result["conditional_stances"].values())
    result["stance_review_status"] = (
        "complete" if expected and n_scored == expected
        else "partial" if annotated else "pending"
    )
    return result


if __name__ == "__main__":
    run()
