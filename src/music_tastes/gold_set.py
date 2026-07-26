"""Stage 10: gold-set validation.

Two label sets, kept separate because they answer different questions and have very
different sampling properties.

**Random set** (`gold_random.csv`). 32 songs drawn at random, four per decade, from
songs with lyrics. Labelled by hand from the lyric text. This is an unbiased estimate
of how the classifiers behave on typical chart songs, and it is the set that can
legitimately be used to estimate the relationship-detection rate.

**Purposive set** (`gold_purposive.csv`). 13 well-known songs chosen precisely because
their stance is uncontroversial, and deliberately balanced between "I don't need you"
and "I want you back". It exists because independence songs are rare: the random
sample of 32 contained exactly one, which is far too few to estimate precision or
recall for the class the research question is about. Because these songs were chosen
for being clear-cut and famous, accuracy on this set is an **upper bound**, not an
unbiased estimate, and it is reported as such.

Songs whose label was genuinely ambiguous on reading are marked `uncertain` and
excluded from scoring rather than being forced into a class.
"""

from __future__ import annotations

import json

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
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )
    return {
        "n": len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / len(y_true) if y_true else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
    }


def evaluate_random(threshold: float = 0.5) -> dict:
    feats = pd.read_parquet(require(DERIVED / "lyric_features_method_b.parquet"))
    feats = feats.set_index("song_id")
    a_path = DERIVED / "lyric_features_method_a.parquet"
    feats_a = pd.read_parquet(a_path).set_index("song_id") if a_path.exists() else None

    rows = []
    for song_id, (is_rel, stance, title) in RANDOM_GOLD.items():
        if is_rel is None or song_id not in feats.index:
            continue
        f = feats.loc[song_id]
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
            row["a_pred_relationship"] = int(fa.get("emb_relationship_margin", 0) > 0)
            row["a_stance"] = fa.get("emb_stance")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"note": "no gold songs scored yet"}

    out = {"n_scored": len(df)}
    out["method_b_relationship"] = _metrics(
        df["gold_relationship"].tolist(), df["b_pred_relationship"].tolist()
    )
    if "a_pred_relationship" in df.columns and df["a_pred_relationship"].notna().all():
        out["method_a_relationship"] = _metrics(
            df["gold_relationship"].tolist(),
            df["a_pred_relationship"].astype(int).tolist(),
        )
        out["methods_agree_kappa"] = float(
            cohen_kappa_score(
                df["b_pred_relationship"].tolist(),
                df["a_pred_relationship"].astype(int).tolist(),
            )
        )

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
        row = {
            "title": title,
            "gold_independence": gold,
            "b_p_independence": f.get("p_independence_max"),
            "b_pred": int(f.get("p_independence_max", 0) > threshold),
        }
        if feats_a is not None and song_id in feats_a.index:
            stance = feats_a.loc[song_id].get("emb_stance")
            row["a_stance"] = stance
            row["a_pred"] = int(stance == "independence_self_sufficiency")
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
    if "a_pred" in df.columns and df["a_pred"].notna().all():
        out["method_a_independence"] = _metrics(
            df["gold_independence"].tolist(), df["a_pred"].astype(int).tolist()
        )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(GOLD_DIR / "gold_purposive_scored.csv", index=False)
    return out


def run(threshold: float = 0.5) -> dict:
    results = {
        "threshold": threshold,
        "random_sample": evaluate_random(threshold),
        "purposive_sample": evaluate_purposive(threshold),
    }
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    (GOLD_DIR / "validation.json").write_text(json.dumps(results, indent=2, default=str))

    print("Gold-set validation\n")
    rand = results["random_sample"]
    if "method_b_relationship" in rand:
        print(f"Random sample ({rand['n_scored']} songs, hand-labelled, 4 per decade)")
        for method in ("method_b_relationship", "method_a_relationship"):
            m = rand.get(method)
            if not m:
                continue
            print(f"  {method:26} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
                  f"rec={m['recall']:.2f} f1={m['f1']:.2f} kappa={m['cohen_kappa']}")
        if "methods_agree_kappa" in rand:
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


if __name__ == "__main__":
    run()
