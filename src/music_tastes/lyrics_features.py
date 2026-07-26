"""Stage 5 (Method A): deterministic lyric features.

Two independent routes, both fully local and reproducible:

1. **Lexicon route.** Word-level emotion and valence norms (NRC VAD, NRC EmoLex,
   VADER) aggregated over the lyric. Transparent and auditable, but blind to
   negation, irony and narrative stance -- "I don't need you" and "I need you" score
   almost identically.

2. **Embedding route.** Cosine similarity between the song's embedding and hand-written
   anchor phrases for each theme and stance (see :mod:`music_tastes.taxonomy`). This
   captures stance, which is what the research question actually turns on, at the cost
   of being harder to inspect.

Both are written out. Method B (LLM) is scored against the same taxonomy, and
agreement between the three is reported rather than assumed.

No lyric text leaves this module: it reads the gitignored cache and emits numbers.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from .fetch_lyrics import load_lyrics
from .paths import DERIVED, RAW
from .taxonomy import (
    INDEPENDENCE_PHRASES,
    NON_RELATIONSHIP_ANCHORS,
    RELATIONSHIP_ANCHORS,
    RELATIONSHIP_KEYWORDS,
    STANCE_ANCHORS,
    STANCE_LABELS,
)

LEX_DIR = RAW / "lexicons"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_SECTION = re.compile(r"\[[^\]]{0,80}\]")
_WORD = re.compile(r"[a-z']+")

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative",
]


def clean_lyrics(text: str) -> str:
    """Strip section headers such as [Chorus] and collapse whitespace."""
    return re.sub(r"\s+", " ", _SECTION.sub(" ", text)).strip()


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# --------------------------------------------------------------------------- lexicon


def _load_lexicons() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vad = pd.read_parquet(LEX_DIR / "nrc_vad.parquet").set_index("word")
    emo = pd.read_parquet(LEX_DIR / "nrc_emolex.parquet").set_index("word")
    vader = pd.read_parquet(LEX_DIR / "vader.parquet").set_index("word")
    return vad, emo, vader


def lexicon_features(tokens: list[str], vad, emo, vader) -> dict:
    """Aggregate word-level norms over a lyric.

    Every feature records the share of tokens the lexicon actually covered, because a
    mean over 3 matched words is not comparable to a mean over 300.
    """
    n = len(tokens)
    if n == 0:
        return {}

    counts = pd.Series(tokens).value_counts()
    out: dict[str, float] = {"n_tokens": n, "n_types": int(counts.size)}
    out["type_token_ratio"] = counts.size / n

    hit = counts.index.intersection(vad.index)
    if len(hit):
        w = counts.loc[hit].to_numpy(dtype=float)
        sub = vad.loc[hit]
        total = w.sum()
        out["vad_valence"] = float((sub["valence"].to_numpy() * w).sum() / total)
        out["vad_arousal"] = float((sub["arousal"].to_numpy() * w).sum() / total)
        out["vad_dominance"] = float((sub["dominance"].to_numpy() * w).sum() / total)
        out["vad_coverage"] = float(total / n)

    hit = counts.index.intersection(emo.index)
    if len(hit):
        w = counts.loc[hit].to_numpy(dtype=float)
        sub = emo.loc[hit]
        for col in EMOTIONS:
            if col in sub.columns:
                # Share of all tokens carrying this association.
                out[f"emo_{col}"] = float((sub[col].to_numpy() * w).sum() / n)
        out["emo_coverage"] = float(w.sum() / n)

    hit = counts.index.intersection(vader.index)
    if len(hit):
        w = counts.loc[hit].to_numpy(dtype=float)
        sub = vader.loc[hit]
        out["vader_valence"] = float((sub["vader_valence"].to_numpy() * w).sum() / w.sum())
        out["vader_coverage"] = float(w.sum() / n)

    return out


def keyword_features(text: str) -> dict:
    """Transparent phrase counts, kept so embedding results can be sanity-checked."""
    low = " " + re.sub(r"[^\w\s']", " ", text.lower()) + " "
    rel = sum(low.count(f" {k} ") if " " not in k else low.count(k)
              for k in RELATIONSHIP_KEYWORDS)
    ind = sum(low.count(p) for p in INDEPENDENCE_PHRASES)
    words = max(len(tokenize(low)), 1)
    return {
        "kw_relationship_hits": rel,
        "kw_relationship_rate": rel / words,
        "kw_independence_hits": ind,
        "kw_independence_rate": ind / words,
        "kw_first_person_sing": len(re.findall(r"\b(i|me|my|mine|myself)\b", low)) / words,
        "kw_second_person": len(re.findall(r"\b(you|your|yours|yourself)\b", low)) / words,
        "kw_first_person_plural": len(re.findall(r"\b(we|us|our|ours)\b", low)) / words,
    }


# ------------------------------------------------------------------------- embedding


def _chunk(text: str, size: int = 60) -> list[str]:
    """Split a lyric into word chunks, roughly verse-sized."""
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)] or [text]


def embedding_features(texts: list[str], batch_size: int = 128) -> pd.DataFrame:
    """Score each lyric against the theme and stance anchors.

    A song is represented two ways: by the mean of its chunk embeddings (overall gist)
    and by the best-matching chunk per anchor (a stance often occupies one verse). The
    max-over-chunks view is what catches a self-sufficiency turn inside an otherwise
    sad breakup song.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)

    anchor_texts: list[str] = []
    anchor_groups: list[str] = []
    for phrase in RELATIONSHIP_ANCHORS:
        anchor_texts.append(phrase)
        anchor_groups.append("relationship")
    for phrase in NON_RELATIONSHIP_ANCHORS:
        anchor_texts.append(phrase)
        anchor_groups.append("non_relationship")
    for label, phrases in STANCE_ANCHORS.items():
        for phrase in phrases:
            anchor_texts.append(phrase)
            anchor_groups.append(label)

    anchors = model.encode(
        anchor_texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
    )
    groups = np.array(anchor_groups)

    # Encode every chunk of every song in one flat pass, then regroup.
    flat, owner = [], []
    for i, text in enumerate(texts):
        chunks = _chunk(text)
        flat.extend(chunks)
        owner.extend([i] * len(chunks))
    owner = np.array(owner)

    emb = model.encode(
        flat,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    sims = emb @ anchors.T  # (n_chunks, n_anchors)

    rows = []
    for i in range(len(texts)):
        mask = owner == i
        if not mask.any():
            rows.append({})
            continue
        chunk_sims = sims[mask]
        mean_doc = chunk_sims.mean(axis=0)
        max_doc = chunk_sims.max(axis=0)

        row: dict[str, float] = {}
        for group in ["relationship", "non_relationship", *STANCE_LABELS]:
            g = groups == group
            row[f"emb_{group}_mean"] = float(mean_doc[g].mean())
            row[f"emb_{group}_max"] = float(max_doc[g].max())

        row["emb_relationship_margin"] = (
            row["emb_relationship_mean"] - row["emb_non_relationship_mean"]
        )
        stance_scores = {s: row[f"emb_{s}_max"] for s in STANCE_LABELS}
        best = max(stance_scores, key=stance_scores.get)
        ordered = sorted(stance_scores.values(), reverse=True)
        row["emb_stance"] = best
        row["emb_stance_score"] = stance_scores[best]
        row["emb_stance_margin"] = ordered[0] - ordered[1]
        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ run


def run(limit: int | None = None, skip_embeddings: bool = False) -> pd.DataFrame:
    index = pd.read_parquet(DERIVED / "lyrics_index.parquet")
    usable = index[
        index["has_lyrics"].fillna(False)
        & ~index["is_instrumental"].fillna(False)
        & index["is_english"].fillna(False)
    ].copy()
    if limit:
        usable = usable.head(limit)

    print(f"Extracting features for {len(usable):,} songs with usable lyrics")

    vad, emo, vader = _load_lexicons()

    records, texts, ids = [], [], []
    for song_id in tqdm(usable["song_id"], desc="lexicon", unit="song"):
        raw = load_lyrics(song_id)
        if not raw:
            continue
        text = clean_lyrics(raw)
        tokens = tokenize(text)
        rec = {"song_id": song_id}
        rec.update(lexicon_features(tokens, vad, emo, vader))
        rec.update(keyword_features(text))
        records.append(rec)
        texts.append(text)
        ids.append(song_id)

    feats = pd.DataFrame(records)

    if not skip_embeddings and texts:
        print(f"\nEmbedding {len(texts):,} songs against "
              f"{len(RELATIONSHIP_ANCHORS) + len(NON_RELATIONSHIP_ANCHORS) + sum(len(v) for v in STANCE_ANCHORS.values())} anchors ...")
        emb = embedding_features(texts)
        emb["song_id"] = ids
        feats = feats.merge(emb, on="song_id", how="left")

    out = DERIVED / "lyric_features_method_a.parquet"
    feats.to_parquet(out, index=False)

    print(f"\nWrote {len(feats):,} rows to {out}")
    if "vad_valence" in feats:
        print(f"  mean VAD valence:   {feats['vad_valence'].mean():.4f} "
              f"(coverage {feats['vad_coverage'].mean():.1%})")
    if "emb_stance" in feats:
        print("\n  Method A stance distribution:")
        for label, cnt in feats["emb_stance"].value_counts().items():
            print(f"    {label:32} {cnt:6,} ({cnt / len(feats):5.1%})")
        rel = feats["emb_relationship_margin"] > 0
        print(f"\n  relationship songs (margin>0): {rel.sum():,} ({rel.mean():.1%})")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()
    run(limit=args.limit, skip_embeddings=args.skip_embeddings)
