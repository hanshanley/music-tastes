"""Stage 6 (Method B): stance classification with a local zero-shot NLI model.

The relationship topic is scored on the first 380 cleaned words. Stances use the
maximum over up to 12 deduplicated 70-word chunks. The two tasks are independent;
analysis subsequently gates stance shares on a relationship-positive topic label.
Legacy small-sample validation is developmental evidence, not proof of population
accuracy. See METHODS.md for the rubric, denominator, and review limitations.
"""

from __future__ import annotations

import argparse
import json
import re
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

from .fetch_lyrics import load_lyrics
from .lyrics_features import clean_lyrics
from .paths import CACHE, DERIVED, atomic_json, atomic_parquet

MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_REVISION = "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"
NLI_CACHE = CACHE / "nli"

CHUNK_WORDS = 70
MAX_CHUNKS = 12
DOC_WORDS = 380

# Each hypothesis is judged independently (multi_label semantics), so they need not
# be mutually exclusive. Phrasing is deliberately about the narrator's claim rather
# than the song's mood, because mood and stance come apart in breakup songs.
HYPOTHESES = {
    "relationship": "This song is about a romantic or sexual relationship.",
    "independence": "The singer does not need this person and will be fine without them.",
    "devotion": "The singer is devoted to their partner and wants to stay with them forever.",
    "longing": "The singer wants someone they cannot have and is longing for them.",
    "heartbreak": "The singer is heartbroken and wants their ex-partner back.",
    "casual": "The singer wants a purely physical relationship with no commitment.",
    "conflict": "The singer is angry at their partner for mistreating or betraying them.",
}

# Hypotheses judged on the whole document rather than per chunk. Topic-level questions
# are better answered from the full lyric; stance questions are better answered from
# the strongest chunk.
DOC_LEVEL = {"relationship"}


def cache_signature(text: str) -> str:
    config = {
        "schema": 1, "model": MODEL, "revision": MODEL_REVISION,
        "hypotheses": HYPOTHESES, "doc_words": DOC_WORDS,
        "chunk_words": CHUNK_WORDS, "max_chunks": MAX_CHUNKS,
        "preprocessing": 1, "text": clean_lyrics(text),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def valid_record(record: dict, text: str) -> bool:
    keys = ["p_relationship_doc"] + [f"p_{key}_max" for key in CHUNK_KEYS]
    return (
        isinstance(record, dict)
        and not record.get("error")
        and record.get("signature") == cache_signature(text)
        and all(
            isinstance(record.get(k), (int, float))
            and np.isfinite(record[k]) and 0 <= record[k] <= 1 for k in keys
        )
    )


def read_record(song_id: str) -> dict | None:
    try:
        rec = json.loads(_cache_path(song_id).read_text())
        text = load_lyrics(song_id)
        return rec if (
            text and valid_record(rec, text) and rec.get("song_id") == song_id
        ) else None
    except (OSError, ValueError, TypeError):
        return None


_SONG_ID_RE = re.compile(r"^s[0-9a-f]{16}$")


def chunk_lyrics(text: str, size: int = CHUNK_WORDS) -> list[str]:
    """Split into verse-sized chunks, dropping duplicates (repeated choruses)."""
    words = text.split()
    if not words:
        return []
    raw = [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    seen: set[str] = set()
    out: list[str] = []
    for chunk in raw:
        key = re.sub(r"[^a-z ]", "", chunk.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(chunk)
        if len(out) >= MAX_CHUNKS:
            break
    return out


def _cache_path(song_id: str):
    if not _SONG_ID_RE.match(str(song_id)):
        raise ValueError(f"refusing to use malformed song id as a path: {song_id!r}")
    d = NLI_CACHE / song_id[1:3]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{song_id}.json"


def _build_pipeline(device: str | None, fp16: bool = True):
    import torch
    from transformers import pipeline

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    kwargs = {}
    if fp16 and device in ("mps", "cuda"):
        kwargs["torch_dtype"] = torch.float16
    print(f"Loading {MODEL} on {device} (fp16={bool(kwargs)}) ...")
    return pipeline(
        "zero-shot-classification", model=MODEL, revision=MODEL_REVISION,
        device=device, **kwargs
    )


CHUNK_KEYS = [k for k in HYPOTHESES if k not in DOC_LEVEL]
CHUNK_HYPS = [HYPOTHESES[k] for k in CHUNK_KEYS]
DOC_KEYS = [k for k in HYPOTHESES if k in DOC_LEVEL]
DOC_HYPS = [HYPOTHESES[k] for k in DOC_KEYS]


def score_batch(clf, texts: dict[str, str], batch_size: int) -> dict[str, dict]:
    """Score a group of songs in as few forward passes as possible.

    Calling the pipeline once per song leaves the GPU idle between songs and was the
    dominant cost (~1.7 s/song). Flattening every chunk of every song in the group
    into a single call keeps batches full.
    """
    order: list[str] = []
    flat_chunks: list[str] = []
    owner: list[str] = []
    docs: list[str] = []

    for song_id, text in texts.items():
        chunks = chunk_lyrics(text)
        if not chunks:
            continue
        order.append(song_id)
        flat_chunks.extend(chunks)
        owner.extend([song_id] * len(chunks))
        docs.append(" ".join(text.split()[:DOC_WORDS]))

    if not order:
        return {}

    out: dict[str, dict] = {sid: {"n_chunks": 0} for sid in order}

    chunk_res = clf(flat_chunks, CHUNK_HYPS, multi_label=True, batch_size=batch_size)
    if isinstance(chunk_res, dict):
        chunk_res = [chunk_res]

    # The pipeline reorders labels by score, so map them back by name.
    acc: dict[str, dict[str, list[float]]] = {
        sid: {h: [] for h in CHUNK_HYPS} for sid in order
    }
    for sid, res in zip(owner, chunk_res):
        for label, score in zip(res["labels"], res["scores"]):
            acc[sid][label].append(score)

    for sid in order:
        rec = out[sid]
        for key, hyp in zip(CHUNK_KEYS, CHUNK_HYPS):
            scores = acc[sid][hyp]
            if scores:
                rec[f"p_{key}_max"] = float(np.max(scores))
                rec[f"p_{key}_mean"] = float(np.mean(scores))
        rec["n_chunks"] = len(acc[sid][CHUNK_HYPS[0]])

    if DOC_HYPS:
        doc_res = clf(docs, DOC_HYPS, multi_label=True, batch_size=batch_size)
        if isinstance(doc_res, dict):
            doc_res = [doc_res]
        for sid, res in zip(order, doc_res):
            mapping = dict(zip(res["labels"], res["scores"]))
            for key, hyp in zip(DOC_KEYS, DOC_HYPS):
                out[sid][f"p_{key}_doc"] = float(mapping[hyp])

    return out


def rebuild_from_cache() -> pd.DataFrame:
    """Reconstruct the Method B feature table from the per-song NLI cache.

    A full pass takes hours and writes its parquet only at the end. Because the
    run order is year-balanced, any partial cache is already a usable
    year-stratified sample, so this makes interim analysis possible and doubles as
    crash recovery.
    """
    records = []
    for path in NLI_CACHE.rglob("*.json"):
        try:
            rec = read_record(path.stem)
            if rec:
                records.append(rec)
        except ValueError:
            continue
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("song_id").reset_index(drop=True)
    out = DERIVED / "lyric_features_method_b.parquet"
    atomic_parquet(out, df)
    print(f"Rebuilt Method B features for {len(df):,} songs -> {out}")
    return df


def _priority_order(song_ids: list[str], songs: pd.DataFrame) -> list[str]:
    """Order songs so that any prefix is roughly balanced across debut years.

    A full pass takes many hours, so the run is ordered to be useful early: within
    each debut year songs are ranked by chart exposure, then years are interleaved.
    Stopping after any number of songs therefore yields a year-balanced,
    exposure-weighted sample rather than an arbitrary alphabetical slice.
    """
    meta = songs[songs["song_id"].isin(song_ids)][["song_id", "debut_year", "points"]]
    meta = meta.sort_values(["debut_year", "points"], ascending=[True, False])
    meta["rank_in_year"] = meta.groupby("debut_year").cumcount()
    meta = meta.sort_values(["rank_in_year", "debut_year"])
    ordered = [s for s in meta["song_id"] if s in set(song_ids)]
    # Anything without metadata still gets processed, just last.
    return ordered + [s for s in song_ids if s not in set(ordered)]


def run(
    limit: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
    group_size: int = 32,
    fp16: bool = False,
    refresh_index: bool = True,
) -> pd.DataFrame:
    if refresh_index:
        from .fetch_lyrics import rebuild_index_from_cache

        rebuild_index_from_cache()

    index = pd.read_parquet(DERIVED / "lyrics_index.parquet")
    songs = pd.read_parquet(DERIVED / "songs_weighted.parquet")
    usable = index[
        index["has_lyrics"].fillna(False)
        & ~index["is_instrumental"].fillna(False)
        & index["is_english"].fillna(False)
    ]
    all_ids = _priority_order(list(usable["song_id"]), songs)
    song_ids = all_ids
    if limit:
        song_ids = song_ids[:limit]

    todo = [s for s in song_ids if read_record(s) is None]
    print(f"{len(song_ids):,} songs with usable lyrics; {len(todo):,} still to score")

    if todo:
        clf = _build_pipeline(device, fp16=fp16)
        with tqdm(total=len(todo), desc="nli", unit="song") as bar:
            for i in range(0, len(todo), group_size):
                group = todo[i : i + group_size]
                texts = {}
                signatures = {}
                for song_id in group:
                    raw = load_lyrics(song_id)
                    if raw:
                        texts[song_id] = clean_lyrics(raw)
                        signatures[song_id] = cache_signature(raw)
                try:
                    scored = score_batch(clf, texts, batch_size)
                except Exception as exc:  # noqa: BLE001 - keep a long run alive
                    scored = {sid: {"error": str(exc)[:200]} for sid in texts}
                for song_id, rec in scored.items():
                    rec["song_id"] = song_id
                    rec["signature"] = signatures[song_id]
                    rec["model_revision"] = MODEL_REVISION
                    atomic_json(_cache_path(song_id), rec)
                bar.update(len(group))

    records = []
    for song_id in all_ids:
        rec = read_record(song_id)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["song_id"])
    out = DERIVED / "lyric_features_method_b.parquet"
    atomic_parquet(out, df)

    print(f"\nWrote {len(df):,} rows to {out}")
    for key in HYPOTHESES:
        col = f"p_{key}_max" if f"p_{key}_max" in df else f"p_{key}_doc"
        if col in df:
            print(f"  {key:24} share>0.5 = {(df[col] > 0.5).mean():6.1%}  "
                  f"mean p = {df[col].mean():.3f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true",
                    help="Half precision. Measured slower than fp32 on Apple MPS.")
    args = ap.parse_args()
    run(
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        group_size=args.group_size,
        fp16=args.fp16,
    )
