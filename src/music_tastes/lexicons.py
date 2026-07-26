"""Download and normalize the sentiment/emotion lexicons used by Method A.

Every lexicon is fetched from a citable origin and recorded in
``data/raw/lexicons/provenance.json`` with its URL, retrieval time and citation, so
the final report can attribute each measure to the group that produced it.

Lexicons
--------
NRC VAD      Valence/Arousal/Dominance ratings for ~20k words.
             Mohammad, S. M. (2018). "Obtaining Reliable Human Ratings of Valence,
             Arousal, and Dominance for 20,000 English Words." ACL 2018. National
             Research Council Canada.
NRC EmoLex   Binary associations with eight emotions and two sentiments.
             Mohammad, S. M., & Turney, P. D. (2013). "Crowdsourcing a Word-Emotion
             Association Lexicon." Computational Intelligence, 29(3). NRC Canada.
VADER        Valence scores tuned for social/informal text, used as an independent
             sentiment check. Hutto, C. J., & Gilbert, E. (2014). ICWSM-14.

Both NRC lexicons are free for research use; see the terms distributed in their
archives. They are downloaded at build time and never redistributed in this repo.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

from .paths import RAW, user_agent

LEX_DIR = RAW / "lexicons"

SOURCES = {
    "nrc_vad": {
        "url": "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip",
        "citation": (
            "Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, "
            "Arousal, and Dominance for 20,000 English Words. ACL 2018. "
            "National Research Council Canada."
        ),
    },
    "nrc_emolex": {
        "url": "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip",
        "citation": (
            "Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word-Emotion "
            "Association Lexicon. Computational Intelligence, 29(3), 436-465. "
            "National Research Council Canada."
        ),
    },
    "vader": {
        "url": (
            "https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/"
            "vaderSentiment/vader_lexicon.txt"
        ),
        "citation": (
            "Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based "
            "Model for Sentiment Analysis of Social Media Text. ICWSM-14."
        ),
    },
}


def _download(name: str) -> tuple[bytes, str]:
    url = SOURCES[name]["url"]
    print(f"  fetching {name} from {url}")
    r = requests.get(url, headers={"User-Agent": user_agent()}, timeout=180)
    r.raise_for_status()
    return r.content, datetime.now(timezone.utc).isoformat()


def _member(zf: zipfile.ZipFile, *needles: str) -> str:
    """Find the English-only archive member whose *filename* contains all needles.

    Matching is done on the basename, not the full path: the archives nest everything
    under a directory whose own name contains the lexicon name, so a full-path match
    also selects README.txt. The archives additionally ship a multilingual file and a
    per-language directory, both carrying ~100 translation columns; we want English.
    """
    excluded = (
        "forvariouslanguages",
        "onefileperlanguage",
        "onefileperdimension",
        "bipolarscale",
        "senselevel",
        "readme",
        "listoflanguages",
    )
    candidates = [
        n
        for n in zf.namelist()
        if all(x in n.rsplit("/", 1)[-1].lower() for x in needles)
        and not n.startswith("__MACOSX")
        and not any(x in n.lower() for x in excluded)
    ]
    if not candidates:
        raise KeyError(f"no archive member matching {needles} in {zf.namelist()[:10]}")
    # Prefer the shallowest, shortest path: the canonical top-level file.
    return min(candidates, key=lambda n: (n.count("/"), len(n)))


def build_vad() -> tuple[pd.DataFrame, str]:
    blob, ts = _download("nrc_vad")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = _member(zf, "vad-lexicon", ".txt")
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                sep="\t",
                header=None,
                names=["word", "valence", "arousal", "dominance"],
                usecols=[0, 1, 2, 3],
            )
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    for col in ("valence", "arousal", "dominance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df.dropna().drop_duplicates("word").reset_index(drop=True)
    out.to_parquet(LEX_DIR / "nrc_vad.parquet", index=False)
    return out, ts


def build_emolex() -> tuple[pd.DataFrame, str]:
    blob, ts = _download("nrc_emolex")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = _member(zf, "wordlevel", ".txt")
        with zf.open(name) as fh:
            df = pd.read_csv(fh, sep="\t", names=["word", "emotion", "flag"])
    wide = df.pivot_table(
        index="word", columns="emotion", values="flag", aggfunc="max", fill_value=0
    ).reset_index()
    wide.columns.name = None
    wide["word"] = wide["word"].astype(str).str.lower().str.strip()
    wide.to_parquet(LEX_DIR / "nrc_emolex.parquet", index=False)
    return wide, ts


def build_vader() -> tuple[pd.DataFrame, str]:
    blob, ts = _download("vader")
    df = pd.read_csv(
        io.BytesIO(blob), sep="\t", names=["word", "mean", "sd", "ratings"], usecols=[0, 1, 2]
    )
    df["word"] = df["word"].astype(str).str.lower().str.strip()
    df = df[["word", "mean"]].rename(columns={"mean": "vader_valence"}).dropna()
    df = df.drop_duplicates("word").reset_index(drop=True)
    df.to_parquet(LEX_DIR / "vader.parquet", index=False)
    return df, ts


def run() -> None:
    LEX_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {}

    print("Building lexicons ...")
    vad, ts_vad = build_vad()
    provenance["nrc_vad"] = {**SOURCES["nrc_vad"], "retrieved_at": ts_vad, "n_terms": len(vad)}
    print(f"    NRC VAD:    {len(vad):,} words")

    emo, ts_emo = build_emolex()
    emo_cols = [c for c in emo.columns if c != "word"]
    provenance["nrc_emolex"] = {
        **SOURCES["nrc_emolex"],
        "retrieved_at": ts_emo,
        "n_terms": len(emo),
        "categories": emo_cols,
    }
    print(f"    NRC EmoLex: {len(emo):,} words, categories: {', '.join(emo_cols)}")

    vad_er, ts_v = build_vader()
    provenance["vader"] = {**SOURCES["vader"], "retrieved_at": ts_v, "n_terms": len(vad_er)}
    print(f"    VADER:      {len(vad_er):,} terms")

    (LEX_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"  wrote provenance to {LEX_DIR / 'provenance.json'}")


if __name__ == "__main__":
    run()
