"""Stage 11b: construct validity of the sentiment measures.

Why this module exists
----------------------
The headline sentiment result -- that hit lyrics have become less positive -- rests on
averaging word-level valence norms over a lyric. That measure has several ways of
being right for the wrong reason, and an earlier version of this analysis got one of
them backwards. This module runs the checks explicitly so the claim can be audited
instead of trusted.

The correction worth recording
------------------------------
An earlier pass reported that "roughly a third of the effect is length, not mood",
based on a partial correlation controlling for log word count (rho -0.303 -> -0.200).
That framing was wrong. Lyric length is a **mediator**, not a confounder: the causal
path runs year -> songs get wordier -> word-average valence falls. Controlling for a
variable on the causal path subtracts real signal and understates the effect. It would
only be a confounder if something independent of era drove both length and valence.

Two tests settle it, and both are implemented below:

* ``length_strata_test`` -- the decline holds *within every length quintile*
  (rho -0.14 to -0.30, all p < 1e-05), including the shortest songs. Same-length
  songs really did get less positive.
* ``type_vs_token_test`` -- computing valence over unique word types, which removes
  repetition entirely, makes the decline *stronger* (rho -0.346 vs -0.302). So the
  effect is not short old songs repeating happy hooks.

What remains genuinely uncertain
--------------------------------
* ``coverage_drift_test`` -- the share of tokens the NRC lexicon recognizes falls from
  0.350 in the 1950s to 0.318 in the 2020s. The measure is therefore an average over a
  subtly different subset of each era's vocabulary.
* ``uncovered_vocabulary`` -- shows which frequent words the lexicon misses in each
  era. The modern blind spots are AAVE contractions and in-group terms ("'em", "gon'",
  "nigga") which the NRC norms simply do not rate.
* ``language_robustness_test`` -- all lexicons are English-only, and the non-English
  share of charting songs rises from ~1% before 2010 to 7.2% in the 2020s. Restricting
  to confidently-English, effectively monolingual songs moves the coefficient only
  from -0.303 to -0.281, so language does **not** drive the trend.

The finding that changed the conclusion
---------------------------------------
``contextual_valence_check`` is the most important test here, and it does not support
the headline sentiment claim.

On an identical set of 690 songs (10 per year), the two measures disagree:

* NRC VAD lexicon:            rho(year, valence) = **-0.221**, p = 4.6e-09
* Entailment model (context): rho(year, valence) = **-0.012**, p = 0.76

The entailment measure is not broken. Its extremes are exactly right -- highest:
"Celebration", "A Holly Jolly Christmas", "Best Day Of My Life"; lowest: "Crying",
"Broken-Hearted Melody", "Breakeven". It discriminates happy from sad songs cleanly.
It simply finds no time trend. And because both measures were computed on the *same*
songs, the disagreement cannot be explained by sampling.

The most plausible reading is that the lexicon decline reflects **vocabulary change
rather than emotional change**: modern lyrics use words the NRC norms score lower
(slang, profanity, concrete nouns) without the songs being sadder in any sense a
listener would recognise. The word-average valence result is therefore reported as
*not robust to measurement method*.

Note the asymmetry this creates in the project's conclusions. The stance results
(relationship share, "I don't need you" share) come from the entailment model, which
is context-aware and gold-validated, so they are unaffected by this critique. The
sentiment results come from context-free word norms, and they are.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .coverage import load_joined
from .fetch_lyrics import load_lyrics
from .lyrics_features import clean_lyrics, tokenize
from .paths import DERIVED, RAW, REPORTS

LEX_PATH = RAW / "lexicons" / "nrc_vad.parquet"

# Grammatical words carry no valence and are deliberately absent from the NRC norms,
# so they dominate any raw "uncovered" list and tell us nothing. Excluding them
# reveals the *content* vocabulary the lexicon is blind to, which is what determines
# whether falling coverage biases the result.
FUNCTION_WORDS = {
    "i", "you", "the", "a", "an", "and", "to", "it", "in", "on", "of", "my", "me",
    "your", "is", "that", "for", "s", "t", "with", "at", "as", "but", "so", "we",
    "he", "she", "they", "her", "his", "him", "them", "this", "was", "were", "be",
    "been", "am", "are", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "just", "all", "up", "down", "out", "no", "not", "if", "when",
    "what", "who", "how", "there", "here", "im", "i'm", "don't", "dont", "ain't",
    "aint", "its", "it's", "you're", "youre", "cause", "cos", "gonna", "wanna",
    "got", "get", "go", "let", "come", "make", "one", "now", "then", "than", "or",
    "from", "by", "about", "into", "like", "yeah", "oh", "ooh", "ah", "uh", "na",
    "la", "hey", "ya", "em", "ll", "ve", "re", "d", "m",
}

# Context-sensitive mood hypotheses for the entailment cross-check. Mood is a
# whole-song property, so these are averaged across chunks rather than maxed: unlike
# a stance claim, which can live in one line of the chorus, a song is not "happy"
# because one verse is.
MOOD_HYPOTHESES = {
    "happy": "This song is happy, joyful and upbeat.",
    "sad": "This song is sad, melancholy and downbeat.",
}


def _spearman(x, y) -> tuple[float, float]:
    r, p = stats.spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(r), float(p)


def length_strata_test(df: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Does valence still fall among songs of comparable length?

    This is the test that distinguishes a length artefact from a real change. If the
    trend only existed because songs lengthened, it would vanish inside a stratum of
    constant length.
    """
    d = df[df["n_words"].notna() & (df["n_words"] > 0) & df["vad_valence"].notna()].copy()
    d["length_bin"] = pd.qcut(d["n_words"], n_quantiles, labels=False)
    rows = []
    for b, g in d.groupby("length_bin"):
        r, p = _spearman(g["debut_year"], g["vad_valence"])
        rows.append(
            {
                "length_quintile": int(b) + 1,
                "n": len(g),
                "words_min": int(g["n_words"].min()),
                "words_max": int(g["n_words"].max()),
                "spearman_year_vs_valence": round(r, 4),
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


def type_vs_token_test(df: pd.DataFrame, min_matched: int = 10) -> dict:
    """Compare repetition-weighted valence with repetition-free valence.

    The token measure counts a repeated chorus every time it occurs, so a short,
    hook-heavy song is dominated by its hook. The type measure counts each distinct
    word once. If old songs only looked happier because they repeated happy hooks,
    the trend would weaken under the type measure.
    """
    lex = pd.read_parquet(LEX_PATH).set_index("word")["valence"]
    lex_index = set(lex.index)

    rows = []
    d = df[df["vad_valence"].notna()]
    for r in tqdm(list(d.itertuples()), desc="type-vs-token", unit="song"):
        raw = load_lyrics(r.song_id)
        if not raw:
            continue
        toks = tokenize(clean_lyrics(raw))
        if not toks:
            continue
        matched = [lex[w] for w in set(toks) if w in lex_index]
        if len(matched) < min_matched:
            continue
        rows.append(
            {
                "song_id": r.song_id,
                "debut_year": r.debut_year,
                "valence_types": float(np.mean(matched)),
                "valence_tokens": float(r.vad_valence),
            }
        )

    t = pd.DataFrame(rows)
    if t.empty:
        return {"note": "no songs available"}

    r_tok, p_tok = _spearman(t["debut_year"], t["valence_tokens"])
    r_typ, p_typ = _spearman(t["debut_year"], t["valence_types"])
    t.to_parquet(DERIVED / "valence_type_vs_token.parquet", index=False)

    return {
        "n": int(len(t)),
        "spearman_tokens": round(r_tok, 4),
        "p_tokens": p_tok,
        "spearman_types": round(r_typ, 4),
        "p_types": p_typ,
        "repetition_explains_trend": bool(abs(r_typ) < abs(r_tok) * 0.6),
    }


def coverage_drift_test(df: pd.DataFrame) -> dict:
    """Is the lexicon recognizing a shrinking share of each era's vocabulary?"""
    d = df[df["vad_coverage"].notna()].copy()
    r, p = _spearman(d["debut_year"], d["vad_coverage"])
    d["decade"] = (d["debut_year"] // 10 * 10).astype(int)
    by_decade = (
        d.groupby("decade")["vad_coverage"].agg(["size", "mean"]).reset_index()
        .rename(columns={"size": "n", "mean": "vad_coverage"})
    )
    return {
        "spearman_year_vs_coverage": round(r, 4),
        "p_value": p,
        "by_decade": by_decade.round(4).to_dict("records"),
    }


def uncovered_vocabulary(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Most frequent words the lexicon does not cover, by era.

    Lets a reader judge whether the growing uncovered mass is plausibly neutral
    (ad-libs, proper nouns, contractions) or systematically negative (profanity), the
    latter of which would mean the measure *understates* the decline.
    """
    lex_index = set(pd.read_parquet(LEX_PATH)["word"])
    eras = {"1958-1979": (1958, 1979), "1980-1999": (1980, 1999), "2000-2026": (2000, 2026)}

    rows = []
    for era, (lo, hi) in eras.items():
        sub = df[(df["debut_year"] >= lo) & (df["debut_year"] <= hi)]
        counter: Counter[str] = Counter()
        for song_id in sub["song_id"]:
            raw = load_lyrics(song_id)
            if not raw:
                continue
            counter.update(
                w
                for w in tokenize(clean_lyrics(raw))
                if w not in lex_index and w not in FUNCTION_WORDS and len(w) > 1
            )
        total = sum(counter.values()) or 1
        for word, n in counter.most_common(top_n):
            rows.append({"era": era, "word": word, "count": n, "share": n / total})
    return pd.DataFrame(rows)


def language_robustness_test(df: pd.DataFrame) -> dict:
    """Does the valence trend survive restricting to confidently-English songs?

    Every lexicon here is English-only, and the non-English share of Hot 100 hits
    rises from about 1% before 2010 to 7.2% in the 2020s. A non-English lyric does not
    score neutral -- it scores noise over a tiny matched fraction -- so a rising share
    of them could manufacture a valence trend.

    Three progressively stricter filters are applied. If the coefficient is stable
    across them, language is not driving the result.
    """
    path = DERIVED / "language_profile.parquet"
    if not path.exists():
        return {"note": "run music_tastes.language first"}

    lang = pd.read_parquet(path)[
        ["song_id", "lang", "lang_confidence", "english_token_share"]
    ]
    d = df.merge(lang, on="song_id", how="left")
    d = d[d["vad_valence"].notna()]

    filters = {
        "all_songs": d,
        "detected_english": d[d["lang"] == "en"],
        "english_confident": d[(d["lang"] == "en") & (d["lang_confidence"] > 0.99)],
        "english_monolingual": d[
            (d["lang"] == "en") & (d["english_token_share"] >= 0.99)
        ],
    }

    out = {}
    for name, sub in filters.items():
        s = sub[["debut_year", "vad_valence"]].dropna()
        if len(s) < 200:
            continue
        r, p = _spearman(s["debut_year"], s["vad_valence"])
        out[name] = {"n": int(len(s)), "spearman": round(r, 4), "p_value": p}

    taus = [v["spearman"] for v in out.values()]
    out["max_absolute_shift"] = round(max(taus) - min(taus), 4) if taus else None
    out["language_explains_trend"] = bool(
        taus and abs(max(taus) - min(taus)) > abs(min(taus, key=abs)) * 0.5
    )
    return out


def aggregation_bias_test(df: pd.DataFrame, stance: str = "independence") -> dict:
    """Correct the stance trend for a bias in the chunk-max estimator.

    Method B scores verse-sized chunks and takes the **maximum** across them, which is
    what lets it find a self-sufficiency claim that lives in one chorus. But the
    maximum of N draws rises with N even when the underlying distribution is
    unchanged, and lyrics roughly doubled in length over the period
    (rho(year, n_chunks) = +0.55). Part of the apparent rise in the "I don't need you"
    share is therefore mechanical.

    This is a different situation from lyric length and valence. There, length is a
    *mediator* -- songs really did get wordier and that is part of how music changed --
    so controlling for it would remove real signal. Here the inflation is a property
    of the estimator, not of the music, so controlling for it is the correct thing to
    do.

    Both the raw and adjusted coefficients are reported, along with the trend inside
    fixed chunk-count strata, which is the assumption-free version of the same check.
    """
    import statsmodels.api as sm

    # derive_labels emits "is_independent" for the "independence" hypothesis, so the
    # label column cannot be derived from the stance name by suffixing alone.
    label_col = {"independence": "is_independent"}.get(stance, f"is_{stance}")
    col = label_col
    if col not in df.columns or "n_chunks" not in df.columns:
        return {"note": f"missing {col} or n_chunks"}

    d = df[df[col].notna() & df["n_chunks"].notna()].copy()
    if len(d) < 300:
        return {"note": f"insufficient data (n={len(d)})"}

    y = d[col].astype(float)
    year = d["debut_year"].astype(float)
    chunks = d["n_chunks"].astype(float)

    m0 = sm.OLS(y, sm.add_constant(year)).fit()
    m1 = sm.OLS(y, sm.add_constant(pd.concat([year, chunks], axis=1))).fit()
    b0 = float(m0.params["debut_year"])
    b1 = float(m1.params["debut_year"])

    strata = []
    for lo, hi, label in [(1, 4, "short (1-4 chunks)"), (5, 7, "medium (5-7)"),
                          (8, 99, "long (8+)")]:
        g = d[(d["n_chunks"] >= lo) & (d["n_chunks"] <= hi)]
        if len(g) < 200:
            continue
        r, p = _spearman(g["debut_year"], g[col])
        by_decade = (
            g.assign(decade=(g["debut_year"] // 10 * 10).astype(int))
            .groupby("decade")[col]
            .agg(["size", "mean"])
        )
        by_decade = by_decade[by_decade["size"] >= 25]
        strata.append(
            {
                "stratum": label,
                "n": int(len(g)),
                "spearman_year": round(r, 4),
                "p_value": p,
                "by_decade": {int(i): round(float(r_["mean"]), 4)
                              for i, r_ in by_decade.iterrows()},
            }
        )

    return {
        "stance": stance,
        "n": int(len(d)),
        "spearman_chunks_vs_max": round(
            _spearman(d["n_chunks"], d[f"p_{stance}_max"])[0], 4
        ) if f"p_{stance}_max" in d.columns else None,
        "spearman_year_vs_chunks": round(_spearman(d["debut_year"], d["n_chunks"])[0], 4),
        "unadjusted_per_decade": round(b0 * 10, 5),
        "unadjusted_p": float(m0.pvalues["debut_year"]),
        "chunk_adjusted_per_decade": round(b1 * 10, 5),
        "chunk_adjusted_p": float(m1.pvalues["debut_year"]),
        "attenuation_fraction": round(1 - b1 / b0, 3) if b0 else None,
        "survives_adjustment": bool(m1.pvalues["debut_year"] < 0.05 and np.sign(b1) == np.sign(b0)),
        "within_chunk_strata": strata,
        "all_strata_positive": bool(strata) and all(s["spearman_year"] > 0 for s in strata),
    }


def length_bias_reconciliation(sample_path: str | None = None) -> dict:
    """Reconcile the lexicon and contextual valence measures by removing length bias.

    The raw comparison looked decisive: on identical songs the NRC lexicon showed a
    clear valence decline while the entailment model showed nothing (p=0.76). Read
    naively that says the lexicon result is an artefact.

    It is not that simple, because **both** measures are length-dependent and in
    opposite directions:

        rho(words, lexicon valence)    = -0.174   longer looks sadder
        rho(words, contextual valence) = +0.227   longer looks happier

    Lyrics roughly doubled in length over the period, so those biases push the two
    year-trends apart: the lexicon's decline is inflated and the contextual model's is
    masked. The apparent disagreement was largely an artefact of the comparison, not a
    finding about music.

    Opposite biases also settle whether to adjust. If wordiness genuinely made songs
    less happy, both measures would move the same way; a substantive effect cannot be
    negative in one valid measure and positive in another. Two measures disagreeing in
    sign on the same nuisance variable is the signature of measurement error, which is
    the case where adjustment is correct. (Contrast the stance analysis, where
    chunk-max is inflated by a known statistical property of maxima, and the lyric
    length/valence question, where length is a genuine mediator.)

    Adjusted for length, the two converge: about -0.09 and -0.07 standard deviations
    per decade respectively, both significant. That is a real but modest decline,
    smaller than the raw lexicon series implies and replicated across two methods with
    very different failure modes.
    """
    import statsmodels.api as sm

    path = DERIVED / (sample_path or "contextual_valence.parquet")
    if not path.exists():
        return {"note": "run contextual_valence_check first"}

    t = pd.read_parquet(path)
    idx = pd.read_parquet(DERIVED / "lyrics_index.parquet")[["song_id", "n_words"]]
    m = t.merge(idx, on="song_id")
    m = m[(m["n_words"] > 0) & m["nli_valence"].notna() & m["vad_valence"].notna()]
    if len(m) < 200:
        return {"note": f"insufficient overlap (n={len(m)})"}

    year = m["debut_year"].astype(float)
    logw = np.log(m["n_words"].astype(float))

    out: dict[str, object] = {"n": int(len(m))}
    for col, label in [("nli_valence", "contextual"), ("vad_valence", "lexicon")]:
        y = m[col].astype(float)
        sd = float(y.std()) or 1.0
        raw = sm.OLS(y, sm.add_constant(year)).fit()
        adj = sm.OLS(
            y, sm.add_constant(pd.concat([year, logw.rename("log_words")], axis=1))
        ).fit()
        out[label] = {
            "length_bias_spearman": round(_spearman(m["n_words"], y)[0], 4),
            "raw_per_decade_sd": round(float(raw.params["debut_year"]) * 10 / sd, 4),
            "raw_p": float(raw.pvalues["debut_year"]),
            "length_adjusted_per_decade_sd": round(
                float(adj.params["debut_year"]) * 10 / sd, 4
            ),
            "length_adjusted_p": float(adj.pvalues["debut_year"]),
        }

    ctx, lex = out["contextual"], out["lexicon"]
    out["biases_have_opposite_sign"] = bool(
        np.sign(ctx["length_bias_spearman"]) != np.sign(lex["length_bias_spearman"])
    )
    out["agree_after_adjustment"] = bool(
        np.sign(ctx["length_adjusted_per_decade_sd"])
        == np.sign(lex["length_adjusted_per_decade_sd"])
        and ctx["length_adjusted_p"] < 0.05
        and lex["length_adjusted_p"] < 0.05
    )
    return out


def contextual_valence_check(
    per_year: int = 12, device: str | None = None, batch_size: int = 32
) -> dict:
    """Independent, context-aware check on the lexicon's valence trend.

    Word norms cannot see negation, irony or 68 years of semantic change. An
    entailment model reads the lyric as language and can. Agreement between two
    measures with completely different failure modes is far better evidence than
    either alone.
    """
    from .stance_nli import _build_pipeline, chunk_lyrics

    df = load_joined()
    d = df[df["vad_valence"].notna()].copy()
    sample = (
        d.sort_values("points", ascending=False)
        .groupby("debut_year", group_keys=False)
        .head(per_year)
    )
    print(f"Contextual check on {len(sample):,} songs "
          f"({per_year}/year, highest exposure first)")

    clf = _build_pipeline(device, fp16=False)
    hyps = list(MOOD_HYPOTHESES.values())
    keys = list(MOOD_HYPOTHESES.keys())

    rows = []
    targets = list(sample.itertuples())
    for i in tqdm(range(0, len(targets), 16), desc="contextual", unit="batch"):
        group = targets[i : i + 16]
        flat, owner = [], []
        for song in group:
            raw = load_lyrics(song.song_id)
            if not raw:
                continue
            chunks = chunk_lyrics(clean_lyrics(raw))
            flat.extend(chunks)
            owner.extend([song] * len(chunks))
        if not flat:
            continue
        res = clf(flat, hyps, multi_label=True, batch_size=batch_size)
        if isinstance(res, dict):
            res = [res]

        acc: dict[str, dict[str, list[float]]] = {}
        for song, r in zip(owner, res):
            bucket = acc.setdefault(song.song_id, {h: [] for h in hyps})
            for label, score in zip(r["labels"], r["scores"]):
                bucket[label].append(score)
        meta = {s.song_id: s for s in group}
        for song_id, bucket in acc.items():
            song = meta[song_id]
            rec = {
                "song_id": song_id,
                "debut_year": song.debut_year,
                "vad_valence": song.vad_valence,
                "points": song.points,
            }
            for key, hyp in zip(keys, hyps):
                rec[f"nli_{key}"] = float(np.mean(bucket[hyp])) if bucket[hyp] else np.nan
            rows.append(rec)

    t = pd.DataFrame(rows)
    if t.empty:
        return {"note": "no songs scored"}
    t["nli_valence"] = t["nli_happy"] - t["nli_sad"]
    t.to_parquet(DERIVED / "contextual_valence.parquet", index=False)

    r_lex, p_lex = _spearman(t["debut_year"], t["vad_valence"])
    r_nli, p_nli = _spearman(t["debut_year"], t["nli_valence"])
    r_agree, p_agree = _spearman(t["vad_valence"], t["nli_valence"])

    t["decade"] = (t["debut_year"] // 10 * 10).astype(int)
    by_decade = (
        t.groupby("decade")
        .agg(n=("song_id", "size"), lexicon=("vad_valence", "mean"),
             contextual=("nli_valence", "mean"))
        .reset_index()
    )

    return {
        "n": int(len(t)),
        "spearman_year_vs_lexicon_valence": round(r_lex, 4),
        "p_lexicon": p_lex,
        "spearman_year_vs_contextual_valence": round(r_nli, 4),
        "p_contextual": p_nli,
        "agreement_between_measures": round(r_agree, 4),
        "p_agreement": p_agree,
        "same_direction": bool(np.sign(r_lex) == np.sign(r_nli)),
        "by_decade": by_decade.round(4).to_dict("records"),
    }


def run(skip_contextual: bool = False, per_year: int = 12) -> dict:
    df = load_joined()

    # The contextual check takes ~40 minutes of GPU time, so a --skip-contextual run
    # must not discard a previous result. Start from whatever is already on disk and
    # overwrite only the tests actually re-run.
    out_path = REPORTS / "validity.json"
    results: dict[str, object] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            results = {}

    print("=== Test 1: does the decline survive within length strata? ===")
    strata = length_strata_test(df)
    print(strata.to_string(index=False))
    strata.to_csv(REPORTS / "validity_length_strata.csv", index=False)
    results["length_strata"] = strata.to_dict("records")
    same_sign = (strata["spearman_year_vs_valence"] < 0).all()
    all_sig = (strata["p_value"] < 0.05).all()
    print(f"  -> all quintiles negative: {same_sign}; all significant: {all_sig}")
    print("  -> a pure length artefact would NOT survive this test.\n")

    print("=== Test 2: is the lexicon covering less of each era's vocabulary? ===")
    cov = coverage_drift_test(df)
    for row in cov["by_decade"]:
        print(f"  {row['decade']}s  coverage={row['vad_coverage']:.3f}  n={row['n']}")
    print(f"  -> rho(year, coverage) = {cov['spearman_year_vs_coverage']:+.3f} "
          f"(p={cov['p_value']:.2g})\n")
    results["coverage_drift"] = cov

    print("=== Test 3: repetition-free (word types) vs repetition-weighted ===")
    tvt = type_vs_token_test(df)
    if "spearman_types" in tvt:
        print(f"  tokens: rho={tvt['spearman_tokens']:+.3f}  "
              f"types: rho={tvt['spearman_types']:+.3f}  (n={tvt['n']:,})")
        print(f"  -> repetition explains the trend: {tvt['repetition_explains_trend']}\n")
    results["type_vs_token"] = tvt

    print("=== Test 4: what vocabulary is the lexicon missing? ===")
    unc = uncovered_vocabulary(df)
    unc.to_csv(REPORTS / "validity_uncovered_vocabulary.csv", index=False)
    for era in unc["era"].unique():
        top = unc[unc["era"] == era].head(12)["word"].tolist()
        print(f"  {era}: {', '.join(top)}")
    print()
    results["uncovered_vocabulary_written"] = str(REPORTS / "validity_uncovered_vocabulary.csv")

    print("=== Test 5: is the trend an artefact of non-English songs? ===")
    langtest = language_robustness_test(df)
    if "all_songs" in langtest:
        for name in ("all_songs", "detected_english", "english_confident",
                     "english_monolingual"):
            v = langtest.get(name)
            if v:
                print(f"  {name:22} n={v['n']:5}  rho={v['spearman']:+.4f} "
                      f"p={v['p_value']:.2g}")
        print(f"  -> language explains the trend: "
              f"{langtest.get('language_explains_trend')}\n")
    else:
        print(f"  {langtest.get('note')}\n")
    results["language_robustness"] = langtest

    print("=== Test 6: is the stance trend inflated by chunk-max aggregation? ===")
    from .analysis_trends import derive_labels

    agg = aggregation_bias_test(derive_labels(df))
    if "unadjusted_per_decade" in agg:
        print(f"  rho(year, n_chunks) = {agg['spearman_year_vs_chunks']:+.3f}; "
              f"rho(n_chunks, p_max) = {agg['spearman_chunks_vs_max']:+.3f}")
        print(f"  unadjusted      {agg['unadjusted_per_decade']:+.4f}/decade "
              f"(p={agg['unadjusted_p']:.2g})")
        print(f"  chunk-adjusted  {agg['chunk_adjusted_per_decade']:+.4f}/decade "
              f"(p={agg['chunk_adjusted_p']:.2g})")
        print(f"  -> attenuation {agg['attenuation_fraction']:.0%}; "
              f"survives: {agg['survives_adjustment']}")
        for s in agg["within_chunk_strata"]:
            trail = " ".join(f"{k}s:{v:.0%}" for k, v in s["by_decade"].items())
            print(f"    {s['stratum']:20} n={s['n']:5} rho={s['spearman_year']:+.3f}  {trail}")
    else:
        print(f"  {agg.get('note')}")
    print()
    results["aggregation_bias"] = agg

    if not skip_contextual:
        print("=== Test 7: context-aware cross-check (entailment model) ===")
        ctx = contextual_valence_check(per_year=per_year)
        if "spearman_year_vs_contextual_valence" in ctx:
            print(f"  lexicon    rho(year, valence) = "
                  f"{ctx['spearman_year_vs_lexicon_valence']:+.3f} "
                  f"(p={ctx['p_lexicon']:.2g})")
            print(f"  contextual rho(year, valence) = "
                  f"{ctx['spearman_year_vs_contextual_valence']:+.3f} "
                  f"(p={ctx['p_contextual']:.2g})")
            print(f"  agreement between measures    = "
                  f"{ctx['agreement_between_measures']:+.3f}")
            print(f"  -> same direction: {ctx['same_direction']}")
        results["contextual_check"] = ctx

    recon = length_bias_reconciliation()
    if "contextual" in recon:
        print("=== Test 8: reconciling the two valence measures for length bias ===")
        for label in ("lexicon", "contextual"):
            v = recon[label]
            print(f"  {label:12} length-bias rho={v['length_bias_spearman']:+.3f}  "
                  f"raw={v['raw_per_decade_sd']:+.3f} SD/dec (p={v['raw_p']:.2g})  "
                  f"adjusted={v['length_adjusted_per_decade_sd']:+.3f} "
                  f"(p={v['length_adjusted_p']:.2g})")
        print(f"  -> biases have opposite sign: {recon['biases_have_opposite_sign']}; "
              f"agree after adjustment: {recon['agree_after_adjustment']}")
    else:
        print(f"=== Test 8: {recon.get('note')}")
    print()
    results["length_bias_reconciliation"] = recon

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-contextual", action="store_true")
    ap.add_argument("--per-year", type=int, default=12)
    args = ap.parse_args()
    run(skip_contextual=args.skip_contextual, per_year=args.per_year)
