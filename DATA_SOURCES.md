# Data sources

Every observation in this project comes from one of the sources below. Nothing is
imputed, interpolated or back-filled; where a source has no value for a song, the
value stays null and is reported as missing coverage.

Access dates are recorded per artefact in the machine-readable provenance files noted
in each section.

---

## Chart data

**Billboard Hot 100.** Compiled by Billboard from sales, radio airplay and streaming
data supplied by Luminate (formerly Nielsen SoundScan and Broadcast Data Systems).
Billboard is the originating publisher; the archives below are third-party mirrors,
which is why two independent ones are cross-checked against each other.

- Primary: Hollingshead, M. *billboard-hot-100* [data set]. GitHub.
  <https://github.com/mhollingshead/billboard-hot-100> — JSON, one document per
  weekly chart, 1958-08-04 to present.
- Cross-check: University of Texas at Austin, School of Journalism and Media
  (`utdata`). *rwd-billboard-data* [data set]. GitHub.
  <https://github.com/utdata/rwd-billboard-data> — CSV.

Provenance file: `data/raw/billboard_hot100_all.provenance.json`
Cross-check result: `data/interim/chart_crosscheck.json`

Agreement between the two archives was 99.66% of shared chart positions
(1,218 disagreements out of 354,558). Inspected disagreements were almost entirely
diacritic differences (`volare` vs `volaré`) plus a small number of 1958 tie-position
quirks.

> Billboard chart data is used here for non-commercial research. Billboard and
> Luminate assert rights over the charts; this repository redistributes no chart data,
> only code that fetches it and derived aggregate statistics.

---

## Lyrics

**Genius.** Lyric transcriptions are user-contributed and licensed by Genius Media
Group from rights holders.

- Genius Media Group, Inc. *Genius API* and song pages. <https://genius.com>

Search is performed through the official API (which requires a client access token);
the lyric body is read from the public song page, which `genius.com/robots.txt`
permits for generic user agents. Requests are rate limited to roughly 3/second and
cached on disk so re-runs generate no traffic.

**Copyright handling.** Lyrics are copyrighted. Raw lyric text is written only to
`data/cache/lyrics_cache/`, which `.gitignore` excludes. No lyric text is committed to
this repository or reproduced in any report; only derived numeric features and
aggregates are published.

Match quality is recorded per song in `data/derived/lyrics_index.parquet` as a
confidence tier:

| Tier | Rule | Purpose |
|---|---|---|
| A | title similarity >= 0.85 and artist similarity >= 0.72 | safe on its own |
| B | title >= 0.92 and artist >= 0.50 | catches artist renames (Lady Antebellum -> Lady A) |
| C | title >= 0.96, artist >= 0.30, top-3 search hit | last resort; can be excluded in sensitivity checks |

---

## Sentiment and emotion lexicons

Downloaded at build time by `music_tastes.lexicons`; provenance including retrieval
timestamps is written to `data/raw/lexicons/provenance.json`. Neither NRC lexicon is
redistributed in this repository.

- Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and
  Dominance for 20,000 English Words. *Proceedings of ACL 2018*. National Research
  Council Canada. (NRC VAD Lexicon; 19,970 terms used.)
- Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word–Emotion Association
  Lexicon. *Computational Intelligence*, 29(3), 436–465. National Research Council
  Canada. (NRC Emotion Lexicon / EmoLex; 14,153 terms used.)
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. *Proceedings of ICWSM-14*. (7,494 terms
  used.)

---

## Models

Both run locally; no lyric text is sent to any third-party service.

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks. *EMNLP 2019*. Model: `sentence-transformers/all-MiniLM-L6-v2`.
  Used for Method A theme anchors.
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). Building Efficient
  Universal Classifiers with Natural Language Inference. Model:
  `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`. Used for Method B stance
  classification.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: Improving DeBERTa using
  ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *ICLR 2023*.
  (Underlying architecture.)

---

## Sources evaluated and not used

- **Spotify Web API audio-features** (tempo, valence). Deprecated for newly registered
  applications on 2024-11-27, so it is unavailable to this project.
- **GetSongBPM**. Reachable only behind a Cloudflare challenge and requires an API key
  plus a referrer header; not used.
- **ReccoBeats**. API confirmed live and ISRC-joinable; retained as a documented
  fallback but not currently needed.

## Acoustic features

- MetaBrainz Foundation. *MusicBrainz* [database]. <https://musicbrainz.org> — used to
  resolve chart songs to recording MBIDs and ISRCs. Accessed at the published limit of
  one request per second with a descriptive User-Agent.
- MetaBrainz Foundation. *AcousticBrainz* [data set]. <https://acousticbrainz.org> —
  community-submitted Essentia analysis providing `rhythm.bpm`, `tonal.key_key`,
  `tonal.key_scale` and high-level mood classifiers.
- Bogdanov, D., Wack, N., Gómez, E., et al. (2013). ESSENTIA: an Audio Analysis Library
  for Music Information Retrieval. *ISMIR 2013*. (The analysis library behind
  AcousticBrainz.)

**Coverage and caveats.** AcousticBrainz features are keyed to a specific *recording*,
and a hit song usually has several recording MBIDs of which only some carry a
submission. Querying only the best-matching MBID yielded BPM for 10% of songs;
querying every acceptable candidate yielded 68%. Coverage remains uneven and is
subject to the same year-dependence check as lyrics. Essentia BPM estimates are also
prone to octave errors (reporting double or half the true tempo), so BPM results are
reported with that caveat rather than as exact tempi.

