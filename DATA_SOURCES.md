# Data sources

Every observation in this project comes from one of the sources below. Historical
gaps are filled only from source records, never imputed or interpolated; where a
source has no value for a song, the value stays null and is reported as missing coverage.

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

Current agreement, artist/title disagreements, missing positions, and retrieval
dates are recorded in the cross-check result. Secondary-only complete chart weeks
may extend the primary archive; `chart_source` records their origin. Do not infer
the latest available issue date from the primary mirror alone.

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
Without a token, candidate song-page URLs include leading-article, dotted-initial,
and optional quoted-nickname variants; the returned page must still pass the
title-and-artist identity checks.
Failed matches can be retried selectively, including cached negative responses.
Short lyric fragments are not treated as confirmed instrumental recordings.
An instrumental section marker within sung words does not make the entire
recording instrumental; the text-only heuristic requires a marker and no words
outside section labels.

**Copyright handling.** Lyrics are copyrighted. Raw lyric text and full HTTP
responses stay under gitignored `data/`, including `data/cache/lyrics_cache/`
and isolated recovery directories. No lyric text is committed to this repository
or reproduced in any report; only derived numeric features and aggregates are
published.

Match quality is recorded per song in `data/derived/lyrics_index.parquet` as a
confidence tier:

| Tier | Rule | Purpose |
|---|---|---|
| A | title similarity >= 0.85 and artist similarity >= 0.72 | safe on its own |
| B | title >= 0.92 and artist >= 0.50 | catches artist renames (Lady Antebellum -> Lady A) |
| C | title >= 0.96, artist >= 0.30, top-3 search hit | last resort; can be excluded in sensitivity checks |

**LRCLIB fallback.** The public LRCLIB API supports anonymous title/artist search.
The optional `--lrclib` fallback accepts only exact normalized title and artist
matches with one unambiguous plain-text result across candidates. Different lyric
bodies are rejected rather than guessed: only case, punctuation, spacing,
and bracketed section labels are ignored when comparing otherwise identical word
sequences. Word order and repetitions must agree. Without duration or recording IDs this
is a metadata-level match, not a verified recording identity; the caveat and
source record ID are saved separately from Genius confidence tiers. Agreement
is checked among returned candidates; the API caps searches at 20 results and
does not support pagination, so this is not an exhaustive recording search. Its HTTP
responses and lyric bodies remain in the private data cache, never public reports.
If every exact metadata candidate explicitly marks a recording instrumental and
none contains plain or synchronized lyrics, that provider evidence is recorded
as an instrumental resolution. No lyric body is fabricated, and it does not
increase lyric-retrieval coverage. Conflicting or absent flags are not sufficient.
Documentation: <https://lrclib.net/docs>.

**Recording corrections.** Page identity alone cannot establish that its words occur
in the charted recording. The recovery audit found words attached to instrumental
recordings: The Tommy Dorsey Orchestra's "Tea For Two Cha Cha", C.C.S.'s "Whole
Lotta Love", Billy Vaughn And His Orchestra's "Blue Hawaii", and the Mar-Keys'
"Morning After". The expanded source-linked registry is
[`recording_instrumentals.csv`](src/music_tastes/recording_instrumentals.csv),
supplemented by explicit review/award evidence in `fetch_lyrics.py`.
Exact normalized title-and-artist corrections take precedence
over provider transcriptions, resolve these as instrumentals with null lyric bodies,
and exclude them from text inference and retrieval coverage. An existing rejected
transcription is retained only in the private `rejected_recordings/` cache directory.
The corrections do not transfer to other artists or live versions.
Registry notes distinguish primary label/artist evidence from secondary
single-discography annotations. The ERIC export accepts original-hit catalog
entries with chart-compatible dates; explicit album versions and re-recordings
are excluded. Reviewed billing differences are enumerated, not a general
relaxation of provider matching.

These are source-based recording classifications, not a claim that every
recording was independently listened to or contains zero speech. Instrumental
hits may include brief chants, harmonies, or spoken introductions. Full
composition lyrics from a vocal cover are not therefore a valid transcription.
Genuine vocal-backed recordings, such as Duane Eddy's "Lonely Boy, Lonely
Guitar", are not excluded based on the performer's instrumental reputation.
Unresolved recording-review candidates remain uncertain, not automatic
instrumentals. Corrections can lower previously overstated lyric coverage;
the recovery audit separates added texts, removed transcriptions, and
unchanged-body eligibility repairs.

- ERIC Records, *Hard To Find Pop Instrumentals*, volume 1, track 9:
  <https://www.ericrecords.com/pop_1.html>.
- RAK/Columbia advertisement, *Record World*, January 23, 1971, page 13:
  <https://www.worldradiohistory.com/Archive-All-Music/Record-World/70s/71/RW-1971-01-23.pdf>.
- "Blue Hawaii" single catalog (instrumental flag):
  <https://www.musik-sammler.de/release/billy-vaughn-blue-hawaii-7-225359/>.
  *Billboard*, May 25, 1959, page 3, also identifies the album as instrumental:
  <https://www.worldradiohistory.com/Archive-All-Music/Billboard/50s/1959/Billboard%201959-05-25.pdf>.
- *Billboard*, September 18, 1961, single review for Stax 112, explicitly
  identifies "Morning After" as instrumental:
  <https://www.worldradiohistory.com/Archive-All-Music/Billboard/60s/1961/Billboard%201961-09-18.pdf>.
- ERIC Records' original-hit instrumental catalogs: the registry records
  the catalog URL, track, title, billing, and chart date for each accepted entry.
- ACE Records, *Teen Beat*, identifies Sandy Nelson and Cozy Cole's recordings:
  <https://acerecords.co.uk/teen-beat>.
- Music VF's secondary per-single catalogs explicitly mark nine additional
  recordings instrumental, with label/catalog numbers and URLs in the registry.
  An absent annotation is not evidence that a recording has or lacks vocals.
- Global Dog Productions' secondary Stax single discography identifies
  "Chinese Checkers" (Stax 137) and "Philly Dog" (Stax 185) as instrumental:
  <https://www.globaldogproductions.net/s/satellite-stax.html>.
- Mason Williams' own composition catalog identifies his 1969 "Greensleeves"
  arrangement as instrumental; his discography identifies the chart single:
  <https://www.masonwilliams-online.com/careermusic.html> and
  <https://www.masonwilliams-online.com/careerdisc.html>.
- Hugh Masekela's official biography identifies his 1968 "Grazing In The Grass"
  single as instrumental: <https://hughmasekela.co.za/biography/>.
- Recording Academy entries distinguish the instrumental recordings of
  "Taste Of Honey" and "Mercy, Mercy, Mercy" from vocal versions:
  <https://www.grammy.com/artists/herb-alpert/1256> and
  <https://www.grammy.com/artists/cannonball-adderley/53/>.
- The explicit registry entries also cite contemporary *Billboard* and *Cash Box*
  evidence for "My Sweet Potato", "The Happening", "Last Tango In Paris", and
  "Chase The Clouds Away". The source URLs and recording-specific notes are
  retained alongside each correction.

---

## Sentiment and emotion lexicons

Downloaded at build time by `music_tastes.lexicons`; provenance including retrieval
timestamps is written to `data/raw/lexicons/provenance.json`. NRC lexicons are
not redistributed in this repository.

- Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal, and
  Dominance for 20,000 English Words. *Proceedings of ACL 2018*. National Research
  Council Canada. (NRC VAD Lexicon; 19,970 terms used.)
- Mohammad, S. M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, and
  Dominance for over 55k English Terms. *arXiv:2503.23547*.
  <https://arxiv.org/abs/2503.23547>. The separate v2.1 robustness analysis imports
  54,801 words/phrases from the official research archive at
  <https://saifmohammad.com/WebPages/nrc-vad.html>. Its native -1 to +1 scores
  are explicitly rescaled to 0-1 for comparison with v1. The lexicon is not
  redistributed. Source, version, import time and archive/normalized hashes are
  recorded in `data/raw/lexicons/nrc_vad_v21.provenance.json`.
- Mohammad, S. M. (2025). Breaking Bad: Norms for Valence, Arousal, and Dominance
  for over 10k English Multiword Expressions. *Proceedings of IJCNLP-AACL 2025*.
  <https://aclanthology.org/2025.ijcnlp-long.107/>.
  The NRC resource page additionally requests this citation when using
  multiword expressions, as in the phrase-matching sensitivity.
- Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a Word–Emotion Association
  Lexicon. *Computational Intelligence*, 29(3), 436–465. National Research Council
  Canada. (NRC Emotion Lexicon / EmoLex; 14,153 terms used.)
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. *Proceedings of ICWSM-14*. (7,494 terms
  used.)

The VADER series here is a token-frequency mean of dictionary ratings, not
the full rule-based analyzer or its compound score. NRC VAD dimensions and
EmoLex emotion-associated token shares are also lexical measures, not direct
whole-song mood or listener-emotion labels. See the version/scoring, coverage,
and matching sensitivities in `METHODS.md`.

---

## Models

Both run locally; no lyric text is sent to any third-party service.
See `METHODS.md` for the precise relationship definition, 380-word input limit,
threshold, conditional stance denominator, and independent-review requirements.

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
  New submissions stopped in 2022. Missing contemporary features are a source
  limitation, not evidence that contemporary music has no trend.
- Bogdanov, D., Wack, N., Gómez, E., et al. (2013). ESSENTIA: an Audio Analysis Library
  for Music Information Retrieval. *ISMIR 2013*. (The analysis library behind
  AcousticBrainz.)

**Coverage and caveats.** AcousticBrainz features are keyed to a specific *recording*,
and a hit song usually has several recording MBIDs of which only some carry a
submission. Querying only the best-matching MBID yielded BPM for about a tenth of attempted
songs; querying every acceptable candidate raised that severalfold. Current coverage
is recorded in `data/derived/acoustic_features.parquet` and summarised by
`music-tastes coverage`. Coverage remains uneven and is
subject to the same year-dependence check as lyrics. Essentia BPM estimates are also
prone to octave errors (reporting double or half the true tempo), so BPM results are
reported with that caveat rather than as exact tempi.
