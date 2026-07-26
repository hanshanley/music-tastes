# music-tastes

Are US hit songs getting sadder? Are fewer of them about love — and among the ones
that are, are more of them about *not needing* the relationship?

This is a reproducible research pipeline over every song that entered the Billboard
Hot 100 from 1958 to the present (354,687 chart positions, 32,602 unique songs). It
runs entirely on your machine: the only network calls fetch chart data, lyrics and
public lexicons.

## Quick start

```bash
uv venv --python 3.12
uv pip install -e .
cp .env.example .env      # then add your Genius token

music-tastes ingest-charts
music-tastes resolve-songs
music-tastes exposure
music-tastes lexicons
music-tastes fetch-lyrics      # long; resumable
music-tastes features-a
music-tastes stance-b          # long; resumable, uses local GPU if present
music-tastes coverage
music-tastes trends
music-tastes report
```

Every stage is resumable — network responses and model outputs are cached on disk, so
an interrupted run picks up where it stopped and a completed stage costs nothing to
re-run.

## Cost: zero

Everything runs locally. There is no paid API in this pipeline and no key to
configure — `.env` holds only free credentials (a Genius token and a contact string
for MusicBrainz's User-Agent policy).

| Component | Where it runs | Cost |
|---|---|---|
| Stance classifier (`deberta-v3-large-zeroshot-v2.0`, 435M params) | your GPU, via MPS/CUDA | free |
| Theme anchors (`all-MiniLM-L6-v2`, 22M params) | your GPU | free |
| Billboard charts, MusicBrainz, AcousticBrainz, NRC lexicons | public endpoints | free |
| Genius lyrics | free API tier + public pages | free |

**On model size.** 435M parameters is small — roughly 1/300th of a frontier LLM — and
it runs comfortably on a laptop. It is not chosen for grandeur: the smaller
`deberta-v3-base` was measured on the validation set first and scored **5/13** against
large's **13/13** on stance, which is the distinction the whole project turns on. The
real cost here is wall-clock time (~3 s/song), not money.

If you want it faster, `music-tastes stance-b --limit N` scores a year-balanced prefix,
so a partial run is still a usable year-stratified sample.

**Exposure.** A song's weight is its chart tenure scored by position (a #1 week is
worth 100, a #100 week is worth 1). Weights are normalized *within* year, because
chart tenure has inflated from ~13 weeks in the 1960s to 90+ weeks today and would
otherwise let the streaming era swamp every pooled statistic. Unweighted results are
always reported alongside.

**Two independent classifiers, because the hard part is stance, not mood.**

- *Method A* — NRC VAD / EmoLex / VADER word norms, plus cosine similarity to
  hand-written theme anchors. Transparent and auditable.
- *Method B* — a local zero-shot NLI model (`deberta-v3-large-zeroshot-v2.0`) scoring
  entailment of explicit claims such as *"The singer does not need this person and
  will be fine without them."*

On a 13-song set with uncontroversial stances, **Method A scored 6/13 and Method B
scored 13/13**. Method A reliably finds devotion, longing and heartbreak but
systematically labels independence songs ("I Will Survive", "Since U Been Gone",
"thank u, next") as heartbreak: cosine similarity measures *topic* — this is a breakup
song — while entailment can capture *stance*. Method A is retained as a baseline with
that weakness measured, not hidden.

Two design choices carry Method B's accuracy, both validated:

- **Chunk-level maximum.** The self-sufficiency claim usually lives in the chorus and
  is diluted across a whole lyric. Scoring verse-sized chunks and taking the max
  lifted "I Will Survive" from 0.33 to 0.97 and "Since U Been Gone" from 0.05 to 0.85
  while producing no false positive (controls stayed between 0.004 and 0.082).
- **No relationship gate.** Stance is scored for every song, because "Survivor" scores
  only 0.26 on "is a relationship song" and a hard gate would discard it.

**Coverage is treated as a gate, not a footnote.** Genius transcription coverage
correlates strongly with year, so a "trend" can be manufactured by which songs happen
to be present. `music-tastes coverage` measures that correlation and builds a
complete-case subset with a constant number of songs per year; every headline result
is re-run on it, and any result whose direction flips is reported as unresolved.

## Tests

```bash
uv pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q
```

56 tests covering song-identity normalization, Genius matching (including the
accept/reject boundary of the slug fallback), and the HTTP rate limiter. The rate
limiter tests exist because of a real incident: per-thread throttling let four workers
issue four simultaneous requests, exhausted the Genius quota, and silently produced
zero successful fetches for six hours.

## Repository layout

```
src/music_tastes/
  ingest_charts.py   Hot 100 download + independent cross-check
  resolve_songs.py   title/artist normalization and song identity
  exposure.py        chart-points weights and the methodology era table
  lexicons.py        NRC VAD / EmoLex / VADER with recorded provenance
  fetch_lyrics.py    Genius matching (tiered) and lyric caching
  taxonomy.py        the stance codebook shared by both methods
  lyrics_features.py Method A
  stance_nli.py      Method B
  coverage.py        the coverage gate
  analysis_trends.py bootstrap CIs, Mann-Kendall, Theil-Sen
  report.py          figures and findings.md
data/                all gitignored
reports/             findings.md, figures, coverage tables
```

## Copyright and data handling

Lyrics are copyrighted. Raw lyric text is written **only** to
`data/cache/lyrics_cache/`, which is gitignored; nothing downstream emits lyric text,
and no lyric text appears in this repository or in any generated report. Chart data is
not redistributed either — only the code that fetches it and derived aggregates.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) for full citations, licence notes and the
sources that were evaluated and rejected.

## Status

**Robust in direction, not in level.** The "I don't need you" stance among
relationship songs rises inside every lyric-length stratum, survives coverage, genre
and era controls, and replicates across 4 of 5 reworded hypotheses (the 5th fires on
0.8% of songs and has no power). Two caveats matter:

- Method B's chunk-**max** aggregation is inflated by lyric length (lyrics doubled;
  ρ(year, chunks)=+0.57) — an estimator artifact, not music. Adjusting halves the
  slope: **+1.4 points per decade, not +2.7**.
- The *absolute share* ranges 0.8%–14.8% depending purely on how the hypothesis is
  worded. **Quote the trend, never the level.**

Relationship share itself is flat.

**Not robust — reported with the caveat:** word-average lyric valence and joy. On an
identical 690 songs, NRC VAD word norms give ρ(year, valence) = −0.221 (p=4.6e-09)
while a context-aware entailment model gives **−0.012 (p=0.76)**. The entailment
measure discriminates mood cleanly (top: *Celebration*, *Holly Jolly Christmas*;
bottom: *Crying*, *Broken-Hearted Melody*) — it simply finds no trend. The lexicon
decline most likely tracks *vocabulary* change, not *emotional* change. Run
`music-tastes validity` to reproduce.

**Ruled out as explanations:** non-English songs (ρ moves only −0.303 → −0.281 under
the strictest English filter), code-switching (English-token share flat at 0.995–0.999
across decades), lyric length (decline persists within every length quintile, and
strengthens when repetition is removed), and chart-methodology era.

**Musical (audio) findings — all weak.** Tempo is **flat** (τ=−0.12, p=0.15; hits are
not getting faster or slower). Minor-key share doubles (13.4% → 36.1%) but genre mix
explains ~52% of it, within-genre signs disagree, and it vanishes post-1991 — so it
reflects the genre mix shifting, not songwriting. Essentia's `mood_happy` falls
sharply, but `mood_sad` falls too; two opposing classifiers moving the same way is
diagnostic of drift (production/mastering change), not emotion, so neither is reported
as evidence about mood.

**Genre mix:** tested — accounts for only ~12% of the lexicon valence trend.

**Other limitations:** Spotify's audio-features endpoint was deprecated for new apps in
November 2024, so BPM comes from AcousticBrainz, whose Essentia estimates suffer octave
errors ("Hey Jude" reads 128 BPM against a true tempo near 74). The Genius search API
enforces a quota a full 32k run exhausts; the fetcher falls back to verified slug URLs
(`--no-api`), resolving 82.9% of songs at tier A versus 93.8% for the API.


