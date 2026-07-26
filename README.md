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

## How it answers the question

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

**Reported:** lyric valence and emotion trends; relationship-share and stance shares
(independence, heartbreak, devotion); coverage audit; chart ingestion; song
resolution; classifier validation.

**In progress:** lyric fetching and stance scoring are still extending coverage; the
acoustic stage (BPM, key, Essentia mood) is running. Results refresh by re-running
`coverage`, `trends` and `report`.

**Known limitations carried into the writeup:**

- Spotify's audio-features endpoint was deprecated for new applications in November
  2024, so BPM comes from AcousticBrainz. Its features are community-submitted, and
  Essentia BPM estimates suffer octave errors — "Hey Jude" is reported at 128 BPM
  against a true tempo near 74.
- Genre mix is uncontrolled. A shift toward genres with different lyrical conventions
  is a live rival explanation for the valence decline.
- The Genius search API enforces a quota that a full 32k run exhausts. The fetcher
  falls back to verified slug URLs (`--no-api`), which resolve 82.9% of songs at tier
  A versus 93.8% for the API.

