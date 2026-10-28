# Methods and reproduction

This is a retrospective analysis through the verified chart-issue cutoff
recorded in the completed run manifest, not a continuously updated service.
Later chart issues are outside that snapshot unless a new analysis is run
explicitly. The retrieval date and chart issue date are recorded separately.

## What “about a relationship” means

The topic is **romantic or sexual relationships**, not all forms of love. Include
commitment, attraction, longing, breakup, rejection, conflict, and casual sex.
Exclude songs solely about friendship, family, religion, self-worth, partying,
politics, or material success. An incidental romantic word does not establish
the topic. Third-person stories qualify if the relationship is substantive.
Mixed-topic songs may qualify; genuinely indeterminate examples should be
marked `uncertain`, not forced negative.

The production baseline is a local natural-language inference model:
`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, pinned to the revision in
`stance_nli.py`. It scores the hypothesis:

> This song is about a romantic or sexual relationship.

It reads the **first 380 cleaned words**, not necessarily the complete lyric.
An entailment score **strictly greater than 0.5** is labeled positive. These
scores have not been established as calibrated probabilities. Missing lyrics,
unscored songs, and failed inference are **missing**, not negative examples.
English eligibility remains a recorded stopword/ASCII heuristic, audited by
the separate language stage; it is not guaranteed language identification.
A whole-text instrumental marker is distinguished from a short text fragment.
An `[Instrumental]` section within otherwise vocal text does not exclude the
entire song. Cached exclusions caused by that old rule are repaired without
changing the lyric body, source, or retrieval time. These eligibility repairs
are counted separately from newly retrieved lyrics. Section labels and the
language heuristic do not independently establish recording identity or
distinguish sung words from spoken introductions.

Stance is a separate task: the model scores independent hypotheses on up to
12 unique 70-word chunks and retains the maximum for each stance. The model
scores stance even for non-relationship songs. Aggregation then restricts
stance shares to relationship-positive songs. A topic error can therefore
affect both the numerator and denominator of a stance trend.

Method A (word norms and embedding anchors) is a baseline, not ground truth.
The relationship threshold sensitivity is published separately. Production
changes to hypothesis wording, truncation, or thresholds require held-out
validation and a versioned before/after comparison.
The relationship wording/input sensitivity uses the same era-stratified songs
for two alternative hypotheses and the baseline. Wording comparisons hold the
first 380 words constant. Broader input uses up to 12 unique 380-word windows
and reports both their maximum and mean score. This changes aggregation as well
as coverage; it is not an untruncated-document accuracy benchmark. Tokenizer
limits still apply. The comparisons do not change production labels.

## Independent review

```bash
uv run --locked --extra all music-tastes review-packet --per-decade 20
```

This writes a blind, seeded, equal-decade sample to
`data/derived/review_packet.csv`. Existing packets are never overwritten.
Sampling probabilities and decade population counts are recorded. A separate
`review_challenge.csv` contains up to 10 additional examples per decade selected
for borderline topic scores, model disagreement, short/truncated inputs, or
historical fallback matches; these rows never enter the random-sample metrics.
Reviewers should read the locally cached lyric, not infer the topic from the
title or accept a model suggestion. Record `gold_relationship` as `0`, `1`, or
`uncertain`, plus a pseudonymous `reviewer_id` and a brief original rationale.
Do not copy lyric passages into labels or reports. Independently double-review
ambiguous cases and a subset of routine cases, and document adjudication.
Optional `gold_independence`, `gold_devotion`, `gold_longing`, `gold_heartbreak`,
`gold_casual`, and `gold_conflict` columns use the same `0`/`1`/`uncertain` rubric
for the corresponding narrator claims. Stances are independent, not exclusive.
Conditional stance evaluation includes the production topic gate within
human relationship-positive songs. Ungated independence is reported separately.
Unfilled stance labels remain pending, never negative.

Keep deliberately selected borderline/model-disagreement examples in a
separate challenge set. Do not mix them into prevalence or population-accuracy
estimates. Equal-decade sample accuracy is also not population-weighted;
publish per-decade results and sample sizes. The scoring command is:

```bash
uv run --locked --extra all music-tastes gold-set
```

The legacy check contains 32 purported random songs, of which two were
uncertain, and 13 purposively selected independence examples. Its sample and
annotation provenance have not been independently re-audited. A good result
on those examples does not establish era-wide accuracy. Missing model scores
are excluded rather than counted as negative predictions. No independent
human review is claimed until actual labels are supplied.

## Data, coverage, and time

Chart issue dates are not retrieval dates and can be later than publication
dates. Refresh checks both archives, records disagreements, validates complete
secondary-only weeks, and preserves their source identity. The December 25,
1961 to January 6, 1962 transition changes chart issue weekdays in both
archives; it is not treated as an ordinary missing-week gap. Other gaps are
reported rather than imputed.

Debut-year trends describe cohorts using observed lifetime chart points.
Chart-year comparisons use each song's points in each calendar year instead.
Neither is a census of listening or a causal estimate. The current year and
decade are incomplete; recent cohorts also have shorter chart follow-up.
The fixed-follow-up comparison uses only the first 52 calendar weeks of each
song's chart points and excludes entire debut-year cohorts until every song in
that cohort has a full observation window.
Bootstrap intervals resample songs within a year. Simple rank tests do not
correct for every source of serial dependence or the many exploratory tests.

Report song coverage, exposure coverage, English/non-instrumental eligibility,
and successful feature coverage separately. The equal-N, high-exposure,
covered-song subset is a sensitivity check, not a fully observed unbiased
sample. Direction agreement does not prove that missingness bias is absent.
The gap/change audit compares available public snapshots by year and decade.
Their chart and lyric populations can differ; a change in coverage is not a
matched-song count of recovered lyrics. Legacy outputs without a manifest have
no verified chart cutoff.
When its lyric-index fingerprint matches current inputs, `reports/gap_audit.json`
also includes a matched recovery audit: added texts, removed recording
transcriptions, unchanged-body eligibility repairs, and bounded retry outcomes.
These are separate from differences between public snapshots. Deferred retries
and unsuccessful provider searches do not establish global lyric unavailability.
AcousticBrainz's historical features cannot establish current audio trends;
it stopped accepting submissions in 2022.

## NRC VAD sentiment robustness

The original NRC VAD v1 (2018) stays the production baseline. A separate
`vad-robustness` stage compares it with NRC VAD v2.1 (2025), created by
Saif M. Mohammad at NRC Canada. Cite both the 2018 ACL paper and the 2025
paper, *NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for
over 55k English Terms* (arXiv:2503.23547). The phrase-matching sensitivity
also uses the multiword-expression norms described in Mohammad's
*Breaking Bad* (IJCNLP-AACL 2025), as requested by the resource's citation
instructions; see `DATA_SOURCES.md`. The archive stays private.

```bash
uv run --locked --extra all music-tastes vad-robustness \
  --vad-archive "/path/to/NRC-VAD-Lexicon-v2.1.zip" --offline
uv run --locked --extra all music-tastes update --offline
uv run --locked --extra all music-tastes site
```

The first command imports and scores the archive; the full update publishes a
consistent snapshot transactionally. Without a local archive, the stage reuses
its fingerprinted private lexicon, or downloads the explicit v2.1 resource when
not offline. Lexicon version, import time, source URL, archive hash and normalized
file hash are recorded. Original v1 files and model scores are not replaced.

**Scale matters:** v1 is 0-1; v2.1 is -1 to +1. The comparison explicitly maps
v2.1 to `(score + 1) / 2`, for all three dimensions. Valence measures
positive-negative associations; arousal measures activation; dominance measures
control. Low valence alone is not sadness, and these dimensions are not combined
into an unvalidated sadness index.

All variants use the existing lowercased ASCII/apostrophe tokenizer without
stemming, lemmatization or synonyms. The larger dictionary does not make
unmatched vocabulary or historical semantic drift disappear.

- Compare token-frequency means and unique-word means to check repetition.
- Compare both versions restricted to their shared vocabulary, separating
  revised ratings from the effect of adding newly covered words.
- Match phrases longest-first without overlap, within newline, section and
  strong-punctuation boundaries. A matched phrase contributes one rated unit;
  its component words are not counted again. This is a declared matching
  sensitivity, not a complete context/negation analyzer. Dictionary terms not
  represented exactly by the tokenizer are not silently reinterpreted.
- Separately average only valence-polar words at fixed absolute native-score
  thresholds 0.167 and 0.333. The thresholds are not selected to maximize trends.
- Compare all variants on exactly the same songs with nonmissing scores;
  additionally report all eligible v2.1 songs and a minimum-ten-matched-token
  sensitivity. No matches means missing, never neutral.
- Publish weighted and unweighted annual series, per-year token coverage,
  completed-period results, chart-year and first-52-week comparisons, and
  length/coverage-adjusted associations with debut-year-clustered intervals.

Annual bootstrap intervals use 1,000 song resamples within year. They do not
correct serial/artist dependence, and the many rank tests remain exploratory.
Length/coverage adjustments are sensitivity analyses, not proof of confounding,
mediation or causality. VAD measures lexical associations, not validated
whole-song mood or listeners' emotions. The complete aggregate downloads are
`nrc_vad_series.csv`, `nrc_vad_coverage.csv` and `nrc_vad_robustness.json`.

### Dependence-aware VAD inference

The same stage also exports `nrc_vad_statistics.csv` and
`nrc_vad_statistics.json`. These supplement, rather than relabel, the earlier
independence-based rank-test p-values. They test an OLS **average linear change
per decade**, not the earlier Theil-Sen slope. Each annual mean receives equal
regression weight; song exposure weighting remains inside weighted annual means.
The partial 1958 start and incomplete cutoff year are excluded. Fixed-follow-up
series retain only their already-complete cohorts.

The primary family is the three v2.1 token dimensions on identical matched songs,
using exposure-weighted debut-year means. Newey-West HAC standard errors use
Bartlett weights, finite-sample covariance correction, and t inference with
`n_years - 2` degrees of freedom. Five annual lags are the main specification;
three and ten are reported without selecting the most favorable result.

A separate check resamples centered, detrended OLS residuals in circular blocks
of three, five and ten years, with 9,999 replicates and seed 20261028. Calendar
years remain fixed. Basic 95% intervals invert centered bootstrap slope errors;
two-sided null p-values compare their absolute size with the observed slope,
using the plus-one Monte Carlo correction. The smallest reported bootstrap
p-value is 0.0001. This assumes approximately stationary, weakly dependent
detrended residuals; it is not an independent replication or a song bootstrap.

Holm corrections are applied separately within every method/window across
the three primary dimensions, across three paired v2.1-minus-v1 annual
contrasts, and across the full set of scoring/time-base/weighting/sample and
paired comparisons. All results are downloadable, including non-significant
ones. An additional conservative Holm correction covers the entire battery,
including all methods and window lengths. The 95% intervals are pointwise, not
multiplicity-adjusted simultaneous intervals. These are **retrospective**, not
preregistered tests.

Neither method corrects arbitrary long-memory or artist dependence, repairs
lyric missingness or historical semantic drift, or establishes a causal effect.
Tests condition on observed annual means; annual song-bootstrap intervals
remain a separate measure of within-year sampling sensitivity. A linear
association can conceal nonlinear change. Failure to reject zero does not show
equivalence or no meaningful change, and no small-effect equivalence bound was
chosen after observing the data.

Implementation references:
[statsmodels robust covariance](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html),
[Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html),
and [circular block resampling](https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html).
No additional dependency is required.

## Chart exports

Charts follow the shared UK project's Substack style: warm `#F7F5F0` background,
serif type, muted gridlines, terracotta focus series, and source notes. Original
PNGs live in `reports/figures/`. A separate Margin of Error set lives in
`reports/figures/page_background/`, with matching filenames and the article's
`#F5F5F5` background.

Headline charts show only the exposure-weighted covered-song series, with one
95% song-bootstrap interval. The balanced covered-song subset remains in the
complete downloads and trend comparisons; omitting its overlapping line is a
presentation decision, not a change to the analysis. Share axes use percentages;
word ratings retain their original score units. Titles and filenames use
reader-facing names instead of internal metric identifiers. The stance-share
figure uses separate panels so its six overlapping themes stay readable.

The export changes only exactly matching, fully opaque background pixels.
Plotted colours, text, dimensions, transparency, and DPI metadata are preserved;
no series is rescored, rescaled, smoothed, or relabelled. The final report stage
regenerates every copy, including stance charts, before a full update can publish
its complete manifest. An unexpected source background fails that stage instead
of silently publishing incomplete exports.

```bash
uv run --locked --extra all music-tastes page-plots
uv run --locked --extra all music-tastes site
```

The first command re-exports existing public PNGs without private data or model
inference. The preview displays page-background charts and provides downloads of
both formats. If source figures or analyses are changed, use the full
`update --offline` command to republish a verified snapshot; a standalone render
does not advance the completed-analysis marker.

## Environment and commands

Use Python 3.12 (minimum supported: 3.11), not the macOS system Python:

```bash
uv sync --locked --python 3.12 --extra all --extra test
cp .env.example .env
uv run --locked --extra all --extra test python -m pytest tests/ -q
uv run --locked --extra all music-tastes status
uv run --locked --extra all music-tastes update
```

The Genius token is optional; without it, verified song-page URLs are tried.
The update command also tries the conservative LRCLIB fallback after Genius.
Genius-only stage runs remain available. No raw lyric text leaves local storage.
URL alternatives cover leading articles, dotted artist initials, and optional
quoted nicknames without
relaxing page-identity checks. LRCLIB candidates must agree on their normalized
word sequence; formatting differences alone do not make a match ambiguous.
Unanimous, text-free instrumental flags on exact LRCLIB matches are recorded
as provider-identified instrumentals, not as retrieved lyrics or failed searches.
Title/artist agreement alone does not establish that the charted recording
contains the supplied words. Source-backed recording corrections override
transcriptions for the exact recording identities documented in
`recording_instrumentals.csv` and `fetch_lyrics.py`. Evidence includes label
catalogs, contemporary reviews, artist catalogs, and explicitly identified
secondary single discographies; it is not universal listening verification.
Their evidence and limitations are listed in `DATA_SOURCES.md`. Rejected cached
bodies remain private. Null-body corrections do not increase lyric coverage,
and removing previously counted transcriptions can reduce it. An instrumental
classification can coexist with brief vocalizing or speech; it does not prove
every attached word is fabricated. Vocal covers and alternate versions are not
excluded merely because the composition also has an instrumental recording.

```bash
uv run --locked --extra all music-tastes ingest-charts --refresh
uv run --locked --extra all music-tastes fetch-lyrics --retry-failed --retry-after-days 7 --lrclib
uv run --locked --extra all music-tastes update --no-api --limit 20  # partial, never published
uv run --locked --extra all music-tastes update --offline         # cached inputs/models required
uv run --locked --extra all music-tastes ingest-charts --offline --as-of 2026-07-25
uv run --locked --extra all music-tastes site
python3 -m http.server 8000 --directory site
```

`--as-of` limits chart **issue dates**; it does not reconstruct the exact
knowledge available on a historical retrieval date. A chart refresh downloads
the full primary snapshot so upstream revisions are not silently missed.
Source files and derived tables remain private/untracked.

Features are cached using lyric/configuration fingerprints. Error records are
not completed inference. Limited runs preserve other valid cache entries.
`update --retry-after-days 0` explicitly retries fresh misses too; successful
lyric records are retained. The default failed-record retry age is seven days.
The update command stages public reports and advances the last-complete marker
only after required stages succeed. Incomplete runs retain a private manifest
and leave the previous report intact. Audio enrichment is a separately invoked,
optional historical stage; absence of audio is explicitly reported, not filled.

## Retrospective snapshot

There is no scheduled analysis refresh and no self-hosted runner requirement.
The commands above support deliberate local reproduction or a later, explicitly
requested update. They do not imply that published results automatically stay
current. The GitHub Actions test workflow checks code, not source freshness.

Keep private inputs and model/HTTP/lyric caches locally; do not commit them or
upload them as workflow artifacts. Public reports and the landing-page preview
describe only their completed snapshot. Neither website deployment nor
automatic commits are part of the analysis.

The lockfile is generated and tested on Python 3.12. The configured public TUNA
PyPI mirror provides the packages because the primary package-file host was
unreachable during restoration. Package hashes are recorded in `uv.lock`.
Neither a partial inference run nor an unreviewed packet is finished analysis.

## Landing page

`music-tastes site` generates a lightweight, relative-path static preview in
`site/` with complete available aggregate CSV downloads. Until a complete run
manifest matches the report fingerprints, it labels old outputs as an
**unverified historical snapshot**
and does not turn newly ingested charts into a claim of refreshed analysis.
Generated `site/index.html` and `site/assets/` are ignored; rebuild them from
the public reports. Source CSS and the builder remain version-controlled.
The main-site integration and deployment are intentionally deferred.
