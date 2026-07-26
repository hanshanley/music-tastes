# Are US hit songs getting sadder, and are fewer of them about love?

_Generated 2026-07-26 17:08 UTC._

## What this measures

Every song that entered the Billboard Hot 100 between 1958-08-04 and the
present. The Hot 100 combines sales, radio airplay and (since 2007-2013)
streaming, so it is the closest long-run proxy available for what Americans
actually listened to. It is a proxy, not a census: see Limitations.

## Coverage, and why it is reported first

- Overall lyric coverage: **77.4%** of charting songs
- Coverage ranges from 51.8% (1959) to 92.3% (2017)
- Spearman(year, coverage) = **+0.954** (p = 1.3e-36)

![coverage](figures/coverage_by_year.png)

Coverage **is** year-dependent, so every result below is reported twice:
once on all covered songs and once on a complete-case subset holding the
number of songs per year constant. Where the two disagree in direction,
the result is marked unresolved rather than reported as a finding.

## How much to trust the classifiers

**Random hand-labelled sample (30 songs, four per decade).** This is an unbiased estimate.

| Task | Method | Accuracy | Precision | Recall | Cohen kappa |
|---|---|---|---|---|---|
| Is it a relationship song? | B (NLI) | 0.97 | 0.94 | 1.00 | 0.93 |
| Is it a relationship song? | A (embeddings) | 0.67 | 0.63 | 0.80 | 0.33 |

Inter-method agreement is poor (Cohen kappa = 0.25). Agreement is therefore not treated as evidence in itself; the hand labels decide which method is right, and Method B is used for all reported stance results.

**Purposive independence set (13 famous songs with uncontroversial stances).** These were chosen for being clear-cut, so this is an **upper bound**, not an unbiased estimate. It exists because independence songs are rare -- the random sample of 32 contained exactly one, too few to estimate precision or recall for the class this project is about.

| Method | Accuracy | Precision | Recall |
|---|---|---|---|
| B (NLI) | 1.00 | 1.00 | 1.00 |
| A (embeddings) | 0.54 | 1.00 | 0.25 |

Method A's failure mode is systematic, not noisy: cosine similarity tracks *topic* (this is a breakup song) and cannot see *stance* (...and the narrator is fine about it), so it labels "I Will Survive", "Since U Been Gone" and "thank u, next" as heartbreak. This is exactly the distinction the research question turns on, which is why the entailment model carries the result.

## Results

### lyric_valence

Lyric valence (NRC VAD, 0=negative 1=positive)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.563, p = 8.1e-12)
- Change per decade: -0.0080
- First 5 years 0.629 -> last 5 years 0.591
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 110 | 64.6% | 62.7% – 66.9% |
| 1960s | 959 | 63.1% | 62.4% – 63.9% |
| 1970s | 808 | 63.4% | 62.8% – 64.1% |
| 1980s | 639 | 62.1% | 61.3% – 62.8% |
| 1990s | 628 | 62.8% | 62.1% – 63.5% |
| 2000s | 678 | 60.3% | 59.6% – 61.0% |
| 2010s | 950 | 59.7% | 59.0% – 60.4% |
| 2020s | 892 | 58.7% | 57.8% – 59.5% |

![lyric_valence](figures/lyric_valence_decade.png)

![lyric_valence yearly](figures/lyric_valence.png)

### lyric_sadness

Share of words with a sadness association (NRC EmoLex)  (higher = sadder)

- Direction: **rising, not significant** (Kendall tau = +0.012, p = 0.88)
- Change per decade: +0.0000
- First 5 years 0.022 -> last 5 years 0.021
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 108 | 1.7% | 1.1% – 2.4% |
| 1960s | 959 | 2.2% | 2.0% – 2.5% |
| 1970s | 807 | 1.9% | 1.7% – 2.1% |
| 1980s | 639 | 2.1% | 1.9% – 2.5% |
| 1990s | 628 | 1.6% | 1.4% – 1.8% |
| 2000s | 678 | 1.9% | 1.6% – 2.2% |
| 2010s | 949 | 1.9% | 1.7% – 2.2% |
| 2020s | 892 | 2.2% | 2.0% – 2.5% |

![lyric_sadness](figures/lyric_sadness_decade.png)

![lyric_sadness yearly](figures/lyric_sadness.png)

### lyric_joy

Share of words with a joy association (NRC EmoLex)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.616, p = 6.9e-14)
- Change per decade: -0.0041
- First 5 years 0.048 -> last 5 years 0.027
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 108 | 4.7% | 3.8% – 5.9% |
| 1960s | 959 | 5.3% | 5.0% – 5.7% |
| 1970s | 807 | 5.0% | 4.5% – 5.4% |
| 1980s | 639 | 4.0% | 3.7% – 4.4% |
| 1990s | 628 | 4.2% | 3.8% – 4.5% |
| 2000s | 678 | 3.0% | 2.7% – 3.4% |
| 2010s | 949 | 3.0% | 2.8% – 3.4% |
| 2020s | 892 | 2.5% | 2.2% – 2.8% |

![lyric_joy](figures/lyric_joy_decade.png)

![lyric_joy yearly](figures/lyric_joy.png)

### vader_valence

Lyric valence (VADER)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.653, p = 2.1e-15)
- Change per decade: -0.1139
- First 5 years 0.987 -> last 5 years 0.251
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 110 | 117.3% | 95.6% – 137.2% |
| 1960s | 957 | 90.5% | 82.5% – 98.5% |
| 1970s | 805 | 84.1% | 74.9% – 92.0% |
| 1980s | 639 | 73.5% | 64.5% – 82.3% |
| 1990s | 628 | 72.7% | 63.5% – 81.1% |
| 2000s | 678 | 47.0% | 38.6% – 56.0% |
| 2010s | 949 | 39.0% | 30.2% – 48.3% |
| 2020s | 892 | 23.8% | 14.5% – 34.7% |

![vader_valence](figures/vader_valence_decade.png)

![vader_valence yearly](figures/vader_valence.png)

### relationship_share

Share of hits that are about a relationship  (higher = more love songs)

- Direction: **rising, not significant** (Kendall tau = +0.141, p = 0.087)
- Change per decade: +0.0063
- First 5 years 0.703 -> last 5 years 0.765
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 158 | 69.6% | 57.7% – 80.6% |
| 1960s | 1279 | 67.5% | 63.7% – 71.1% |
| 1970s | 1136 | 64.6% | 60.9% – 68.5% |
| 1980s | 962 | 76.1% | 72.6% – 79.2% |
| 1990s | 882 | 73.5% | 69.1% – 77.1% |
| 2000s | 936 | 69.8% | 65.8% – 73.8% |
| 2010s | 1301 | 70.5% | 66.1% – 74.4% |
| 2020s | 1183 | 73.3% | 68.1% – 78.7% |

![relationship_share](figures/relationship_share_decade.png)

![relationship_share yearly](figures/relationship_share.png)

### independence_share

Share of relationship songs taking an 'I don't need you' stance  (higher = more independence)

- Direction: **rising, significant** (Kendall tau = +0.508, p = 6.7e-10)
- Change per decade: +0.0218
- First 5 years 0.026 -> last 5 years 0.139
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 114 | 2.2% | 0.0% – 7.2% |
| 1960s | 875 | 4.3% | 2.7% – 6.1% |
| 1970s | 690 | 4.7% | 2.8% – 6.8% |
| 1980s | 705 | 6.5% | 4.2% – 8.9% |
| 1990s | 599 | 8.0% | 5.3% – 10.9% |
| 2000s | 593 | 14.4% | 10.6% – 18.0% |
| 2010s | 781 | 19.3% | 15.2% – 24.0% |
| 2020s | 667 | 15.1% | 10.5% – 20.2% |

![independence_share](figures/independence_share_decade.png)

![independence_share yearly](figures/independence_share.png)

### heartbreak_share

Share of relationship songs about heartbreak/wanting an ex back  (higher = more heartbreak)

- Direction: **rising, not significant** (Kendall tau = +0.098, p = 0.23)
- Change per decade: +0.0061
- First 5 years 0.312 -> last 5 years 0.408
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 114 | 24.1% | 14.8% – 34.6% |
| 1960s | 875 | 36.6% | 32.5% – 40.8% |
| 1970s | 690 | 28.5% | 24.0% – 32.9% |
| 1980s | 705 | 34.9% | 30.3% – 39.3% |
| 1990s | 599 | 37.3% | 31.6% – 43.0% |
| 2000s | 593 | 38.0% | 32.5% – 43.4% |
| 2010s | 781 | 29.5% | 24.3% – 35.0% |
| 2020s | 667 | 41.2% | 33.7% – 48.8% |

![heartbreak_share](figures/heartbreak_share_decade.png)

![heartbreak_share yearly](figures/heartbreak_share.png)

### devotion_share

Share of relationship songs about devotion/commitment  (higher = more devotion)

- Direction: **falling, significant** (Kendall tau = -0.213, p = 0.0096)
- Change per decade: -0.0123
- First 5 years 0.202 -> last 5 years 0.124
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 114 | 26.1% | 15.3% – 36.5% |
| 1960s | 875 | 21.4% | 18.3% – 25.0% |
| 1970s | 690 | 15.6% | 12.2% – 19.1% |
| 1980s | 705 | 20.9% | 17.4% – 24.5% |
| 1990s | 599 | 27.8% | 23.0% – 32.9% |
| 2000s | 593 | 17.2% | 13.4% – 21.5% |
| 2010s | 781 | 14.3% | 10.1% – 18.5% |
| 2020s | 667 | 11.7% | 6.7% – 17.1% |

![devotion_share](figures/devotion_share_decade.png)

![devotion_share yearly](figures/devotion_share.png)

### bpm

Tempo in beats per minute (Essentia via AcousticBrainz)  (higher = faster)

- Direction: **falling, not significant** (Kendall tau = -0.115, p = 0.17)
- Change per decade: -0.5305
- First 5 years 116.694 -> last 5 years 119.539
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 46 | 11724.4% | 10929.6% – 12521.9% |
| 1960s | 225 | 12170.8% | 11872.0% – 12464.6% |
| 1970s | 284 | 12217.9% | 11944.3% – 12485.0% |
| 1980s | 287 | 12105.6% | 11850.5% – 12333.6% |
| 1990s | 311 | 11633.6% | 11373.6% – 11911.6% |
| 2000s | 314 | 11813.2% | 11520.3% – 12105.5% |
| 2010s | 303 | 12051.7% | 11775.9% – 12332.4% |
| 2020s | 74 | 12331.1% | 11673.9% – 12988.2% |

![bpm](figures/bpm_decade.png)

![bpm yearly](figures/bpm.png)

### minor_key_share

Share of songs in a minor key (Essentia)  (higher = more minor-key)

- Direction: **rising, significant** (Kendall tau = +0.394, p = 3.4e-06)
- Change per decade: +0.0339
- First 5 years 0.117 -> last 5 years 0.319
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 46 | 11.4% | 3.6% – 21.1% |
| 1960s | 225 | 19.5% | 14.3% – 24.8% |
| 1970s | 284 | 22.4% | 17.5% – 27.3% |
| 1980s | 287 | 23.6% | 18.6% – 28.6% |
| 1990s | 311 | 33.0% | 27.6% – 38.6% |
| 2000s | 314 | 41.1% | 35.5% – 47.0% |
| 2010s | 302 | 28.9% | 23.8% – 33.9% |
| 2020s | 74 | 35.0% | 23.8% – 46.7% |

![minor_key_share](figures/minor_key_share_decade.png)

![minor_key_share yearly](figures/minor_key_share.png)

### acoustic_mood_happy

Essentia 'happy' mood probability  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.613, p = 5.1e-13)
- Change per decade: -0.0443
- First 5 years 0.643 -> last 5 years 0.344
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 46 | 62.3% | 53.4% – 71.2% |
| 1960s | 225 | 63.2% | 59.1% – 67.2% |
| 1970s | 283 | 49.9% | 46.9% – 52.9% |
| 1980s | 286 | 50.3% | 47.1% – 53.5% |
| 1990s | 311 | 43.0% | 40.0% – 46.1% |
| 2000s | 312 | 41.4% | 38.5% – 44.8% |
| 2010s | 301 | 37.6% | 34.7% – 40.5% |
| 2020s | 74 | 35.4% | 29.7% – 41.5% |

![acoustic_mood_happy](figures/acoustic_mood_happy_decade.png)

![acoustic_mood_happy yearly](figures/acoustic_mood_happy.png)

### acoustic_mood_sad

Essentia 'sad' mood probability  (higher = sadder)

- Direction: **falling, significant** (Kendall tau = -0.328, p = 0.00011)
- Change per decade: -0.0183
- First 5 years 0.520 -> last 5 years 0.408
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 46 | 52.0% | 44.1% – 60.6% |
| 1960s | 225 | 44.3% | 41.2% – 47.6% |
| 1970s | 283 | 43.4% | 40.8% – 46.0% |
| 1980s | 286 | 35.9% | 33.8% – 38.3% |
| 1990s | 311 | 38.7% | 36.5% – 41.2% |
| 2000s | 312 | 32.5% | 30.3% – 35.0% |
| 2010s | 301 | 35.5% | 33.0% – 37.9% |
| 2020s | 74 | 41.9% | 36.5% – 47.3% |

![acoustic_mood_sad](figures/acoustic_mood_sad_decade.png)

![acoustic_mood_sad yearly](figures/acoustic_mood_sad.png)

### lyric_length

Words per song (drives a third of the valence trend)  (higher = wordier)

- Direction: **rising, significant** (Kendall tau = +0.744, p = 1.5e-19)
- Change per decade: +52.2368
- First 5 years 134.458 -> last 5 years 366.481
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 925 | 13047.9% | 12014.3% – 14115.8% |
| 1960s | 6849 | 15469.5% | 15039.1% – 15863.0% |
| 1970s | 5296 | 21090.8% | 20531.7% – 21664.7% |
| 1980s | 4113 | 25580.5% | 24997.4% – 26170.4% |
| 1990s | 3422 | 34974.8% | 33857.4% – 36097.2% |
| 2000s | 3418 | 42511.4% | 41363.1% – 43745.2% |
| 2010s | 4430 | 41404.9% | 40250.2% – 42464.5% |
| 2020s | 4149 | 38149.4% | 36962.1% – 39270.1% |

![lyric_length](figures/lyric_length_decade.png)

![lyric_length yearly](figures/lyric_length.png)

## Rival explanations, tested

### The sentiment result does not survive a change of method

This is the most important check in the project, and it goes against the headline. On an **identical set of 690 songs** (10 per year), two measures of the same construct disagree:

| Measure | Sees negation/context? | rho(year, valence) | p |
|---|---|---|---|
| NRC VAD word norms | no | -0.221 | 4.6e-09 |
| Entailment model | yes | -0.012 | 0.76 |

The entailment measure is not broken: its extremes are exactly right (highest — *Celebration*, *A Holly Jolly Christmas*, *Best Day Of My Life*; lowest — *Crying*, *Broken-Hearted Melody*, *Breakeven*). It discriminates happy from sad songs cleanly; it just finds no trend over time. Because both measures ran on the same songs, sampling cannot explain the gap.

**Most plausible reading:** the lexicon decline reflects *vocabulary* change rather than *emotional* change — modern lyrics use words the NRC norms score lower (slang, profanity, concrete nouns) without the songs being sadder in any sense a listener would recognise.

**The word-average valence and joy trends below should therefore be read as not robust to measurement method.** The stance results (relationship share, independence share) come from the entailment model and are unaffected by this.

### Non-English songs — ruled out

All lexicons are English-only, and the non-English share of charting songs rises from about 1% before 2010 to 7.2% in the 2020s. Restricting to confidently-English, effectively monolingual songs moves the coefficient only from -0.303 to -0.281, so language does not drive the trend.

### Lyric length — a mediator, not an artefact (correction)

An earlier version of this report claimed *"roughly a third of the effect is length, not mood"*, based on a partial correlation controlling for word count. **That was wrong.** Length is a mediator (year → songs get wordier → word-average valence falls), and controlling for a variable on the causal path subtracts real signal.

Two tests settle it:

- Within every length quintile the decline persists (rho -0.30 to -0.14, all significant), including the shortest songs.

- Computing valence over unique word *types*, which removes repetition entirely, makes the decline **stronger** (rho -0.346 vs -0.302), so it is not old songs repeating happy hooks.

So within the lexicon's own terms the decline is real. That is a separate question from whether the lexicon measures mood, which the contextual check above puts in doubt.

### Measurement regime

Restricted to 1991 onward (SoundScan era only, one consistent chart methodology), the lexicon valence decline survives: tau = -0.537, p = 4.1e-06.

### The independence rise is a step, not a slope

Within the post-1991 era alone it is not significant (tau = +0.241, p = 0.038), consistent with the decade table: a jump around 2000 followed by a plateau, rather than a continuing climb.

### Genre mix — tested, and not the driver

Rap and R&B went from absent to dominant on the Hot 100, and they have different lyrical conventions, so a change in the *mix* could move the average without any genre changing. Re-estimating the year effect with genre fixed effects:

| Metric | Unadjusted / decade | Genre-adjusted | Attenuation | n |
|---|---|---|---|---|
| lyric valence | -0.00640 | -0.00567 (p=0.0063) | 12% | 406 |
| independence share | +0.02424 | +0.01865 (p=0.069) | 23% | 359 |

Genre mix accounts for only about a tenth of the lexicon valence trend, so it is not the explanation. The independence trend keeps roughly three quarters of its magnitude under genre control but loses significance — note this runs on the 359 songs carrying both a genre label and a stance label, against 3,510 for the headline estimate, so this is a power limitation rather than evidence of absence.

**Caveat on the labels themselves.** These are Essentia's automatic classifiers, not editorial metadata. Its `genre_dortmund` model was discarded outright: it labels 95–98% of everything after 1980 "electronic", which is not a credible description of the Hot 100. `genre_rosamerica` is used instead — balanced across seven classes with a plausible trajectory (hip-hop 0% in the 1950s to 22.8% in the 1990s) — but it is still wrong often enough that this is a sanity check, not a clean genre control.

## Limitations

1. **The chart is not listening.** Hot 100 methodology changed in 1991
   (SoundScan/BDS), 1998, 2005 (digital), 2007 and 2013 (streaming/video).
   Comparisons spanning those dates cross measurement regimes.
2. **Chart tenure has inflated.** Songs now stay on the chart 90+ weeks versus
   ~13 in the 1960s, so exposure weights are normalized within year.
3. **Lyric coverage is uneven** and correlates with year; see above.
4. **Lexicon sentiment is blind to stance and context.** Word-level valence
   cannot tell 'I don't need you' from 'I need you', cannot see negation, and
   cannot track 68 years of semantic change. This is not hypothetical here: a
   context-aware model run on the same songs finds no valence trend at all.
   Stance questions are therefore answered by the entailment model instead.
5. **Genre labels are model-inferred, not editorial.** Essentia's classifiers
   are noisy; `genre_dortmund` was discarded as degenerate. Genre control is a
   sanity check rather than a clean adjustment.
6. **Acoustic coverage is partial.** AcousticBrainz is community-submitted, so
   BPM and mood exist for a subset of songs, and Essentia BPM is prone to
   octave errors.

## Sources

See `DATA_SOURCES.md`.
