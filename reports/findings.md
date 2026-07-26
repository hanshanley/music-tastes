# Are US hit songs getting sadder, and are fewer of them about love?

_Generated 2026-07-26 14:00 UTC._

## What this measures

Every song that entered the Billboard Hot 100 between 1958-08-04 and the
present. The Hot 100 combines sales, radio airplay and (since 2007-2013)
streaming, so it is the closest long-run proxy available for what Americans
actually listened to. It is a proxy, not a census: see Limitations.

## Coverage, and why it is reported first

- Overall lyric coverage: **22.9%** of charting songs
- Coverage ranges from 12.7% (1959) to 31.6% (2026)
- Spearman(year, coverage) = **+0.902** (p = 4.3e-26)

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
| 1950s | 110 | 64.6% | 62.8% – 66.7% |
| 1960s | 959 | 63.1% | 62.4% – 63.8% |
| 1970s | 808 | 63.4% | 62.8% – 64.1% |
| 1980s | 639 | 62.1% | 61.4% – 62.8% |
| 1990s | 628 | 62.8% | 62.1% – 63.5% |
| 2000s | 678 | 60.3% | 59.6% – 61.0% |
| 2010s | 950 | 59.7% | 59.0% – 60.4% |
| 2020s | 892 | 58.7% | 57.9% – 59.4% |

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
| 2010s | 949 | 1.9% | 1.7% – 2.1% |
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
| 1950s | 108 | 4.7% | 3.7% – 5.9% |
| 1960s | 959 | 5.3% | 5.0% – 5.7% |
| 1970s | 807 | 5.0% | 4.6% – 5.5% |
| 1980s | 639 | 4.0% | 3.7% – 4.4% |
| 1990s | 628 | 4.2% | 3.8% – 4.6% |
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
| 1950s | 110 | 117.3% | 96.2% – 139.5% |
| 1960s | 957 | 90.5% | 82.7% – 97.9% |
| 1970s | 805 | 84.1% | 75.4% – 92.2% |
| 1980s | 639 | 73.5% | 64.6% – 82.6% |
| 1990s | 628 | 72.7% | 63.3% – 81.3% |
| 2000s | 678 | 47.0% | 39.4% – 55.7% |
| 2010s | 949 | 39.0% | 30.1% – 48.2% |
| 2020s | 892 | 23.8% | 13.3% – 34.2% |

![vader_valence](figures/vader_valence_decade.png)

![vader_valence yearly](figures/vader_valence.png)

### relationship_share

Share of hits that are about a relationship  (higher = more love songs)

- Direction: **rising, not significant** (Kendall tau = +0.027, p = 0.74)
- Change per decade: +0.0027
- First 5 years 0.689 -> last 5 years 0.705
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 70 | 62.3% | 45.8% – 81.9% |
| 1960s | 590 | 68.9% | 63.0% – 74.8% |
| 1970s | 469 | 61.9% | 56.4% – 67.8% |
| 1980s | 395 | 74.3% | 68.2% – 79.5% |
| 1990s | 392 | 76.5% | 71.0% – 81.4% |
| 2000s | 426 | 70.2% | 63.7% – 76.1% |
| 2010s | 623 | 69.2% | 63.2% – 75.5% |
| 2020s | 545 | 67.6% | 58.5% – 76.8% |

![relationship_share](figures/relationship_share_decade.png)

![relationship_share yearly](figures/relationship_share.png)

### independence_share

Share of relationship songs taking an 'I don't need you' stance  (higher = more independence)

- Direction: **rising, significant** (Kendall tau = +0.403, p = 1e-06)
- Change per decade: +0.0211
- First 5 years 0.020 -> last 5 years 0.125
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 50 | 0.0% | 0.0% – 0.0% |
| 1960s | 415 | 4.9% | 2.3% – 7.8% |
| 1970s | 268 | 3.7% | 1.5% – 6.7% |
| 1980s | 287 | 8.7% | 4.7% – 13.2% |
| 1990s | 269 | 7.9% | 4.2% – 13.1% |
| 2000s | 273 | 17.6% | 11.0% – 25.0% |
| 2010s | 373 | 17.8% | 12.2% – 24.0% |
| 2020s | 293 | 16.6% | 9.5% – 25.1% |

![independence_share](figures/independence_share_decade.png)

![independence_share yearly](figures/independence_share.png)

### heartbreak_share

Share of relationship songs about heartbreak/wanting an ex back  (higher = more heartbreak)

- Direction: **rising, not significant** (Kendall tau = +0.052, p = 0.53)
- Change per decade: +0.0057
- First 5 years 0.323 -> last 5 years 0.377
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 50 | 19.9% | 7.7% – 35.3% |
| 1960s | 415 | 37.6% | 31.6% – 43.6% |
| 1970s | 268 | 27.0% | 20.3% – 34.2% |
| 1980s | 287 | 35.4% | 28.7% – 42.4% |
| 1990s | 269 | 35.9% | 27.5% – 44.3% |
| 2000s | 273 | 35.0% | 26.6% – 43.2% |
| 2010s | 373 | 27.5% | 19.5% – 35.1% |
| 2020s | 293 | 39.3% | 28.3% – 51.9% |

![heartbreak_share](figures/heartbreak_share_decade.png)

![heartbreak_share yearly](figures/heartbreak_share.png)

### devotion_share

Share of relationship songs about devotion/commitment  (higher = more devotion)

- Direction: **falling, not significant** (Kendall tau = -0.105, p = 0.2)
- Change per decade: -0.0092
- First 5 years 0.165 -> last 5 years 0.158
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 50 | 23.0% | 10.2% – 39.0% |
| 1960s | 415 | 22.2% | 17.2% – 27.5% |
| 1970s | 268 | 10.1% | 6.1% – 14.7% |
| 1980s | 287 | 16.8% | 11.3% – 22.8% |
| 1990s | 269 | 32.7% | 25.1% – 40.9% |
| 2000s | 273 | 15.8% | 10.7% – 21.8% |
| 2010s | 373 | 15.3% | 8.2% – 22.2% |
| 2020s | 293 | 14.2% | 5.5% – 24.2% |

![devotion_share](figures/devotion_share_decade.png)

![devotion_share yearly](figures/devotion_share.png)

## Limitations

1. **The chart is not listening.** Hot 100 methodology changed in 1991
   (SoundScan/BDS), 1998, 2005 (digital), 2007 and 2013 (streaming/video).
   Comparisons spanning those dates cross measurement regimes.
2. **Chart tenure has inflated.** Songs now stay on the chart 90+ weeks versus
   ~13 in the 1960s, so exposure weights are normalized within year.
3. **Lyric coverage is uneven** and correlates with year; see above.
4. **Lexicon sentiment is blind to stance.** Word-level valence cannot tell
   'I don't need you' from 'I need you', which is why the stance question is
   answered by an entailment model instead.
5. **Genre mix is uncontrolled** in this version. A shift toward genres with
   different lyrical conventions is a live rival explanation for any change
   in valence.

## Sources

See `DATA_SOURCES.md`.
