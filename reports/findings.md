# Are US hit songs getting sadder, and are fewer of them about love?

_Generated 2026-07-26 22:47 UTC._

## Summary

| Question | Answer | Confidence |
|---|---|---|
| Are fewer hits about love/relationships? | **No.** Exposure-weighted share is flat at 65–76% across seven decades. | Good — but the unweighted series declines and fails the coverage check, so the two views differ. |
| Among relationship songs, are more about *not needing* one? | **Yes** — the *direction* is the strongest finding here, ~+1.4 points/decade after correcting for aggregation bias. But the *level* is not quotable: it ranges 0.8%–14.8% purely on how the question is worded. | Direction: strong (survives coverage, genre, era, length, and 4 of 5 paraphrases). Level: unreliable. |
| Are the lyrics getting sadder? | **Modestly, yes** — about 0.08–0.10 SD per decade. A word-norm lexicon and a context-aware model agree once their *opposite* lyric-length biases are removed. | Moderate — the raw lexicon series overstates it. |
| Is the music getting sadder? | **No usable evidence.** Essentia's happy *and* sad scores both fall, which indicates classifier drift. Minor-key share doubles but ~60% is genre mix and it vanishes post-1991. | Weak. |
| Are songs getting faster or slower? | **No change.** Tempo is flat (tau -0.06, p=0.47). | Good. |

The short version: **what songs are *about* changed more than how they *feel*.** Love songs are as common as ever, but the stance inside them shifted markedly toward self-sufficiency. Lyrics did get somewhat less positive, though far less than a naive word-count reading suggests, and the *musical* sadness signals (tempo, mood classifiers) show nothing usable at all.

## What this measures

Every song that entered the Billboard Hot 100 between 1958-08-04 and the
present. The Hot 100 combines sales, radio airplay and (since 2007-2013)
streaming, so it is the closest long-run proxy available for what Americans
actually listened to. It is a proxy, not a census: see Limitations.

## Coverage, and why it is reported first

- Overall lyric coverage: **77.4%** of charting songs, and **83.5% of total chart exposure** — the misses are disproportionately low-exposure deep cuts (median peak position #66), so weighted results are better covered than the song count suggests
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

### Check the classifier yourself

Aggregate accuracy figures are easy to publish and hard to trust. These are the songs the model is most confident about, so a reader who knows the music can judge directly.

| Decade | Highest-confidence "I don't need you" songs |
|---|---|
| 1950s | Makin' Love — Floyd Robinson (0.85); Down In Virginia — Jimmy Reed (0.57); Mr. Blue — The Fleetwoods (0.55); You Got What It Takes — Marv Johnson (0.43) |
| 1960s | Red Rubber Ball — The Cyrkle (1.00); Drive My Car — Bob Kuban And The In-Men (0.99); Bye Bye Baby — Mary Wells (0.99); Ain't Doing Too Bad (Part 1) — Bobby Bland (0.99) |
| 1970s | As Long As He Takes Care Of Home — Candi Staton (0.98); I Will Survive — Gloria Gaynor (0.97); Down To Love Town — The Originals (0.97); Beast Of Burden — The Rolling Stones (0.96) |
| 1980s | The Glamorous Life — Sheila E. (0.98); Dreamin' — John Schneider (0.97); Free Fallin' — Tom Petty (0.96); Another Night — Aretha Franklin (0.96) |
| 1990s | Believe — Cher (1.00); Don't Want To Be A Fool — Luther Vandross (0.99); Don't Turn Around — Ace Of Base (0.98); Diggin' On You — TLC (0.97) |
| 2000s | Me, Myself And I — Beyonce (0.99); Don't Bother — Shakira (0.99); U + Ur Hand — P!nk (0.99); Another Dumb Blonde — Hoku (0.98) |
| 2010s | Don't Call Me Up — Mabel (1.00); Buy My Own Drinks — Runaway June (0.99); Ayo — Chris Brown & Tyga (0.99); Demons And Angels — A Boogie Wit da Hoodie Featuring Juice WRLD (0.99) |
| 2020s | Closure — Taylor Swift (1.00); Do It Again — NLE Choppa & 2Rare (0.99); Cairo — Karol G & Ovy On The Drums (0.99); Captain Hook — Megan Thee Stallion (0.99) |

**Why the two-stage design matters.** The stance model keys on the literal claim, not the romantic context, so on its own it fires on *Another Brick In The Wall* ("we don't need no education"), on J. Cole's *Brackets* (about tax), and on *The Little Drummer Boy*. The relationship gate removes these before the statistic is computed — these songs score high on the stance but near zero on being about a relationship:

| Song | stance | is-relationship |
|---|---|---|
| Wake Me Up! — Avicii | 0.96 | 0.01 |
| Old Town Road — Lil Nas X Featuring Billy Ray Cyrus | 0.91 | 0.07 |
| Royals — Lorde | 0.93 | 0.02 |
| The Bones — Maren Morris | 0.92 | 0.04 |
| Family Affair — Mary J. Blige | 0.95 | 0.00 |
| TiK ToK — Ke$ha | 0.94 | 0.00 |

The separation is clean (stance >0.9 against relationship <0.08), which is why the headline share is computed *within* relationship songs rather than over the whole chart.

_13 metrics are tracked, each in four variants, plus a battery of confound tests — well over a hundred hypothesis tests in total. At p<0.05 several 'significant' results are expected by chance alone. The findings leaned on here clear that bar comfortably (the independence trend is p=1.8e-09 after adjustment); isolated marginal results, such as tempo rising within the post-1991 window at p=0.025, are not treated as findings._

### lyric_valence

Lyric valence (NRC VAD, 0=negative 1=positive)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.563, p = 8.1e-12)
- Change per decade: -0.0080
- First 5 years 0.629 -> last 5 years 0.591
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 110 | 64.6% | 62.8% – 66.9% |
| 1960s | 959 | 63.1% | 62.5% – 63.8% |
| 1970s | 808 | 63.4% | 62.8% – 64.1% |
| 1980s | 639 | 62.1% | 61.3% – 62.8% |
| 1990s | 628 | 62.8% | 62.1% – 63.4% |
| 2000s | 678 | 60.3% | 59.6% – 60.9% |
| 2010s | 950 | 59.7% | 59.0% – 60.4% |
| 2020s | 892 | 58.7% | 57.9% – 59.5% |

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
| 1980s | 639 | 2.1% | 1.9% – 2.4% |
| 1990s | 628 | 1.6% | 1.4% – 1.8% |
| 2000s | 678 | 1.9% | 1.6% – 2.2% |
| 2010s | 949 | 1.9% | 1.7% – 2.2% |
| 2020s | 892 | 2.2% | 1.9% – 2.5% |

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
| 1970s | 807 | 5.0% | 4.5% – 5.4% |
| 1980s | 639 | 4.0% | 3.7% – 4.4% |
| 1990s | 628 | 4.2% | 3.8% – 4.6% |
| 2000s | 678 | 3.0% | 2.7% – 3.4% |
| 2010s | 949 | 3.0% | 2.7% – 3.3% |
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
| 1950s | 110 | 117.3% | 96.1% – 138.2% |
| 1960s | 957 | 90.5% | 83.1% – 98.2% |
| 1970s | 805 | 84.1% | 75.2% – 92.4% |
| 1980s | 639 | 73.5% | 64.4% – 82.8% |
| 1990s | 628 | 72.7% | 63.9% – 81.4% |
| 2000s | 678 | 47.0% | 38.6% – 55.6% |
| 2010s | 949 | 39.0% | 29.8% – 48.3% |
| 2020s | 892 | 23.8% | 13.8% – 34.5% |

![vader_valence](figures/vader_valence_decade.png)

![vader_valence yearly](figures/vader_valence.png)

### relationship_share

Share of hits that are about a relationship  (higher = more love songs)

- Direction: **rising, not significant** (Kendall tau = +0.084, p = 0.31)
- Change per decade: +0.0044
- First 5 years 0.721 -> last 5 years 0.753
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 212 | 72.1% | 63.5% – 80.0% |
| 1960s | 1537 | 68.6% | 65.5% – 71.6% |
| 1970s | 1396 | 65.4% | 62.2% – 68.4% |
| 1980s | 1233 | 76.6% | 73.8% – 79.2% |
| 1990s | 1123 | 74.1% | 70.6% – 77.0% |
| 2000s | 1180 | 69.1% | 65.7% – 72.3% |
| 2010s | 1545 | 70.0% | 66.5% – 73.6% |
| 2020s | 1346 | 73.7% | 69.6% – 78.0% |

![relationship_share](figures/relationship_share_decade.png)

![relationship_share yearly](figures/relationship_share.png)

### independence_share

Share of relationship songs taking an 'I don't need you' stance  (higher = more independence)

- Direction: **rising, significant** (Kendall tau = +0.591, p = 7e-13)
- Change per decade: +0.0256
- First 5 years 0.030 -> last 5 years 0.159
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 155 | 2.2% | 0.0% – 6.0% |
| 1960s | 1059 | 4.8% | 3.3% – 6.4% |
| 1970s | 862 | 5.0% | 3.4% – 6.9% |
| 1980s | 916 | 7.5% | 5.7% – 9.6% |
| 1990s | 780 | 9.2% | 6.8% – 11.5% |
| 2000s | 760 | 16.0% | 12.6% – 19.3% |
| 2010s | 949 | 19.9% | 16.3% – 23.5% |
| 2020s | 786 | 17.5% | 13.4% – 21.7% |

![independence_share](figures/independence_share_decade.png)

![independence_share yearly](figures/independence_share.png)

### heartbreak_share

Share of relationship songs about heartbreak/wanting an ex back  (higher = more heartbreak)

- Direction: **rising, not significant** (Kendall tau = +0.040, p = 0.63)
- Change per decade: +0.0019
- First 5 years 0.306 -> last 5 years 0.361
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 155 | 27.5% | 19.1% – 36.0% |
| 1960s | 1059 | 33.2% | 29.6% – 36.7% |
| 1970s | 862 | 30.7% | 27.2% – 34.3% |
| 1980s | 916 | 35.1% | 31.5% – 38.9% |
| 1990s | 780 | 35.3% | 31.2% – 39.4% |
| 2000s | 760 | 34.6% | 30.2% – 39.2% |
| 2010s | 949 | 29.4% | 25.3% – 33.9% |
| 2020s | 786 | 36.9% | 31.3% – 42.8% |

![heartbreak_share](figures/heartbreak_share_decade.png)

![heartbreak_share yearly](figures/heartbreak_share.png)

### devotion_share

Share of relationship songs about devotion/commitment  (higher = more devotion)

- Direction: **falling, significant** (Kendall tau = -0.278, p = 0.00073)
- Change per decade: -0.0153
- First 5 years 0.212 -> last 5 years 0.108
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 155 | 25.5% | 17.8% – 33.4% |
| 1960s | 1059 | 22.4% | 19.3% – 25.5% |
| 1970s | 862 | 17.6% | 14.6% – 20.8% |
| 1980s | 916 | 21.5% | 18.4% – 24.8% |
| 1990s | 780 | 26.7% | 22.9% – 30.6% |
| 2000s | 760 | 18.6% | 15.3% – 22.3% |
| 2010s | 949 | 15.2% | 11.6% – 18.6% |
| 2020s | 786 | 11.0% | 7.0% – 15.1% |

![devotion_share](figures/devotion_share_decade.png)

![devotion_share yearly](figures/devotion_share.png)

### bpm

Tempo in beats per minute (Essentia via AcousticBrainz)  (higher = faster)

- Direction: **falling, not significant** (Kendall tau = -0.061, p = 0.47)
- Change per decade: -0.1842
- First 5 years 117.512 -> last 5 years 122.313
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 216 | 11698.5% | 11322.2% – 12111.7% |
| 1960s | 1030 | 12075.3% | 11930.1% – 12221.8% |
| 1970s | 1116 | 12313.4% | 12182.3% – 12459.2% |
| 1980s | 1141 | 12095.8% | 11979.3% – 12220.4% |
| 1990s | 1176 | 11540.0% | 11393.7% – 11698.5% |
| 2000s | 1230 | 11792.2% | 11633.6% – 11953.6% |
| 2010s | 1152 | 12142.7% | 11978.9% – 12313.4% |
| 2020s | 279 | 12143.1% | 11735.7% – 12561.4% |

![bpm](figures/bpm_decade.png)

![bpm yearly](figures/bpm.png)

### minor_key_share

Share of songs in a minor key (Essentia)  (higher = more minor-key)

- Direction: **rising, significant** (Kendall tau = +0.488, p = 6.9e-09)
- Change per decade: +0.0286
- First 5 years 0.129 -> last 5 years 0.393
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 216 | 13.4% | 8.7% – 18.1% |
| 1960s | 1030 | 19.6% | 16.9% – 22.0% |
| 1970s | 1116 | 21.6% | 19.0% – 24.0% |
| 1980s | 1141 | 23.6% | 21.1% – 26.3% |
| 1990s | 1173 | 33.0% | 29.7% – 35.9% |
| 2000s | 1230 | 35.4% | 32.1% – 38.6% |
| 2010s | 1147 | 27.0% | 24.0% – 30.5% |
| 2020s | 279 | 33.5% | 26.7% – 41.0% |

![minor_key_share](figures/minor_key_share_decade.png)

![minor_key_share yearly](figures/minor_key_share.png)

### acoustic_mood_happy

Essentia 'happy' mood probability  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.703, p = 6.8e-17)
- Change per decade: -0.0438
- First 5 years 0.636 -> last 5 years 0.330
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 216 | 62.6% | 58.4% – 66.7% |
| 1960s | 1027 | 62.4% | 60.4% – 64.3% |
| 1970s | 1112 | 51.2% | 49.5% – 52.9% |
| 1980s | 1137 | 51.3% | 49.7% – 52.9% |
| 1990s | 1172 | 42.3% | 40.7% – 43.9% |
| 2000s | 1224 | 43.4% | 41.7% – 45.2% |
| 2010s | 1141 | 40.1% | 38.2% – 41.9% |
| 2020s | 279 | 35.4% | 31.9% – 39.2% |

![acoustic_mood_happy](figures/acoustic_mood_happy_decade.png)

![acoustic_mood_happy yearly](figures/acoustic_mood_happy.png)

### acoustic_mood_sad

Essentia 'sad' mood probability  (higher = sadder)

- Direction: **falling, significant** (Kendall tau = -0.297, p = 0.00042)
- Change per decade: -0.0156
- First 5 years 0.514 -> last 5 years 0.427
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 216 | 50.8% | 46.8% – 54.5% |
| 1960s | 1027 | 43.9% | 42.4% – 45.6% |
| 1970s | 1112 | 42.7% | 41.4% – 44.1% |
| 1980s | 1137 | 34.1% | 32.9% – 35.2% |
| 1990s | 1172 | 37.3% | 36.0% – 38.5% |
| 2000s | 1224 | 33.9% | 32.6% – 35.1% |
| 2010s | 1141 | 35.1% | 33.6% – 36.5% |
| 2020s | 279 | 40.8% | 37.7% – 44.4% |

![acoustic_mood_sad](figures/acoustic_mood_sad_decade.png)

![acoustic_mood_sad yearly](figures/acoustic_mood_sad.png)

### lyric_length

Words per song  (higher = wordier)

- Direction: **rising, significant** (Kendall tau = +0.744, p = 1.5e-19)
- Change per decade: +52.2368
- First 5 years 134.458 -> last 5 years 366.786
- Survives complete-case check: yes

| Decade | n | Exposure-weighted mean | 95% CI |
|---|---|---|---|
| 1950s | 925 | 13047.9% | 12004.2% – 14146.0% |
| 1960s | 6849 | 15476.0% | 15093.6% – 15861.6% |
| 1970s | 5296 | 21096.9% | 20527.6% – 21680.9% |
| 1980s | 4113 | 25580.5% | 24990.3% – 26221.4% |
| 1990s | 3422 | 34974.8% | 33892.5% – 36079.0% |
| 2000s | 3418 | 42511.4% | 41240.4% – 43704.4% |
| 2010s | 4430 | 41439.1% | 40462.0% – 42491.9% |
| 2020s | 4149 | 38164.7% | 36883.5% – 39453.4% |

![lyric_length](figures/lyric_length_decade.png)

![lyric_length yearly](figures/lyric_length.png)

## Rival explanations, tested

### The two valence measures disagree — until you remove length bias

On an identical set of 2070 songs the raw comparison looks decisive against the sentiment finding:

| Measure | Sees context? | raw rho(year, valence) | p |
|---|---|---|---|
| NRC VAD word norms | no | -0.278 | 3.7e-38 |
| Entailment model | yes | -0.002 | 0.94 |

Read naively that says the lexicon result is an artefact. It is not that simple, because **both measures are length-dependent and in opposite directions**:

- lexicon: rho(words, valence) = **-0.227** — longer looks sadder
- contextual: rho(words, valence) = **+0.234** — longer looks happier

Lyrics roughly doubled in length, so those biases drive the two year-trends apart: the lexicon's decline is inflated and the contextual model's is masked. The apparent disagreement was mostly an artefact of the comparison.

**Opposite biases also settle whether to adjust.** A substantive effect cannot be negative in one valid measure and positive in another; two measures disagreeing in *sign* on the same nuisance variable is the signature of measurement error, which is the case where adjustment is correct. Adjusted for length, they converge:

| Measure | raw SD/decade | length-adjusted SD/decade | p |
|---|---|---|---|
| lexicon | -0.129 | **-0.100** | 1.1e-15 |
| contextual | -0.006 | **-0.085** | 1.3e-11 |

**Revised conclusion.** Hit lyrics did become modestly less positive — about 0.08–0.10 standard deviations per decade — and this replicates across two methods with very different failure modes on 2,070 songs. It is real but smaller than the raw lexicon series implies, and an earlier version of this report over-retracted it on the strength of the unadjusted comparison alone.

### Essentia mood scores move together, which means drift

Essentia's `mood_happy` falls sharply (tau -0.703, p=6.8e-17, 0.636 to 0.330). Taken alone that looks like strong evidence the music itself got sadder.

But `mood_sad` falls too (tau -0.297, p=0.00042, 0.514 to 0.427).

**Two opposing classifiers moving the same direction is diagnostic of drift, not emotion.** If songs were genuinely sadder, happy should fall while sad rises. Both falling points at something systematic in the audio — most plausibly production and mastering changes (loudness, compression, stereo width) shifting Essentia's features away from its 1990s training distribution. These two series are therefore **not reported as evidence about mood**.

### Minor-key share — a weaker signal than it first appears

Minor-key share roughly doubles across the period and is significant in all four headline variants, which made it look like the one solid musical result. Under the same scrutiny applied elsewhere it does not hold up well:

- **Genre mix explains about half of it** — +0.0223/decade unadjusted falls to +0.0090 with genre fixed effects (60% attenuation). Minor keys are simply more common in the genres that grew.
- **Within genre the direction is inconsistent** (cla +0.14, dan +0.32, hip +0.07, jaz +0.10, pop -0.03, rhy +0.44, roc -0.19); only 3 of 7 strata is significant and the signs disagree.
- **No trend within the post-1991 era** (tau -0.049, p=0.7), the period with one consistent chart methodology.

So the honest reading is that hits shifted toward minor keys mostly *because the genre mix shifted*, not because songwriting within genres moved that way. It is reported as suggestive, not established.

### The independence rise is real, but half the headline size

Method B scores verse-sized chunks and takes the **maximum**, which is what lets it find a self-sufficiency claim living in a single chorus. But the maximum of N draws rises with N even if nothing underlying changes, and lyrics roughly doubled in length over the period (rho(year, chunks) = +0.58; rho(chunks, p_max) = +0.39). Part of the apparent rise is therefore mechanical.

Unlike lyric length and valence — where length is a mediator and controlling it would remove real signal — this inflation is a property of the **estimator**, not of the music, so adjusting for it is correct.

- Unadjusted: **+0.0281/decade** (p=5.9e-50)
- Chunk-adjusted: **+0.0141/decade** (p=6.6e-11) — 50% attenuation

The trend nonetheless rises inside **every** fixed chunk-count stratum (short (1-4 chunks) n=2839, rho +0.064; medium (5-7) n=2416, rho +0.110; long (8+) n=1012, rho +0.183), including short songs where the bias cannot operate (2% in the 1950s to 10% in the 2020s). So the direction is solid and the **adjusted figure of about +1.4 points per decade should be read as the headline**, not the raw +2.7.

### Does the result depend on how the question was worded?

The independence finding rests on one sentence handed to a zero-shot model — *"The singer does not need this person and will be fine without them."* Zero-shot classifiers are phrasing-sensitive, so four paraphrases were scored on the same 1,380-song year-balanced sample.

| Phrasing | Mean share | Kendall tau | p |
|---|---|---|---|
| `original` | 12.3% | +0.457 | 2e-07 |
| `better_alone` | 0.8% | +0.035 | 0.72 |
| `self_sufficient` | 6.7% | +0.316 | 0.00052 |
| `rejecting` | 14.8% | +0.475 | 4.5e-08 |
| `no_need_love` | 4.0% | +0.365 | 8.6e-05 |

**Direction is robust: all five paraphrases give a positive trend and 4 of 5 are significant.** The exception, `better_alone`, fires on only 0.8% of songs — too strict a claim to have any statistical power — so it is degenerate rather than contradictory. Per-song scores correlate 0.31–0.76 across phrasings.

**But the absolute level is not robust.** Mean share ranges from 0.8% to 14.8% depending purely on wording. So *"the share of love songs about not needing a partner rose"* is supported; *"19% of love songs are about not needing a partner"* is **not** a fact about music, it is a fact about one sentence. Quote the trend, not the level.

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

### The independence rise continues inside the modern era

Restricted to 1991 onward — one consistent chart methodology — the trend is still present and significant (tau = +0.346, p = 0.003). An earlier pass on roughly half this much data found it non-significant within that window and read the rise as a single step around 2000; with the fuller sample it looks like a continuing climb rather than a one-off shift.

### Genre mix — tested, and not the driver

Rap and R&B went from absent to dominant on the Hot 100, and they have different lyrical conventions, so a change in the *mix* could move the average without any genre changing. Re-estimating the year effect with genre fixed effects:

| Metric | Unadjusted / decade | Genre-adjusted | Attenuation | n |
|---|---|---|---|---|
| lyric valence | -0.00797 | -0.00696 (p=5.6e-11) | 13% | 1415 |
| independence share | +0.02818 | +0.02679 (p=1.7e-13) | 5% | 2365 |

Genre mix accounts for only 13% of the lexicon valence trend, so it is not the explanation there.

For the independence trend genre control removes only 5% of the effect (p=1.7e-13 adjusted, n=2,365). **Genre mix is not driving it.** An earlier pass on a sixth as much genre-labelled data put the attenuation at 23% and lost significance; that was a power limitation and it resolved with more data.

Within individual genres the independence trend is positive in pop (+0.32), rhy (+0.27), roc (+0.32). It is flat in hip-hop, so this is **not** a rap phenomenon riding the genre shift — it appears inside the older guitar- and vocal-led genres too.

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
