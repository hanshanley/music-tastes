# Are US hit songs getting sadder, and are fewer of them about love?

_Generated 2026-07-26 02:53 UTC._

## What this measures

Every song that entered the Billboard Hot 100 between 1958-08-04 and the
present. The Hot 100 combines sales, radio airplay and (since 2007-2013)
streaming, so it is the closest long-run proxy available for what Americans
actually listened to. It is a proxy, not a census: see Limitations.

## Coverage, and why it is reported first


- Overall lyric coverage: **10.6%**
- Coverage ranges from 5.7% (1962) to 16.3% (2015)
- Spearman(year, coverage) = **+0.802** (p = 1.3e-16)

Coverage **is** year-dependent, so every result below is reported twice:
once on all covered songs and once on a complete-case subset holding the
number of songs per year constant. Where the two disagree in direction,
the result is marked unresolved rather than reported as a finding.

## Results

### lyric_valence

Lyric valence (NRC VAD, 0=negative 1=positive)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.479, p = 5.8e-09)
- Change per decade: -0.0080
- First 5 years 0.623 -> last 5 years 0.588
- Survives complete-case check: yes

![lyric_valence](figures/lyric_valence.png)

### lyric_sadness

Share of words with a sadness association (NRC EmoLex)  (higher = sadder)

- Direction: **falling, not significant** (Kendall tau = -0.024, p = 0.77)
- Change per decade: -0.0001
- First 5 years 0.023 -> last 5 years 0.021
- Survives complete-case check: yes

![lyric_sadness](figures/lyric_sadness.png)

### lyric_joy

Share of words with a joy association (NRC EmoLex)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.517, p = 3.4e-10)
- Change per decade: -0.0043
- First 5 years 0.042 -> last 5 years 0.026
- Survives complete-case check: yes

![lyric_joy](figures/lyric_joy.png)

### vader_valence

Lyric valence (VADER)  (higher = happier)

- Direction: **falling, significant** (Kendall tau = -0.578, p = 2.2e-12)
- Change per decade: -0.1077
- First 5 years 0.974 -> last 5 years 0.228
- Survives complete-case check: yes

![vader_valence](figures/vader_valence.png)

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
