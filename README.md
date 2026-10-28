# Music tastes

How have relationships and mood in US hit songs changed?

A reproducible Billboard Hot 100 retrospective: **32,732 songs**, from August 4,
1958 through the **October 31, 2026 chart issue**. 

Lyrics cover **89.4% of songs** and **94.3% of chart exposure**. All **28,885
eligible songs** have both sets of features; missing lyrics remain missing.

## Explore

- [Findings and figures](reports/findings.md)
- [Methods, relationship classification, and reproduction](METHODS.md)
- [Coverage and remaining gaps](reports/gap_audit.json)
- [Data sources and recording corrections](DATA_SOURCES.md)

“Relationship” means **romantic or sexual relationships**, not every form of love.
A local NLI model tests the first 380 cleaned words; scores strictly above 0.5
count as positive. Independent human validation remains pending.

## Run locally

```bash
uv sync --locked --python 3.12 --extra all
uv run --locked --extra all music-tastes update --as-of 2026-10-31
uv run --locked --extra all music-tastes site
python3 -m http.server 8000 --directory site
```

Open `http://localhost:8000` for the preview and complete aggregate downloads.
The Genius token is optional. Models and private lyric caches stay local and
are not included in the repository; see [reproduction details](METHODS.md).

This is a fixed retrospective, not an automatically updated service. Website
deployment is separate work.
