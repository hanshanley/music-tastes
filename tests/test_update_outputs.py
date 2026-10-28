import json
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from music_tastes import (
    analysis_trends, coverage, exhibit, fetch_lyrics, gold_set, prompt_robustness,
    refresh, report, site, validity,
)
from music_tastes.http import Response


def test_gate_exhibit_excludes_missing_topic_labels():
    df = pd.DataFrame({
        "title_display": ["rejected", "accepted", "unscored"],
        "artist_display": ["Synthetic act"] * 3,
        "debut_year": [2000] * 3,
        "points": [1, 2, 3],
        "p_independence_max": [.95] * 3,
        "p_relationship_doc": [.2, .8, float("nan")],
        "is_relationship": [0, 1, float("nan")],
    })
    result = exhibit.gate_leakage(df)
    assert result["title_display"].tolist() == ["rejected"]


def test_lrclib_rejects_other_artists_versions_and_ambiguous_bodies(monkeypatch):
    row = {
        "id": 1, "trackName": "Example", "artistName": "Act",
        "plainLyrics": "Synthetic test text", "instrumental": False,
    }

    def response(rows):
        monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
            "", 200, json.dumps(rows), "", False
        ))

    response([row])
    assert fetch_lyrics.fetch_lrclib("Example", "Act")["source"] == "lrclib"
    for changed in (
        {"artistName": "Other act"}, {"artistName": "Act feat. Guest"},
        {"trackName": "Example (Live)"}, {"trackName": "Example (Remastered)"},
    ):
        response([{**row, **changed}])
        assert fetch_lyrics.fetch_lrclib("Example", "Act") is None
    response([row, {**row, "id": 2, "plainLyrics": "A different synthetic text"}])
    assert fetch_lyrics.fetch_lrclib("Example", "Act") is None


def test_lrclib_fallback_does_not_require_a_genius_match(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    monkeypatch.setattr(fetch_lyrics, "search_genius", lambda *a: [])
    monkeypatch.setattr(fetch_lyrics, "match_via_slug", lambda *a: None)
    monkeypatch.setattr(fetch_lyrics, "fetch_lrclib", lambda *a: {
        "lyrics": "This is synthetic text about you and me " * 5,
        "source": "lrclib", "match_tier": "exact-title-artist-lrclib",
    })
    song = SimpleNamespace(song_id="s0000000000000001", title_display="Example", artist_display="Act")
    record = fetch_lyrics.fetch_one(song, "", use_lrclib=True)
    assert record["has_lyrics"] and record["source"] == "lrclib"
    assert record["match_tier"] == "exact-title-artist-lrclib"


def test_lrclib_accepts_formatting_equivalence_without_rewriting_source(monkeypatch):
    rows = [
        {"id": 1, "trackName": "Example", "artistName": "Act",
         "plainLyrics": "[Verse 1]\nYou and me, together!", "instrumental": False},
        {"id": 2, "trackName": "Example", "artistName": "Act",
         "plainLyrics": "YOU AND ME TOGETHER", "instrumental": False},
    ]
    monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
        "", 200, json.dumps(rows), "", True
    ))
    result = fetch_lyrics.fetch_lrclib("Example", "Act")
    assert result["lyrics"] == rows[0]["plainLyrics"]
    assert result["source_record_id"] == 1


@pytest.mark.parametrize("second", [
    "You and me together together",
    "You and them together",
    "Me and you together",
    "You and m\u00e9 together",
])
def test_lrclib_still_rejects_word_repetition_substitution_and_order_changes(monkeypatch, second):
    row = {"id": 1, "trackName": "Example", "artistName": "Act",
           "plainLyrics": "You and me together", "instrumental": False}
    monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
        "", 200, json.dumps([row, {**row, "id": 2, "plainLyrics": second}]), "", True
    ))
    assert fetch_lyrics.fetch_lrclib("Example", "Act") is None


def test_lrclib_rejects_empty_word_sequence(monkeypatch):
    row = {"id": 1, "trackName": "Example", "artistName": "Act",
           "plainLyrics": "[Verse]\n...", "instrumental": False}
    monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
        "", 200, json.dumps([row]), "", True
    ))
    assert fetch_lyrics.fetch_lrclib("Example", "Act") is None


def test_lrclib_instrumental_metadata_is_resolved_without_fabricated_lyrics(monkeypatch, tmp_path):
    row = {"id": 1, "trackName": "Example", "artistName": "Act",
           "plainLyrics": None, "syncedLyrics": None, "instrumental": True}
    monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
        "", 200, json.dumps([row, {**row, "id": 2}]), "", True
    ))
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    monkeypatch.setattr(fetch_lyrics, "search_genius", lambda *a: [])
    monkeypatch.setattr(fetch_lyrics, "match_via_slug", lambda *a: None)
    song = SimpleNamespace(song_id="s0000000000000001", title_display="Example", artist_display="Act")
    record = fetch_lyrics.fetch_one(song, "", use_api=False, use_lrclib=True)
    assert record["status"] == "instrumental" and record["is_instrumental"]
    assert record["matched"] and not record["has_lyrics"]
    assert pd.isna(record["n_words"])
    assert record["instrumental_evidence"] == "lrclib_metadata"
    yearly = coverage.coverage_by_year(pd.DataFrame([{
        **record, "debut_year": 1960, "points": 100,
    }])).iloc[0]
    assert yearly["n_matched"] == 1
    assert yearly["n_lyrics"] == 0 and yearly["n_eligible"] == 0
    cached = json.loads((tmp_path / f"{song.song_id}.json").read_text())
    assert cached["lyrics"] is None and cached["source_record_id"] == 1

    def no_network(*a, **k):
        raise AssertionError("Resolved instrumental should reuse its cache")

    monkeypatch.setattr(fetch_lyrics, "search_genius", no_network)
    result = fetch_lyrics.fetch_one(
        song, "", retry_failed=True, retry_after_days=0, use_lrclib=True
    )
    assert result["status"] == "instrumental" and not result["has_lyrics"]


@pytest.mark.parametrize("changed", [
    {"instrumental": False}, {"instrumental": None},
    {"plainLyrics": "Contradictory text"}, {"syncedLyrics": "[00:01]Contradictory text"},
])
def test_lrclib_instrumental_metadata_requires_unanimous_noncontradictory_evidence(monkeypatch, changed):
    row = {"id": 1, "trackName": "Example", "artistName": "Act",
           "plainLyrics": None, "syncedLyrics": None, "instrumental": True}
    monkeypatch.setattr(fetch_lyrics, "get", lambda *a, **k: Response(
        "", 200, json.dumps([row, {**row, "id": 2, **changed}]), "", True
    ))
    assert fetch_lyrics.fetch_lrclib("Example", "Act") is None


@pytest.mark.parametrize("title,artist", [
    ("Tea For Two Cha Cha", "The Tommy Dorsey Orchestra"),
    ("Whole Lotta Love", "C.C.S."),
    ("Blue Hawaii", "Billy Vaughn And His Orchestra"),
    ("Morning After", "Mar-Keys"),
    ("The Happening", "Herb Alpert & The Tijuana Brass"),
    ("Alley Cat", "Bent Fabric and His Piano"),
    ("One Mint Julep", "Ray Charles"),
    ("Casino Royale", "Herb Alpert & The Tijuana Brass"),
    ("Peter Gunn", "Deodato"),
    ("Greensleeves", "Mason Williams"),
])
def test_recording_correction_excludes_misattributed_words_and_preserves_evidence(
    monkeypatch, tmp_path, title, artist,
):
    song = SimpleNamespace(song_id="s0000000000000001", title_display=title, artist_display=artist)
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    path = tmp_path / f"{song.song_id}.json"
    original = {
        "song_id": song.song_id, "has_lyrics": True,
        "lyrics": "Synthetic words wrongly attached to an instrumental.",
    }
    path.write_text(json.dumps(original))

    def no_network(*a, **k):
        raise AssertionError("A source-backed recording correction must not search for lyrics")

    monkeypatch.setattr(fetch_lyrics, "search_genius", no_network)
    record = fetch_lyrics.fetch_one(song, "")
    assert record["status"] == "instrumental" and record["is_instrumental"]
    assert not record["has_lyrics"] and not record["is_english"]
    assert record["instrumental_evidence"] == "recording_source" and record["source_url"]
    assert json.loads(path.read_text())["lyrics"] is None
    assert pd.isna(record["n_words"])
    assert json.loads((tmp_path / "rejected_recordings" / path.name).read_text()) == original
    before = path.read_bytes()
    fetch_lyrics.fetch_one(song, "", retry_failed=True, retry_after_days=0)
    assert path.read_bytes() == before


def test_recording_corrections_do_not_transfer_to_other_artists_or_versions():
    for title, artist in [
        ("Whole Lotta Love", "Led Zeppelin"),
        ("Whole Lotta Love (Live)", "C.C.S."),
        ("Tea For Two", "The Tommy Dorsey Orchestra"),
        ("Tea For Two Cha Cha", "Other Orchestra"),
        ("Morning After", "Other Act"),
        ("The Happening", "The Supremes"),
        ("Alley Cat (Live)", "Bent Fabric and His Piano"),
        ("Let's Go (pony)", "The Routers"),
        ("Lonely Boy, Lonely Guitar", "Duane Eddy"),
        ("Saturday Night At The World", "Mason Williams"),
        ("Grazing In The Grass", "The Friends Of Distinction"),
        ("Mercy, Mercy, Mercy", "The Buckinghams"),
        ("Red Roses For A Blue Lady", "Vic Dana"),
    ]:
        song = SimpleNamespace(song_id="test", title_display=title, artist_display=artist)
        assert fetch_lyrics.recording_instrumental(song) is None


def test_cached_vocal_song_with_instrumental_section_is_reclassified_without_refetch(
    monkeypatch, tmp_path,
):
    song = SimpleNamespace(song_id="test", title_display="Example", artist_display="Act")
    text = "[Verse]\n" + "You and me together in the world " * 5 + "\n[Instrumental]"
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    path = tmp_path / "test.json"
    original = {
        "song_id": "test", "has_lyrics": True, "is_instrumental": True,
        "is_english": True, "status": "instrumental", "lyrics": text,
        "source": "synthetic", "retrieved_at": "2026-01-01T00:00:00Z",
    }
    path.write_text(json.dumps(original))

    def no_network(*a, **k):
        raise AssertionError("Eligibility repair must reuse the existing transcription")

    monkeypatch.setattr(fetch_lyrics, "search_genius", no_network)
    record = fetch_lyrics.fetch_one(song, "")
    assert record["status"] == "usable" and not record["is_instrumental"]
    corrected = json.loads(path.read_text())
    for key in ("lyrics", "source", "retrieved_at"):
        assert corrected[key] == original[key]
    before = path.read_bytes()
    fetch_lyrics.fetch_one(song, "")
    assert path.read_bytes() == before


def test_partial_fetch_queue_includes_never_attempted_songs(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_lyrics, "DERIVED", tmp_path)
    from music_tastes import paths

    monkeypatch.setattr(paths, "REPORTS", tmp_path)
    pd.DataFrame({
        "song_id": ["a", "b"], "debut_year": [1960, 1961], "points": [100, 10],
    }).to_parquet(tmp_path / "songs_weighted.parquet")
    monkeypatch.setattr(fetch_lyrics, "fetch_one", lambda *a: {
        "song_id": "a", "has_lyrics": False, "matched": False, "status": "no_confident_match",
    })
    fetch_lyrics.run(limit=1, use_api=False)
    queue = pd.read_csv(tmp_path / "lyrics_unresolved.csv").set_index("song_id")
    assert queue.loc["a", "status"] == "no_confident_match"
    assert queue.loc["b", "status"] == "never_attempted"


def test_fetch_index_is_identical_despite_completion_and_field_order(monkeypatch, tmp_path):
    from music_tastes import paths

    monkeypatch.setattr(fetch_lyrics, "DERIVED", tmp_path)
    monkeypatch.setattr(paths, "REPORTS", tmp_path)
    pd.DataFrame({
        "song_id": ["a", "b"], "debut_year": [1960, 2000], "points": [100, 10],
    }).to_parquet(tmp_path / "songs_weighted.parquet")
    reverse = False

    def fetch(song, *args):
        record = {"song_id": song.song_id, "matched": False, "has_lyrics": False}
        return dict(reversed(list(record.items()))) if reverse else record

    monkeypatch.setattr(fetch_lyrics, "fetch_one", fetch)
    fetch_lyrics.run(use_api=False)
    original = (tmp_path / "lyrics_index.parquet").read_bytes()
    reverse = True
    monkeypatch.setattr(fetch_lyrics, "as_completed", lambda futures: reversed(futures))
    fetch_lyrics.run(use_api=False)
    assert (tmp_path / "lyrics_index.parquet").read_bytes() == original


@pytest.mark.parametrize("record", [[], {"song_id": "s0000000000000001", "has_lyrics": True}])
def test_malformed_lyrics_cache_is_retried(monkeypatch, tmp_path, record):
    song_id = "s0000000000000001"
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    (tmp_path / f"{song_id}.json").write_text(json.dumps(record))
    monkeypatch.setattr(fetch_lyrics, "search_genius", lambda *a: [])
    monkeypatch.setattr(fetch_lyrics, "match_via_slug", lambda *a: None)
    song = SimpleNamespace(song_id=song_id, title_display="Synthetic", artist_display="Test")
    result = fetch_lyrics.fetch_one(song, "", use_api=False)
    assert result["status"] == "no_confident_match" and not result["has_lyrics"]


def test_method_a_cache_preserves_partial_runs_and_retries_changed_or_partial_records(monkeypatch, tmp_path):
    from music_tastes import lyrics_features

    for name in ("DERIVED", "CACHE", "LEX_DIR"):
        monkeypatch.setattr(lyrics_features, name, tmp_path)
    ids = ["s0000000000000001", "s0000000000000002"]
    pd.DataFrame({
        "song_id": ids, "has_lyrics": True, "is_english": True, "is_instrumental": False,
    }).to_parquet(tmp_path / "lyrics_index.parquet")
    texts = dict(zip(ids, ["Synthetic text about people", "Other synthetic example"]))
    monkeypatch.setattr(lyrics_features, "load_lyrics", texts.get)
    monkeypatch.setattr(lyrics_features, "_load_lexicons", lambda: (pd.DataFrame(),) * 3)
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return pd.DataFrame([{
            "emb_relationship_margin": .2, "emb_independence_axis": .3,
            "emb_stance": lyrics_features.STANCE_LABELS[0],
        }] * len(texts))

    monkeypatch.setattr(lyrics_features, "embedding_features", embed)
    assert len(lyrics_features.run(limit=1)) == 1
    assert len(lyrics_features.run(limit=1)) == 2
    lyrics_features.run()
    assert calls == [1, 1]
    texts[ids[0]] += " changed"
    lyrics_features.run()
    assert calls == [1, 1, 1]
    cached = tmp_path / "features_a" / f"{ids[1]}.json"
    record = json.loads(cached.read_text())
    del record["emb_independence_axis"]
    cached.write_text(json.dumps(record))
    assert len(lyrics_features.run()) == 2
    assert calls == [1, 1, 1, 1]
    cached_bytes = cached.read_bytes()
    index = pd.read_parquet(tmp_path / "lyrics_index.parquet")
    index.loc[index["song_id"].eq(ids[1]), ["has_lyrics", "is_english", "is_instrumental"]] = [
        False, False, True,
    ]
    index.to_parquet(tmp_path / "lyrics_index.parquet")
    assert lyrics_features.run()["song_id"].tolist() == [ids[0]]
    assert cached.read_bytes() == cached_bytes
    assert calls == [1, 1, 1, 1]


def test_nli_output_excludes_corrected_instrumental_without_deleting_old_private_scores(
    monkeypatch, tmp_path,
):
    from music_tastes import stance_nli

    monkeypatch.setattr(stance_nli, "DERIVED", tmp_path)
    monkeypatch.setattr(stance_nli, "NLI_CACHE", tmp_path / "nli")
    ids = ["s0000000000000001", "s0000000000000002"]
    texts = dict.fromkeys(ids, "Synthetic relationship text")
    monkeypatch.setattr(stance_nli, "load_lyrics", texts.get)

    def no_model(*a, **k):
        raise AssertionError("Unchanged eligible scores must be reused")

    monkeypatch.setattr(stance_nli, "_build_pipeline", no_model)
    index = pd.DataFrame({
        "song_id": ids, "has_lyrics": True, "is_english": True, "is_instrumental": False,
    })
    index.to_parquet(tmp_path / "lyrics_index.parquet")
    pd.DataFrame({
        "song_id": ids, "debut_year": [1960, 1961], "points": [100, 50],
    }).to_parquet(tmp_path / "songs_weighted.parquet")
    for sid in ids:
        record = {
            "song_id": sid, "signature": stance_nli.cache_signature(texts[sid]),
            "p_relationship_doc": .8,
            **{f"p_{key}_max": .4 for key in stance_nli.CHUNK_KEYS},
        }
        stance_nli._cache_path(sid).write_text(json.dumps(record))
    assert set(stance_nli.run(refresh_index=False)["song_id"]) == set(ids)
    old_score = stance_nli._cache_path(ids[1])
    cached_bytes = old_score.read_bytes()
    index.loc[1, ["has_lyrics", "is_english", "is_instrumental"]] = [False, False, True]
    index.to_parquet(tmp_path / "lyrics_index.parquet")
    texts[ids[1]] = None
    assert stance_nli.run(refresh_index=False)["song_id"].tolist() == [ids[0]]
    assert old_score.read_bytes() == cached_bytes


def test_nli_written_signature_reuses_once_cleaned_inputs_with_nested_section_labels(
    monkeypatch, tmp_path,
):
    from music_tastes import stance_nli

    monkeypatch.setattr(stance_nli, "DERIVED", tmp_path)
    monkeypatch.setattr(stance_nli, "NLI_CACHE", tmp_path / "nli")
    sid = "s0000000000000001"
    raw = "[" + "x" * 75 + "[inner]] Synthetic words about two people"
    monkeypatch.setattr(stance_nli, "load_lyrics", lambda _: raw)
    pd.DataFrame({
        "song_id": [sid], "has_lyrics": True, "is_english": True, "is_instrumental": False,
    }).to_parquet(tmp_path / "lyrics_index.parquet")
    pd.DataFrame({
        "song_id": [sid], "debut_year": [1960], "points": [100],
    }).to_parquet(tmp_path / "songs_weighted.parquet")
    monkeypatch.setattr(stance_nli, "_build_pipeline", lambda *a, **k: object())
    inputs = []

    def score(clf, texts, batch_size):
        inputs.append(texts[sid])
        return {sid: {
            "p_relationship_doc": .8,
            **{f"p_{key}_max": .4 for key in stance_nli.CHUNK_KEYS},
        }}

    monkeypatch.setattr(stance_nli, "score_batch", score)
    assert stance_nli.run(refresh_index=False)["song_id"].tolist() == [sid]
    assert stance_nli.read_record(sid) is not None
    before = stance_nli._cache_path(sid).read_bytes()
    assert stance_nli.run(refresh_index=False)["song_id"].tolist() == [sid]
    assert inputs == [stance_nli.clean_lyrics(raw)]
    assert stance_nli._cache_path(sid).read_bytes() == before


def test_coverage_changes_use_percentage_points_and_leave_unmatched_years_missing():
    previous = pd.DataFrame({
        "debut_year": [1960], "n_songs": [100], "n_lyrics": [80],
        "coverage_lyrics": [.8], "coverage_exposure": [.9],
    })
    current = pd.DataFrame({
        "debut_year": [1960, 2026], "n_songs": [100, 100], "n_lyrics": [90, 90],
        "coverage_lyrics": [.9, .9], "coverage_exposure": [.95, .95],
    })
    result = coverage.coverage_changes(previous, current, "debut_year").set_index("debut_year")
    assert result.loc[1960, "coverage_lyrics_change_pp"] == pytest.approx(10)
    assert pd.isna(result.loc[2026, "coverage_lyrics_change_pp"])


@pytest.mark.parametrize("current_inputs", [True, False])
def test_matched_recovery_audit_requires_current_inputs_and_exports_only_counts(
    monkeypatch, tmp_path, current_inputs,
):
    from music_tastes.cache import fingerprint

    for name in ("DATA", "DERIVED", "ROOT", "INTERIM", "REPORTS"):
        monkeypatch.setattr(coverage, name, tmp_path)
    index = tmp_path / "lyrics_index.parquet"
    index.write_bytes(b"synthetic index fingerprint")
    (tmp_path / "chart_integrity.json").write_text(json.dumps({"chart_through": "2026-10-31"}))
    (tmp_path / "lyric_recovery_audit.json").write_text(json.dumps({
        "lyrics_index_signature": fingerprint({}, [index]) if current_inputs else "stale",
        "additional_lyrics": 4, "removed_misattributed_lyrics": 1,
        "eligibility_reclassifications": 2, "private_field": "not for export",
    }))
    frame = pd.DataFrame({
        "debut_year": [1960], "n_songs": [100], "n_lyrics": [80],
        "total_points": [1000], "covered_points": [900],
        "coverage_lyrics": [.8], "coverage_exposure": [.9],
    })
    coverage.write_change_audit(frame, {})
    audit = json.loads((tmp_path / "gap_audit.json").read_text())
    if current_inputs:
        assert audit["matched_recovery"] == {
            "additional_lyrics": 4, "removed_recording_transcriptions": 1,
            "eligibility_reclassifications": 2,
            "removal_note": (
                "Excluded instrumental-recording transcriptions; this does not establish "
                "that every attached word was fabricated or that the recordings lack speech."
            ),
        }
    else:
        assert "matched_recovery" not in audit


def test_gap_audit_without_prior_snapshot_does_not_invent_baseline(monkeypatch, tmp_path):
    for name in ("ROOT", "INTERIM", "REPORTS"):
        monkeypatch.setattr(coverage, name, tmp_path)
    (tmp_path / "chart_integrity.json").write_text(json.dumps({"chart_through": "2026-10-31"}))
    current = pd.DataFrame({
        "debut_year": [1960], "n_songs": [100], "n_lyrics": [90],
        "total_points": [1000], "covered_points": [950],
        "coverage_lyrics": [.9], "coverage_exposure": [.95],
    })
    coverage.write_change_audit(current, {"n_songs": 100})
    audit = json.loads((tmp_path / "gap_audit.json").read_text())
    assert audit["baseline"] is None
    change = pd.read_csv(tmp_path / "coverage_change_by_decade.csv")
    assert change["coverage_lyrics_change_pp"].isna().all()


def test_site_checks_complete_fingerprint_and_cleans_only_generated_assets(monkeypatch, tmp_path):
    from music_tastes.cache import fingerprint

    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(site, "REPORTS", reports)
    csv = reports / "series.csv"
    csv.write_text("year,value\n1960,0.2\n")
    (reports / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "chart_through": "2026-10-31",
        "output_signature": fingerprint({}, [csv]),
    }))
    output = site.run(tmp_path / "preview")
    assert "Complete analysis through chart issue 2026-10-31" in output.read_text()
    unrelated = output.parent / "assets" / "keep.txt"
    unrelated.write_text("not a generated asset")
    csv.write_text("year,value\n1960,0.8\n")
    site.run(output.parent)
    assert "Unverified historical snapshot" in output.read_text()
    csv.unlink()
    site.run(output.parent)
    assert not (output.parent / "assets" / "series.csv").exists()
    assert unrelated.exists()


@pytest.mark.parametrize("with_page", [False, True])
def test_site_vad_claim_and_chart_formats(monkeypatch, tmp_path, with_page):
    from music_tastes.cache import fingerprint

    reports = tmp_path / "reports"
    (reports / "figures").mkdir(parents=True)
    monkeypatch.setattr(site, "REPORTS", reports)
    (reports / "figures" / "lyrical-emotion-dimensions.png").write_bytes(b"synthetic chart")
    if with_page:
        page = reports / "figures" / "page_background"
        page.mkdir()
        (page / "lyrical-emotion-dimensions.png").write_bytes(b"synthetic page-background chart")
    (reports / "nrc_vad_statistics.json").write_text(json.dumps({"tests": [
        {"is_primary": True, "method": "newey_west", "window_years": 5,
         "dimension": "valence", "slope_per_decade": -.004,
         "reject_primary_05": True, "p_holm_primary": .003},
        {"is_primary": True, "method": "newey_west", "window_years": 5,
         "dimension": "dominance", "slope_per_decade": .0001,
         "reject_primary_05": False, "p_holm_primary": .84},
    ]}))
    (reports / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "chart_through": "2026-10-31",
        "output_signature": fingerprint({}, sorted(p for p in reports.rglob("*") if p.is_file())),
    }))
    preview = site.run(tmp_path / "preview")
    text = preview.read_text()
    assert "valence: decreases (p=0.003)" in text
    assert "dominance: no clear linear trend (p=0.84)" in text
    assert "Holm-adjusted" in text
    assert (preview.parent / "assets" / "lyrical-emotion-dimensions.png").read_bytes() == b"synthetic chart"
    if with_page:
        assert 'src="assets/page_lyrical-emotion-dimensions.png"' in text
        assert "download>Margin of Error</a>" in text and "download>Substack</a>" in text
        assert (preview.parent / "assets" / "page_lyrical-emotion-dimensions.png").read_bytes() == (
            b"synthetic page-background chart"
        )
    else:
        assert 'src="assets/lyrical-emotion-dimensions.png"' in text
        assert "page_lyrical-emotion-dimensions.png" not in text


def test_publish_preserves_previous_report_on_marker_failure(tmp_path):
    staged, destination = tmp_path / "staged", tmp_path / "reports"
    staged.mkdir()
    destination.mkdir()
    (staged / "findings.md").write_text("new")
    (destination / "findings.md").write_text("old")

    def fail():
        raise OSError("marker failed")

    with pytest.raises(OSError, match="marker"):
        refresh.publish(staged, destination, fail)
    assert (destination / "findings.md").read_text() == "old"
    refresh.publish(staged, destination)
    assert (destination / "findings.md").read_text() == "new"
    assert (staged / "findings.md").exists()
    assert not list(tmp_path.glob(".reports-*"))


def test_limited_refresh_never_publishes(monkeypatch, tmp_path):
    for name in ("DATA", "RAW", "INTERIM", "DERIVED", "REPORTS"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(refresh, name, path)
    (refresh.REPORTS / "findings.md").write_text("previous complete report")
    commands = []
    monkeypatch.setattr(refresh.subprocess, "run", lambda command, **k: commands.append(command))
    result = refresh._run_locked(2, 1, False, False, None, retry_after_days=0)
    assert result["status"] == "partial"
    assert (refresh.REPORTS / "findings.md").read_text() == "previous complete report"
    assert not (refresh.DATA / "last_complete.json").exists()
    fetch = next(command for command in commands if "fetch-lyrics" in command)
    assert fetch[fetch.index("--retry-after-days") + 1] == "0"


def test_update_cli_passes_retry_age(monkeypatch):
    from music_tastes import cli

    calls = []
    monkeypatch.setattr(refresh, "run", lambda **kwargs: calls.append(kwargs))
    cli.main(["update", "--retry-after-days", "0", "--no-api"])
    assert calls[0]["retry_after_days"] == 0 and not calls[0]["use_api"]


def test_failed_stage_never_publishes(monkeypatch, tmp_path):
    for name in ("DATA", "RAW", "INTERIM", "DERIVED", "REPORTS"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(refresh, name, path)
    (refresh.REPORTS / "findings.md").write_text("old")
    monkeypatch.setattr(refresh.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("stage failed")
    ))
    with pytest.raises(RuntimeError, match="stage failed"):
        refresh._run_locked(None, 1, False, False, None)
    assert (refresh.REPORTS / "findings.md").read_text() == "old"
    manifests = list((refresh.DATA / "runs").glob("*/manifest.json"))
    assert json.loads(manifests[0].read_text())["status"] == "failed"


def test_chart_year_uses_annual_not_lifetime_points():
    songs = pd.DataFrame({
        "song_id": ["a"], "debut_year": [2025], "points": [1000],
        "is_relationship": [1.0],
    })
    annual = pd.DataFrame({
        "song_id": ["a", "a"], "chart_year": [2025, 2026], "points": [300, 700],
    })
    frame = analysis_trends.chart_year_frame(songs, annual)
    assert frame["points"].tolist() == [300, 700]
    assert frame["debut_year"].tolist() == [2025, 2026]
    assert frame["cohort_year"].tolist() == [2025, 2025]


def test_fixed_followup_excludes_late_points_and_incomplete_cohorts():
    songs = pd.DataFrame({
        "song_id": ["a", "b", "c"], "debut_year": [2024, 2025, 2025],
        "first_charted": ["2024-01-06", "2025-01-04", "2025-12-27"],
        "points": [9999, 9999, 9999],
    })
    entries = pd.DataFrame({
        "song_id": ["a", "a", "a", "b", "c"],
        "chart_week": ["2024-01-06", "2024-12-28", "2025-01-04", "2025-01-04", "2026-10-31"],
        "rank": [1, 2, 1, 1, 1],
    })
    matched = analysis_trends.fixed_followup_frame(songs, entries)
    assert matched["song_id"].tolist() == ["a"]
    assert matched["points"].tolist() == [199]


def test_review_packets_are_disjoint_blind_and_record_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    frame = pd.DataFrame({
        "song_id": [f"s{i}" for i in range(12)],
        "title_display": "Synthetic", "artist_display": "Test",
        "debut_year": [1960] * 6 + [2000] * 6,
        "has_lyrics": True, "p_relationship_doc": .5,
    })
    monkeypatch.setattr(coverage, "load_joined", lambda: frame)
    sample = gold_set.prepare_review(per_decade=2)
    challenge = pd.read_csv(tmp_path / "review_challenge.csv")
    assert len(sample) == 4 and len(challenge) == 4
    assert set(sample["song_id"]).isdisjoint(challenge["song_id"])
    assert sample["sample_probability"].tolist() == [1 / 3] * 4
    assert "p_relationship_doc" not in sample and "gold_independence" in sample
    with pytest.raises(SystemExit, match="overwrite"):
        gold_set.prepare_review(per_decade=2)


def test_review_all_uncertain_is_not_completed_accuracy(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    pd.DataFrame({
        "song_id": ["a"], "gold_relationship": ["uncertain"],
        "reviewer_id": ["reviewer"], "decade": [1960],
    }).to_csv(tmp_path / "review_packet.csv", index=False)
    pd.DataFrame({
        "song_id": ["a"], "p_relationship_doc": [.8],
    }).to_parquet(tmp_path / "lyric_features_method_b.parquet")
    result = gold_set.evaluate_review()
    assert result["status"] == "unscorable" and result["n_scored"] == 0


def test_review_conditional_stance_includes_topic_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    pd.DataFrame({
        "song_id": ["a", "b"], "gold_relationship": ["1", "0"],
        "gold_independence": ["1", "0"], "reviewer_id": ["reviewer"] * 2,
        "decade": [1960, 1960],
    }).to_csv(tmp_path / "review_packet.csv", index=False)
    pd.DataFrame({
        "song_id": ["a", "b"], "p_relationship_doc": [.2, .8],
        "p_independence_max": [.9, .1],
    }).to_parquet(tmp_path / "lyric_features_method_b.parquet")
    result = gold_set.evaluate_review()
    assert result["conditional_stances"]["independence"]["fn"] == 1
    assert result["ungated_independence"]["accuracy"] == 1
    assert result["stance_review_status"] == "partial"


def test_legacy_validation_does_not_turn_missing_method_a_into_negatives(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    monkeypatch.setattr(gold_set, "GOLD_DIR", tmp_path / "gold")
    monkeypatch.setattr(gold_set, "RANDOM_GOLD", {
        "a": (1, None, "A"), "b": (0, None, "B"), "missing": (0, None, "Missing"),
    })
    pd.DataFrame({
        "song_id": ["a", "b"], "p_relationship_doc": [.8, .2],
    }).to_parquet(tmp_path / "lyric_features_method_b.parquet")
    pd.DataFrame({
        "song_id": ["a", "b"], "emb_relationship_margin": [.3, None],
    }).to_parquet(tmp_path / "lyric_features_method_a.parquet")
    result = gold_set.evaluate_random()
    assert result["n_missing_score"] == 1
    assert result["method_a_relationship"]["n"] == 1


def test_relationship_sensitivity_separates_long_input_disagreement():
    frame = pd.DataFrame({
        "original": [.2, .8], "main_subject": [.2, .8], "describes": [.3, .7],
        "window_max": [.9, .8], "window_mean": [.6, .8],
        "n_windows": [2, 1], "decade": [1960, 2000],
    })
    result = prompt_robustness.relationship_summary(frame)
    broad = result["comparisons"]["window_max"]
    assert broad["label_disagreement"] == .5
    assert broad["label_disagreement_multiple_windows"] == 1
    assert result["comparisons"]["main_subject"]["label_disagreement"] == 0


def test_contextual_cache_reuses_only_valid_matching_summary(monkeypatch, tmp_path):
    from music_tastes import cache, stance_nli

    monkeypatch.setattr(validity, "DERIVED", tmp_path)
    monkeypatch.setattr(cache, "DERIVED", tmp_path)
    path = tmp_path / "contextual_valence.parquet"
    pd.DataFrame({
        "song_id": ["a"], "nli_valence": [.2], "vad_valence": [.3],
    }).to_parquet(path)
    signature = cache.analysis_signature({
        "mood_hypotheses": validity.MOOD_HYPOTHESES, "per_year": 12,
    })
    cache.mark_cache(path, signature)
    path.with_suffix(".summary.json").write_text(json.dumps({"n": 1, "cached": True}))
    monkeypatch.setattr(validity, "load_joined", lambda: (_ for _ in ()).throw(
        RuntimeError("recompute")
    ))
    monkeypatch.setattr(stance_nli, "_build_pipeline", lambda *a, **k: pytest.fail("inference"))
    assert validity.contextual_valence_check()["cached"]
    with pytest.raises(RuntimeError, match="recompute"):
        validity.contextual_valence_check(per_year=13)
    path.with_suffix(".summary.json").write_text("[]")
    with pytest.raises(RuntimeError, match="recompute"):
        validity.contextual_valence_check()


def test_balancing_happens_after_metric_missingness():
    df = pd.DataFrame({
        "debut_year": [2000] * 20 + [2001] * 20,
        "has_lyrics": True, "points": list(range(40)),
        "metric": [None] * 10 + [1.0] * 30,
    })
    sub = coverage.complete_case_subset(df, column="metric")
    assert sub.groupby("debut_year").size().tolist() == [10, 10]
    assert sub["metric"].notna().all()


def test_review_uncertain_is_reviewed_but_not_scored(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    pd.DataFrame({
        "song_id": ["a", "b"], "gold_relationship": ["1", "uncertain"],
        "reviewer_id": ["reviewer", "reviewer"], "decade": [1960, 1960],
    }).to_csv(tmp_path / "review_packet.csv", index=False)
    pd.DataFrame({
        "song_id": ["a"], "p_relationship_doc": [0.8],
    }).to_parquet(tmp_path / "lyric_features_method_b.parquet")
    result = gold_set.evaluate_review()
    assert result["status"] == "complete"
    assert result["n_reviewed"] == 2 and result["n_scored"] == 1
    assert result["n_uncertain"] == 1


def test_review_rejects_duplicate_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(gold_set, "DERIVED", tmp_path)
    pd.DataFrame({
        "song_id": ["a", "a"], "gold_relationship": ["1", "0"],
        "reviewer_id": ["reviewer", "reviewer"],
    }).to_csv(tmp_path / "review_packet.csv", index=False)
    with pytest.raises(ValueError, match="unique"):
        gold_set.evaluate_review()


def test_gold_metrics_show_small_sample_uncertainty():
    metrics = gold_set._metrics([1] * 15, [1] * 15)
    assert 0.79 < metrics["ci95_wilson"]["accuracy"][0] < 0.81
    assert gold_set._metrics([1, 0], [0, 1])["f1"] == 0


def test_report_claim_changes_with_evidence():
    positive = {"weighted_all": {"kendall_tau": 0.6, "p_value": 0.001, "significant_at_05": True}}
    assert report.summary_for("metric", {"metric": positive}).startswith("Increasing")
    negative = {"weighted_all": {**positive["weighted_all"], "kendall_tau": -0.6}}
    assert report.summary_for("metric", {"metric": negative}).startswith("Decreasing")
    negative["weighted_complete_case"] = {"kendall_tau": 0.3}
    assert report.summary_for("metric", {"metric": negative}).startswith("Unresolved")


@pytest.mark.parametrize("metric,mean,expected", [("bpm", 117.0, 117.0), ("relationship_share", 0.6, 60)])
def test_decade_plot_preserves_units(monkeypatch, tmp_path, metric, mean, expected):
    heights = []
    import matplotlib.axes
    original = matplotlib.axes.Axes.bar

    def capture(self, x, height, *args, **kwargs):
        heights.extend(height)
        return original(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", capture)
    frame = pd.DataFrame({
        "metric": [metric], "variant": ["weighted_all"], "decade": [2000],
        "mean": [mean], "ci_lo": [mean * .9], "ci_hi": [mean * 1.1],
    })
    report._plot_decades(frame, metric, "Test", tmp_path / "figure.png")
    assert heights == [expected]


def test_site_exports_complete_series_and_resolves_relative_assets(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(site, "REPORTS", reports)
    csv = "year,n,mean,ci_lo,ci_hi\n2000,12,0.6,0.5,0.7\n"
    (reports / "yearly_series.csv").write_text(csv)
    output = site.run(tmp_path / "preview")
    assert (output.parent / "assets" / "yearly_series.csv").read_text() == csv
    text = output.read_text()
    assert "Unverified historical snapshot" in text
    assert "scores strictly above 0.5" in text

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"href", "src"} and not value.startswith(("#", "http")):
                    assert (output.parent / value).exists(), value

    Links().feed(text)
