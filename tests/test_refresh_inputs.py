import json
from types import SimpleNamespace

import pandas as pd
import pytest

from music_tastes import fetch_lyrics, ingest_charts, stance_nli
from music_tastes.http import REFRESH_ERRORS, Response


def chart(dates):
    return pd.DataFrame([
        {"chart_week": pd.Timestamp(date).date(), "rank": rank, "title": f"Title {rank}", "artist": "Artist"}
        for date in dates for rank in range(1, 101)
    ])


def test_chart_integrity_distinguishes_date_transition_from_missing_week():
    assert not ingest_charts.validate_charts(chart(["1961-12-25", "1962-01-06"]))["gaps_before"]
    assert ingest_charts.validate_charts(chart(["2026-10-10", "2026-10-24"]))["gaps_before"]
    duplicate = pd.concat([chart(["2026-10-24"])] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_charts.validate_charts(duplicate)


def test_crosscheck_includes_artist_and_missing_position():
    primary = chart(["2026-10-24"])
    secondary = primary.iloc[:-1].copy()
    secondary.loc[0, "artist"] = "Different performer"
    result = ingest_charts._cross_check(primary, secondary)
    assert result["missing_secondary_positions"] == 1
    assert result["mismatched_positions"] == 1


def test_invalid_download_preserves_good_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_charts, "RAW", tmp_path)
    local = tmp_path / "billboard_hot100_all.json"
    local.write_text("previous archive")
    monkeypatch.setattr(ingest_charts, "get", lambda *a, **k: Response("", 200, "[]", "", False))
    with pytest.raises((ValueError, KeyError)):
        ingest_charts._fetch_primary(refresh=True)
    assert local.read_text() == "previous archive"


def test_retry_failure_preserves_success_and_retries_old_misses(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_lyrics, "LYRICS_CACHE", tmp_path)
    song = SimpleNamespace(song_id="s0000000000000001", title_display="Example", artist_display="Act")
    path = tmp_path / f"{song.song_id}.json"
    path.write_text(json.dumps({"has_lyrics": False, "retrieved_at": "2020-01-01T00:00:00Z"}))
    monkeypatch.setattr(fetch_lyrics, "search_genius", lambda *a: [])
    monkeypatch.setattr(fetch_lyrics, "match_via_slug", lambda *a: None)
    rec = fetch_lyrics.fetch_one(song, "", retry_failed=True)
    assert rec["status"] == "no_confident_match"
    path.write_text(json.dumps({
        "song_id": song.song_id, "has_lyrics": True, "lyrics": "synthetic input",
    }))
    rec = fetch_lyrics.fetch_one(song, "", retry_failed=True)
    assert rec["has_lyrics"] and "lyrics" not in rec


def test_failed_http_cache_can_be_refreshed(monkeypatch, tmp_path):
    from music_tastes import http

    monkeypatch.setattr(http, "CACHE", tmp_path)
    monkeypatch.setattr(http, "_throttle", lambda host: None)
    url = "https://example.test/missing"
    path = http._cache_path(url, "test")
    path.write_text(json.dumps({"status": 404, "text": "", "retrieved_at": "old"}))
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, text="now available", ok=True
    ))
    assert http.get(url, namespace="test").status == 404
    token = REFRESH_ERRORS.set(True)
    try:
        assert http.get(url, namespace="test").text == "now available"
    finally:
        REFRESH_ERRORS.reset(token)


def test_nli_signature_rejects_changed_text_config_and_errors(monkeypatch):
    text = "synthetic classifier input"
    rec = {"signature": stance_nli.cache_signature(text), "p_relationship_doc": 0.8}
    rec.update({f"p_{key}_max": 0.2 for key in stance_nli.CHUNK_KEYS})
    assert stance_nli.valid_record(rec, text)
    assert not stance_nli.valid_record(rec, text + " changed")
    assert not stance_nli.valid_record({**rec, "error": "failed"}, text)
    monkeypatch.setattr(stance_nli, "DOC_WORDS", 500)
    assert not stance_nli.valid_record(rec, text)
