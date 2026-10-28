"""Synthetic norms only: no redistributed NRC entries or lyric excerpts."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from music_tastes import lexicons
from music_tastes.vad_robustness import make_lookups, score_lyrics


def archive_bytes(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt", text)
        archive.writestr("NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt",
                         "not the canonical table")
    return buffer.getvalue()


def test_v21_parser_preserves_native_scale_and_literal_na_terms():
    frame = lexicons.parse_vad_v21(archive_bytes(
        "term\tvalence\tarousal\tdominance\nnull\t-0.8\t0.2\t0.4\n"
    ))
    assert frame["term"].tolist() == ["null"]
    assert frame["valence"].tolist() == [-.8]


@pytest.mark.parametrize("text", [
    "term\tvalence\tarousal\tdominance\nalpha\t1.1\t0\t0\n",
    "term\tvalence\tarousal\tdominance\nalpha\tnan\t0\t0\n",
    "term\tvalence\tarousal\tdominance\nalpha\t0\t0\t0\nalpha\t0\t0\t0\n",
    "word\tvalence\tarousal\tdominance\nalpha\t0\t0\t0\n",
])
def test_v21_parser_rejects_bad_scores_duplicates_and_wrong_version(text):
    with pytest.raises(ValueError):
        lexicons.parse_vad_v21(archive_bytes(text))


def test_v21_import_is_separate_and_detects_cache_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(lexicons, "LEX_DIR", tmp_path)
    baseline = tmp_path / "nrc_vad.parquet"
    baseline.write_bytes(b"unchanged baseline")
    source = tmp_path / "input.zip"
    source.write_bytes(archive_bytes(
        "term\tvalence\tarousal\tdominance\nalpha\t-0.8\t0.2\t0.4\n"
    ))
    imported = lexicons.load_vad_v21(source, offline=True)
    pd.testing.assert_frame_equal(imported, lexicons.load_vad_v21(offline=True))
    assert baseline.read_bytes() == b"unchanged baseline"
    (tmp_path / "nrc_vad_v21.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cache changed"):
        lexicons.load_vad_v21(offline=True)


@pytest.fixture
def lookups():
    baseline = pd.DataFrame({
        "word": ["bright", "bleak"], "valence": [.9, .1],
        "arousal": [.6, .2], "dominance": [.8, .2],
    })
    expanded = pd.DataFrame([
        ("bright", .8, .2, .6), ("bleak", -.8, -.6, -.6),
        ("not", 0, 0, 0), ("very", 0, 0, 0),
        ("not bright", -.6, -.4, -.4), ("not very bright", -.9, -.8, -.8),
        ("nearneutral", .1, 0, 0), ("boundary", .333, 0, 0),
    ], columns=["term", "valence", "arousal", "dominance"])
    return make_lookups(baseline, expanded)


def test_versions_use_same_scale_and_unique_words_remove_repetition(lookups):
    features = score_lyrics("bright bright bleak", *lookups)
    assert features["v1_tokens_valence"] == pytest.approx((.9 * 2 + .1) / 3)
    assert features["v21_tokens_valence"] == pytest.approx(features["v1_tokens_valence"])
    assert features["v21_types_valence"] == pytest.approx(.5)
    assert features["v21_types_coverage"] == 1


def test_phrases_are_longest_nonoverlapping_units_without_double_counting(lookups):
    features = score_lyrics("not very bright bright", *lookups)
    assert features["phrase_matches"] == 1
    assert features["v21_phrases_matched_tokens"] == 4
    assert features["v21_phrases_valence"] == pytest.approx((.05 + .9) / 2)


@pytest.mark.parametrize("separator", ["\n", ". ", " [Chorus] ", "; "])
def test_phrases_never_cross_structural_boundaries(lookups, separator):
    features = score_lyrics(f"not{separator}bright", *lookups)
    assert features["phrase_matches"] == 0
    assert features["v21_phrases_valence"] == pytest.approx(.7)


def test_unknown_and_nonpolar_words_stay_missing_not_neutral(lookups):
    unknown = score_lyrics("unrecognized", *lookups)
    assert np.isnan(unknown["v21_tokens_valence"])
    assert unknown["v21_tokens_coverage"] == 0
    neutral = score_lyrics("nearneutral", *lookups)
    assert neutral["v21_tokens_valence"] == pytest.approx(.55)
    assert np.isnan(neutral["v21_polar_0167_valence"])
    assert neutral["v21_polar_0333_matched_tokens"] == 0


def test_polar_filter_includes_exact_threshold(lookups):
    features = score_lyrics("boundary nearneutral", *lookups)
    assert features["v21_polar_0333_matched_tokens"] == 1
    assert features["v21_polar_0333_valence"] == pytest.approx(.6665)


def test_shared_vocabulary_separates_new_words_from_revised_ratings():
    v1 = pd.DataFrame({
        "word": ["shared", "oldonly"], "valence": [.9, .8],
        "arousal": [.5, .5], "dominance": [.5, .5],
    })
    v21 = pd.DataFrame([
        ("shared", .2, 0, 0), ("newonly", -.8, 0, 0),
    ], columns=["term", "valence", "arousal", "dominance"])
    features = score_lyrics("shared oldonly newonly", *make_lookups(v1, v21))
    assert features["v1_shared_tokens_valence"] == pytest.approx(.9)
    assert features["v21_shared_tokens_valence"] == pytest.approx(.6)
    assert features["v21_tokens_valence"] == pytest.approx(.35)
    assert features["v1_shared_tokens_matched_tokens"] == features["v21_shared_tokens_matched_tokens"] == 1


def test_vad_cli_accepts_local_archive_and_offline(monkeypatch, tmp_path):
    from music_tastes import cli, vad_robustness

    calls = []
    monkeypatch.setattr(vad_robustness, "run", lambda **kwargs: calls.append(kwargs))
    path = tmp_path / "source.zip"
    cli.main(["vad-robustness", "--vad-archive", str(path), "--offline"])
    assert calls == [{"archive": path, "offline": True}]
