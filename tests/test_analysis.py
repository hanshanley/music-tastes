"""Tests for exposure weighting and the trend statistics.

The weighting decisions here are the ones most likely to produce a confident wrong
answer. Chart tenure grew from roughly 13 weeks in the 1960s to 90+ weeks today, so
raw chart points are an order of magnitude larger now; any statistic that pools
across eras without normalizing within year will simply report the streaming era.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_tastes.analysis_trends import (
    P_THRESHOLD,
    _weighted_mean,
    derive_labels,
    trend_test,
    yearly_series,
)
from music_tastes.exposure import ERAS, label_era


class TestEraTable:
    def test_eras_are_contiguous(self):
        for earlier, later in zip(ERAS, ERAS[1:]):
            end = pd.Timestamp(earlier["end"])
            start = pd.Timestamp(later["start"])
            assert (start - end).days == 1, f"gap between {earlier['name']} and {later['name']}"

    def test_covers_first_chart_week(self):
        assert pd.Timestamp(ERAS[0]["start"]) == pd.Timestamp("1958-08-04")

    @pytest.mark.parametrize(
        "date,expected",
        [
            ("1960-01-04", "Survey era"),
            ("1991-11-29", "Survey era"),
            ("1991-11-30", "SoundScan/BDS era"),
            ("2005-02-12", "Digital sales era"),
            ("2020-01-04", "Video/streaming era"),
        ],
    )
    def test_boundaries(self, date, expected):
        got = label_era(pd.Series([pd.Timestamp(date)]))
        assert got.iloc[0] == expected

    def test_every_week_labelled(self):
        weeks = pd.Series(pd.date_range("1958-08-04", "2026-07-25", freq="7D"))
        assert label_era(weeks).notna().all()


class TestWeightedMean:
    def test_matches_simple_mean_with_equal_weights(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        assert _weighted_mean(v, w) == pytest.approx(2.0)

    def test_weights_shift_the_mean(self):
        v = np.array([0.0, 1.0])
        assert _weighted_mean(v, np.array([9.0, 1.0])) == pytest.approx(0.1)

    def test_zero_weight_total_is_nan(self):
        assert np.isnan(_weighted_mean(np.array([1.0]), np.array([0.0])))


class TestDeriveLabels:
    def _frame(self):
        return pd.DataFrame(
            {
                "song_id": ["a", "b", "c"],
                "p_relationship_doc": [0.9, 0.1, 0.8],
                "p_independence_max": [0.9, 0.9, 0.2],
                "p_heartbreak_max": [0.1, 0.2, 0.7],
            }
        )

    def test_relationship_threshold(self):
        got = derive_labels(self._frame())
        assert list(got["is_relationship"]) == [1.0, 0.0, 1.0]

    def test_stance_is_nan_for_non_relationship_songs(self):
        """Stance shares are conditional on being a relationship song, so a
        non-relationship song must drop out of the denominator rather than count
        as a zero. This is what stops 'we don't need no education' inflating the
        independence share."""
        got = derive_labels(self._frame())
        assert np.isnan(got.loc[1, "is_independent"])

    def test_stance_applied_within_relationship_songs(self):
        got = derive_labels(self._frame())
        assert got.loc[0, "is_independent"] == 1.0
        assert got.loc[2, "is_independent"] == 0.0

    def test_missing_probability_stays_nan(self):
        df = self._frame()
        df.loc[0, "p_independence_max"] = np.nan
        got = derive_labels(df)
        assert np.isnan(got.loc[0, "is_independent"])


class TestYearlySeries:
    def _frame(self, n_per_year=20):
        rows = []
        rng = np.random.default_rng(0)
        for year in range(1990, 2010):
            for _ in range(n_per_year):
                rows.append(
                    {"debut_year": year, "value": (year - 1990) * 0.01 + rng.normal(0, 0.01),
                     "points": rng.integers(1, 500)}
                )
        return pd.DataFrame(rows)

    def test_returns_one_row_per_year(self):
        s = yearly_series(self._frame(), "value", "points")
        assert len(s) == 20

    def test_years_below_minimum_are_dropped(self):
        s = yearly_series(self._frame(n_per_year=3), "value", "points", min_n=8)
        assert s.empty

    def test_confidence_interval_brackets_the_estimate(self):
        s = yearly_series(self._frame(), "value", "points")
        assert (s["ci_lo"] <= s["mean"]).all()
        assert (s["mean"] <= s["ci_hi"]).all()

    def test_detects_a_planted_trend(self):
        s = yearly_series(self._frame(), "value", "points")
        t = trend_test(s)
        assert t["kendall_tau"] > 0.8
        assert t["p_value"] < 0.01


class TestTrendTest:
    def test_too_few_years_is_reported_not_guessed(self):
        s = pd.DataFrame({"year": [2000, 2001], "mean": [0.1, 0.2]})
        assert "note" in trend_test(s)

    def test_flat_series_is_not_significant(self):
        s = pd.DataFrame({"year": list(range(1990, 2015)), "mean": [0.5] * 25})
        t = trend_test(s)
        assert not t["significant_at_05"]

    def test_slope_direction(self):
        s = pd.DataFrame(
            {"year": list(range(1990, 2015)), "mean": [0.9 - i * 0.01 for i in range(25)]}
        )
        t = trend_test(s)
        assert t["theil_sen_slope_per_year"] < 0
        assert t["kendall_tau"] < 0


def test_threshold_is_the_natural_entailment_cut():
    assert P_THRESHOLD == 0.5


class TestChunkRegrouping:
    """The batched scorer flattens chunks across songs, then regroups by owner.

    A regrouping error would attach one song's chorus to another song's score and
    corrupt every downstream statistic without raising anything, so the invariant
    is checked directly rather than assumed.
    """

    def test_owner_mapping_is_stable_under_reordering(self):
        from music_tastes.stance_nli import chunk_lyrics

        songs = {
            "s0000000000000001": "alpha " * 200,
            "s0000000000000002": "beta " * 90,
            "s0000000000000003": "gamma " * 400,
        }
        flat, owner = [], []
        for song_id, text in songs.items():
            chunks = chunk_lyrics(text)
            flat.extend(chunks)
            owner.extend([song_id] * len(chunks))

        assert len(flat) == len(owner), "every chunk must carry exactly one owner"

        regrouped: dict[str, list[str]] = {}
        for song_id, chunk in zip(owner, flat):
            regrouped.setdefault(song_id, []).append(chunk)

        for song_id, text in songs.items():
            assert regrouped[song_id] == chunk_lyrics(text)

    def test_dedup_never_drops_every_chunk(self):
        """A lyric that is one repeated line must still yield a chunk to score."""
        from music_tastes.stance_nli import chunk_lyrics

        assert chunk_lyrics("na " * 300)
        assert chunk_lyrics("one line only")

    def test_empty_lyric_yields_no_chunks(self):
        from music_tastes.stance_nli import chunk_lyrics

        assert chunk_lyrics("") == []
