"""Tests for report rendering.

These exist because of a real published-number bug: the decade tables formatted
every metric as a percentage, so tempo appeared in findings.md as "11698.5%" when
the value was 117.0 BPM, and VADER's signed score appeared as "117.3%" when it was
1.17. Shares read correctly on a percent scale; nothing else does.
"""

from __future__ import annotations

import json

import pytest

from music_tastes import report
from music_tastes.analysis_trends import METRICS


class TestMetricUnits:
    @pytest.mark.parametrize(
        "metric",
        [
            "relationship_share",
            "independence_share",
            "heartbreak_share",
            "devotion_share",
            "minor_key_share",
        ],
    )
    def test_shares_render_as_percent(self, metric):
        assert report._metric_unit(metric).fmt(0.193) == "19.3%"

    def test_bpm_renders_as_tempo(self):
        got = report._metric_unit("bpm").fmt(116.985)
        assert got == "117.0"
        assert "%" not in got

    def test_word_count_renders_as_count(self):
        got = report._metric_unit("lyric_length").fmt(430.2)
        assert got == "430"
        assert "%" not in got

    def test_vader_is_not_a_percentage(self):
        """VADER emits a signed score, not a proportion."""
        got = report._metric_unit("vader_valence").fmt(1.1727)
        assert "%" not in got
        assert got.startswith("1.")

    def test_bounded_valence_keeps_precision(self):
        # NRC VAD valence moves in the third decimal across the whole period, so
        # one decimal place would flatten the entire trend to a constant.
        assert report._metric_unit("lyric_valence").fmt(0.6182) == "0.618"

    def test_missing_values_render_as_dash(self):
        assert report._metric_unit("bpm").fmt(float("nan")) == "—"
        assert report._metric_unit("relationship_share").fmt(None) == "—"

    def test_every_non_share_metric_declares_its_unit(self):
        """Any metric whose values are not in [0, 1] must opt out of percent.

        This is the guard that would have caught the original bug: adding a new
        non-share metric without declaring a unit fails here rather than silently
        publishing a nonsense percentage.
        """
        undeclared = [
            m
            for m in METRICS
            if not m.endswith("_share") and m not in report._UNITS
        ]
        assert not undeclared, (
            f"metrics rendered on the default percent scale without declaring a "
            f"unit: {undeclared}"
        )


class TestDerivedClaims:
    """The prose figures must come from data, not string literals."""

    def test_no_hardcoded_stale_statistics_in_emitted_strings(self):
        """Stale figures must not survive in strings that reach the report.

        Docstrings legitimately quote past wrong values to explain a fix, so they
        are excluded; only string literals that can be emitted are checked.
        """
        import ast

        tree = ast.parse(open(report.__file__).read())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.add(doc)

        emitted = [
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value not in docstrings
        ]
        blob = "\n".join(emitted)
        for stale in ("65–76%", "1.8e-09", "seven decades", "11698.5%"):
            assert stale not in blob, (
                f"stale hardcoded figure still reachable in an emitted string: {stale}"
            )

    def test_missed_median_peak_is_an_integer_rank(self, monkeypatch, tmp_path):
        import pandas as pd

        monkeypatch.setattr(report, "DERIVED", tmp_path)
        pd.DataFrame({"song_id": ["a", "b"], "has_lyrics": [True, False]}).to_parquet(
            tmp_path / "lyrics_index.parquet"
        )
        pd.DataFrame({"song_id": ["a", "b"], "peak_rank": [1, 59]}).to_parquet(
            tmp_path / "songs_weighted.parquet"
        )
        got = report._missed_median_peak()
        assert isinstance(got, int)
        assert 1 <= got <= 100

    def test_share_range_text_mentions_decades(self):
        assert "decade" in report._share_range_text()

    def test_vad_appendix_renders_current_estimate_and_comparison_chart(self, monkeypatch, tmp_path):
        import pandas as pd

        monkeypatch.setattr(report, "REPORTS", tmp_path)
        monkeypatch.setattr(report, "FIGURES", tmp_path)
        result = {
            "n_matched_all_variants": 60, "n_eligible": 64,
            "results": [{
                "sample": "matched_all_variants", "time_basis": "debut_year",
                "weighting": "exposure", "dimension": "valence", "variant": "v21_tokens",
                "trend": {"change_per_decade": .0123, "kendall_tau": .4},
            }],
        }
        (tmp_path / "nrc_vad_robustness.json").write_text(json.dumps(result))
        (tmp_path / "nrc_vad_statistics.json").write_text(json.dumps({
            "first_complete_year": 1959, "last_complete_year": 2025, "n_comparisons": 67,
            "tests": [{
                "is_primary": True, "method": "newey_west", "window_years": 5,
                "dimension": "valence", "slope_per_decade": -.004,
                "ci95_lo": -.006, "ci95_hi": -.002, "p_holm_primary": .003,
            }],
        }))
        pd.DataFrame([
            {"sample": "matched_all_variants", "time_basis": "debut_year",
             "weighting": "exposure", "dimension": dimension, "variant": variant,
             "year": year, "mean": .5, "ci_lo": .4, "ci_hi": .6}
            for dimension in ("valence", "arousal", "dominance")
            for variant in ("v1_tokens", "v21_tokens") for year in (2000, 2001)
        ]).to_csv(tmp_path / "nrc_vad_series.csv", index=False)
        text = "\n".join(report._vad_section())
        assert "| v21_tokens | +0.0123 | +0.40 |" in text
        assert "**60 identical songs**" in text and "**64 eligible lyrics**" in text
        assert "validated sadness classifier" in text
        assert "| Valence | -0.00400 | [-0.00600, -0.00200] | 0.003 |" in text
        assert "not establish a flat trend" in text
        assert "nrc_vad_statistics.csv" in text
        assert (tmp_path / "nrc_vad_robustness.png").is_file()


class TestReportArtifacts:
    def test_findings_has_no_implausible_percentages(self):
        """No published percentage should exceed 100% — that means a unit error."""
        path = report.REPORTS / "findings.md"
        if not path.exists():
            pytest.skip("findings.md not generated yet")
        import re

        bad = [
            m
            for m in re.findall(r"\|\s*([0-9]+\.[0-9])%", path.read_text())
            if float(m) > 100.0
        ]
        assert not bad, f"percentages above 100% in findings.md: {bad[:5]}"

    def test_summary_figures_match_the_json_they_come_from(self):
        vpath = report.REPORTS / "validity.json"
        fpath = report.REPORTS / "findings.md"
        if not (vpath.exists() and fpath.exists()):
            pytest.skip("reports not generated yet")
        agg = json.loads(vpath.read_text()).get("aggregation_bias", {})
        if "chunk_adjusted_p" not in agg:
            pytest.skip("aggregation bias not computed yet")
        assert f"p={agg['chunk_adjusted_p']:.2g}" in fpath.read_text()
