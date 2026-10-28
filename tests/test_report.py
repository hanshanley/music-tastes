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

        captured = []
        save_fig = report.save_fig

        def capture(fig, *args, **kwargs):
            captured.append(fig)
            return save_fig(fig, *args, **kwargs)

        monkeypatch.setattr(report, "save_fig", capture)
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
        assert (tmp_path / "lyrical-emotion-dimensions.png").is_file()
        for ax in captured[0].axes:
            assert [line.get_linestyle() for line in ax.lines] == ["--", "-"]
            assert "0-1" in ax.get_ylabel()


class TestReportArtifacts:
    @pytest.mark.parametrize(
        "first,last,years,year_labels,decades,decade_labels",
        [
            ("1958-08-04", "2026-10-31", [1958, 1959, 2025, 2026],
             ["1958*", "1959", "2025", "2026*"], [1950, 1960, 2020],
             ["1950s*", "1960s", "2020s*"]),
            ("1960-01-02", "1979-12-29", [1960, 1979],
             ["1960", "1979"], [1960, 1970], ["1960s", "1970s"]),
            ("1961-01-07", "2025-12-27", [1961, 2025],
             ["1961", "2025"], [1960, 2020], ["1960s*", "2020s*"]),
        ],
    )
    def test_partial_labels_use_actual_calendar_boundaries(
        self, monkeypatch, tmp_path, first, last, years, year_labels, decades, decade_labels,
    ):
        from music_tastes import vizstyle

        monkeypatch.setattr(vizstyle, "INTERIM", tmp_path)
        (tmp_path / "chart_integrity.json").write_text(json.dumps({
            "chart_from": first, "chart_through": last,
        }))
        assert vizstyle.period_labels(years) == year_labels
        assert vizstyle.period_labels(decades, decade=True) == decade_labels

    def test_standalone_chart_identifies_partial_years_and_model_limitations(self, monkeypatch, tmp_path):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import pandas as pd
        from music_tastes import vizstyle

        monkeypatch.setattr(vizstyle, "INTERIM", tmp_path)
        (tmp_path / "chart_integrity.json").write_text(json.dumps({
            "chart_from": "1958-08-04", "chart_through": "2026-10-31",
        }))
        captured = []
        save_fig = report.save_fig

        def capture(fig, *args, **kwargs):
            captured.append(fig)
            return save_fig(fig, *args, **kwargs)

        monkeypatch.setattr(report, "save_fig", capture)
        frame = pd.DataFrame({
            "metric": ["relationship_share"] * 3, "variant": ["weighted_all"] * 3,
            "year": [1958, 2025, 2026], "mean": [.5, .6, .7],
            "ci_lo": [.4, .5, .6], "ci_hi": [.6, .7, .8],
        })
        report._plot_metric(frame, "relationship_share", "", tmp_path / "chart.png")
        fig, ax = captured[0], captured[0].axes[0]
        labels = [text.get_text() for text in ax.get_xticklabels()]
        assert "1958*" in labels and "2026*" in labels
        note = fig.texts[0]
        assert "2026-10-31" in note.get_text()
        assert "independent human validation is pending" in note.get_text()
        assert note.get_fontsize() >= 10
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        assert note.get_window_extent(renderer).y1 < ax.xaxis.label.get_window_extent(renderer).y0
        plt.close(fig)

    def test_headline_chart_keeps_primary_values_and_omits_balanced_overlay(self, monkeypatch, tmp_path):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        captured = []
        monkeypatch.setattr(report, "save_fig", lambda fig, *a, **k: captured.append(fig))
        frame = pd.DataFrame({
            "metric": ["relationship_share"] * 4,
            "variant": ["weighted_all"] * 2 + ["weighted_complete_case"] * 2,
            "year": [1960, 1961] * 2, "mean": [.5, .6, .8, .9],
            "ci_lo": [.4, .5, .7, .8], "ci_hi": [.6, .7, .9, 1.0],
        })
        assert report._plot_metric(frame, "relationship_share", "Internal label", tmp_path / "chart.png")
        ax = captured[0].axes[0]
        assert len(ax.lines) == 1
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [.5, .6])
        assert ax.yaxis.get_major_formatter()(.5) == "50%"
        assert ax.get_title(loc="left") == "Relationship songs"
        assert ax.get_legend() is None
        assert "balanced" not in " ".join(text.get_text() for text in ax.texts).lower()
        captured[0].set_dpi(200)
        captured[0].tight_layout()
        captured[0].canvas.draw()
        renderer = captured[0].canvas.get_renderer()
        title = ax._left_title.get_window_extent(renderer)
        subtitle = ax.texts[0].get_window_extent(renderer)
        assert title.y0 >= subtitle.y1 + 5 / 72 * 200
        plt.close(captured[0])
        assert not report._plot_metric(
            frame[frame.variant == "weighted_complete_case"], "relationship_share",
            "Internal label", tmp_path / "missing-primary.png",
        )

    def test_stance_panels_preserve_shares_without_an_overlapping_legend(self, monkeypatch, tmp_path):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from music_tastes import stance_composition

        captured = []
        monkeypatch.setattr(stance_composition, "save_fig", lambda fig, *a, **k: captured.append(fig))
        shares = pd.DataFrame({
            "decade": [2000, 2010],
            **{name: [.1 + i * .02, .2 + i * .02]
               for i, name in enumerate(stance_composition.STANCES)},
        })
        assert stance_composition._plot_independent(shares, tmp_path / "themes.png")
        assert len(captured[0].axes) == 6
        for ax, name in zip(captured[0].axes, stance_composition.STANCES):
            assert len(ax.lines) == 1 and ax.get_legend() is None
            np.testing.assert_allclose(ax.lines[0].get_ydata(), shares[name] * 100)
        plt.close(captured[0])

    def test_report_exports_both_publication_backgrounds(self, monkeypatch, tmp_path):
        import pandas as pd

        derived, reports = tmp_path / "derived", tmp_path / "reports"
        derived.mkdir()
        reports.mkdir()
        monkeypatch.setattr(report, "DERIVED", derived)
        monkeypatch.setattr(report, "INTERIM", tmp_path)
        monkeypatch.setattr(report, "REPORTS", reports)
        monkeypatch.setattr(report, "FIGURES", reports / "figures")
        monkeypatch.setattr(report, "METRICS", {"relationship_share": METRICS["relationship_share"]})
        frame = pd.DataFrame({
            "metric": ["relationship_share"] * 2, "variant": ["weighted_all"] * 2,
            "year": [1960, 1961], "decade": [1960, 1970],
            "mean": [.5, .6], "ci_lo": [.4, .5], "ci_hi": [.6, .7],
        })
        frame.to_parquet(derived / "yearly_series.parquet")
        frame.to_csv(reports / "decade_series.csv", index=False)
        pd.DataFrame({
            "debut_year": [1960, 1961], "coverage_lyrics": [.8, .9],
            "coverage_exposure": [.9, .95],
        }).to_csv(reports / "coverage_by_year.csv", index=False)
        report.run()
        originals = sorted((reports / "figures").glob("*.png"))
        assert len(originals) == 3
        assert {p.name for p in originals} == {
            p.name for p in (reports / "figures" / "page_background").glob("*.png")
        }

    def test_page_exports_preserve_foreground_transparency_dimensions_and_originals(self, tmp_path):
        import numpy as np
        from PIL import Image
        from music_tastes.page_plots import export

        figures = tmp_path / "figures"
        source = figures / "nested" / "chart.png"
        source.parent.mkdir(parents=True)
        pixels = np.array([
            [(247, 245, 240, 255), (200, 90, 61, 255)],
            [(247, 245, 240, 0), (247, 245, 240, 128)],
        ], dtype=np.uint8)
        Image.fromarray(pixels).save(source, dpi=(200, 200))
        original = source.read_bytes()
        destination = figures / "page_background" / "nested" / "chart.png"
        assert export(figures) == [destination]
        with Image.open(destination) as image:
            actual = np.array(image)
            assert image.size == (2, 2)
            with Image.open(source) as before:
                assert image.info["dpi"] == before.info["dpi"]
        np.testing.assert_array_equal(actual[0, 0], [245, 245, 245, 255])
        np.testing.assert_array_equal(actual[0, 1], pixels[0, 1])
        np.testing.assert_array_equal(actual[1], pixels[1])
        assert source.read_bytes() == original
        stale = destination.parent / "stale.png"
        stale.write_bytes(original)
        assert export(figures) == [destination]
        assert not stale.exists()
        assert not (figures / "page_background" / "page_background").exists()

    def test_page_exports_reject_missing_or_unexpected_originals(self, tmp_path):
        from PIL import Image
        from music_tastes.page_plots import export

        with pytest.raises(ValueError, match="No original plot PNGs"):
            export(tmp_path)
        Image.new("RGB", (2, 2), "white").save(tmp_path / "chart.png")
        with pytest.raises(ValueError, match="expected chart background"):
            export(tmp_path)

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
