"""Statistical checks on synthetic annual series, without private lyrics."""

import numpy as np
import pandas as pd
import pytest

from music_tastes.vad_statistics import (
    _block_slope_errors, _correct_families, analyze_series, trend_tests,
)


def annual_series(slope=.02):
    years = np.arange(1958, 2027)
    values = .5 + slope * (years - years.mean()) / 10
    values += np.random.default_rng(42).normal(0, .005, len(years))
    return pd.DataFrame({"year": years, "mean": values, "n": 300})


def test_known_linear_change_has_per_decade_units_and_reproducible_block_intervals():
    frame = annual_series()
    rows = trend_tests(frame, n_boot=999)
    assert rows == trend_tests(frame.sample(frac=1, random_state=4), n_boot=999)
    for row in rows:
        assert row["slope_per_decade"] == pytest.approx(.02, abs=.001)
        assert row["ci95_lo"] < .02 < row["ci95_hi"]
        assert row["p_value"] < .01


def test_constant_series_is_not_a_significant_trend():
    frame = annual_series().assign(mean=.5)
    for row in trend_tests(frame, n_boot=99):
        assert row["slope_per_decade"] == 0
        assert row["ci95_lo"] == row["ci95_hi"] == 0
        assert row["p_value"] == 1


def test_serial_dependence_inflates_uncertainty_relative_to_iid():
    rng = np.random.default_rng(17)
    residuals = np.zeros(150)
    for i in range(1, len(residuals)):
        residuals[i] = .9 * residuals[i - 1] + rng.normal(0, .01)
    rows = trend_tests(pd.DataFrame({"year": np.arange(1900, 2050), "mean": residuals}), n_boot=999)
    for row in rows:
        if row["method"] == "newey_west" and row["window_years"] == 5:
            assert row["standard_error"] > 1.5 * row["iid_standard_error"]


@pytest.mark.parametrize("change", ["gap", "duplicate", "missing", "short"])
def test_invalid_annual_design_is_rejected(change):
    frame = annual_series()
    if change == "gap":
        frame = frame.drop(index=20)
    elif change == "duplicate":
        frame.loc[20, "year"] = frame.loc[19, "year"]
    elif change == "missing":
        frame.loc[20, "mean"] = np.nan
    else:
        frame = frame.head(20)
    with pytest.raises(ValueError, match="consecutive annual means"):
        trend_tests(frame, n_boot=99)


def test_block_bootstrap_uses_wrapping_contiguous_residuals_and_fixed_years(monkeypatch):
    class FixedStarts:
        def integers(self, low, high, size):
            return np.array([[3, 1], [0, 2]])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedStarts())
    residuals = np.array([1., 2., -1., -2.])
    x = np.array([-1.5, -.5, .5, 1.5])
    got = _block_slope_errors(residuals, x, 2, 2)
    expected = np.array([residuals[[3, 0, 1, 2]] @ x, residuals[[0, 1, 2, 3]] @ x]) / (x @ x)
    np.testing.assert_allclose(got, expected)


def test_holm_primary_family_is_not_confused_with_all_comparisons():
    rows = [
        {"method": "newey_west", "window_years": 5, "comparison": "level",
         "is_primary": True, "p_value": p}
        for p in [.01, .04, .03]
    ]
    rows.append({"method": "newey_west", "window_years": 5, "comparison": "level",
                 "is_primary": False, "p_value": .9})
    rows.append({"method": "circular_block", "window_years": 3, "comparison": "level",
                 "is_primary": False, "p_value": .99})
    _correct_families(rows)
    assert [r["p_holm_primary"] for r in rows[:3]] == pytest.approx([.03, .06, .06])
    assert [r["p_holm_all"] for r in rows[:3]] == pytest.approx([.04, .09, .09])
    assert [r["p_holm_entire_battery"] for r in rows[:3]] == pytest.approx([.05, .12, .12])
    assert "p_holm_primary" not in rows[3]


def comparison_series():
    rows = []
    for dimension in ("valence", "arousal", "dominance"):
        for variant in ("v1_tokens", "v21_tokens"):
            frame = annual_series(.02 if variant == "v1_tokens" else .01)
            rows.append(frame.assign(sample="matched_all_variants", time_basis="debut_year",
                                     weighting="exposure", variant=variant, dimension=dimension))
    return pd.concat(rows, ignore_index=True)


def test_boundary_years_and_paired_version_differences_use_identical_populations():
    result = analyze_series(comparison_series(), "1958-08-04", "2026-10-31", n_boot=99)
    assert result["n_comparisons"] == 9
    assert result["n_tests"] == 54
    for row in result["tests"]:
        assert row["n_years"] == 67
        assert (row["year_min"], row["year_max"]) == (1959, 2025)
        if row["comparison"] == "v21_minus_v1":
            assert row["slope_per_decade"] == pytest.approx(-.01)
            assert "p_holm_paired" in row
        if row["is_primary"]:
            assert row["variant"] == "v21_tokens"
            assert "p_holm_primary" in row


def test_version_contrast_rejects_mismatched_yearly_sample_sizes():
    frame = comparison_series()
    frame.loc[(frame["variant"] == "v21_tokens") & (frame["year"] == 2000), "n"] = 299
    with pytest.raises(ValueError, match="identical annual matched-song populations"):
        analyze_series(frame, "1958-08-04", "2026-10-31", n_boot=99)
