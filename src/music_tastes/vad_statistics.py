"""Dependence-aware inference for annual NRC VAD means, not song-mood validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .cache import fingerprint
from .paths import atomic_json

WINDOWS = (3, 5, 10)
N_BOOT = 9999
SEED = 20261028
KEYS = ["sample", "time_basis", "variant", "dimension", "weighting"]


def _block_slope_errors(residuals, centered_years, block_length, n_boot):
    rng = np.random.default_rng(SEED)
    n = len(residuals)
    starts = rng.integers(0, n, size=(n_boot, (n + block_length - 1) // block_length))
    indices = ((starts[..., None] + np.arange(block_length)) % n).reshape(n_boot, -1)[:, :n]
    return residuals[indices] @ centered_years / (centered_years @ centered_years)


def trend_tests(series: pd.DataFrame, n_boot: int = N_BOOT) -> list[dict]:
    ordered = series.sort_values("year")
    years = ordered["year"].to_numpy(dtype=float)
    values = ordered["mean"].to_numpy(dtype=float)
    if (len(years) < 30 or not np.isfinite(values).all()
            or not np.isfinite(years).all() or not (np.diff(years) == 1).all()):
        raise ValueError("Trend inference requires at least 30 finite, consecutive annual means")
    if n_boot < 99:
        raise ValueError("Use at least 99 bootstrap replicates")
    x = (years - years.mean()) / 10
    model = sm.OLS(values, sm.add_constant(x)).fit()
    constant = np.ptp(values) == 0
    slope = 0.0 if constant else float(model.params[1])
    residuals = np.zeros(len(x)) if constant else model.resid - model.resid.mean()
    acf = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if not constant else 0.0
    base = {
        "n_years": len(years), "year_min": int(years.min()), "year_max": int(years.max()),
        "slope_per_decade": slope, "residual_lag1_correlation": acf,
        "iid_standard_error": 0.0 if constant else float(model.bse[1]),
    }
    tests = []
    for window in WINDOWS:
        robust = model.get_robustcov_results(
            cov_type="HAC", maxlags=window, kernel="bartlett", use_correction=True, use_t=True,
        )
        lo, hi = robust.conf_int()[1] if not constant else (0.0, 0.0)
        tests.append({
            **base, "method": "newey_west", "window_years": window,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "standard_error": 0.0 if constant else float(robust.bse[1]),
            "p_value": 1.0 if constant else float(robust.pvalues[1]),
        })
        errors = _block_slope_errors(residuals, x, window, n_boot)
        qlo, qhi = np.quantile(errors, [.025, .975])
        tests.append({
            **base, "method": "circular_block", "window_years": window,
            # Basic intervals invert the centered bootstrap estimation errors.
            "ci95_lo": float(slope - qhi), "ci95_hi": float(slope - qlo),
            "standard_error": float(errors.std(ddof=1)),
            "p_value": float((1 + np.count_nonzero(np.abs(errors) >= abs(slope))) / (n_boot + 1)),
        })
    return tests


def _correct_families(tests: list[dict]) -> None:
    entire = multipletests([row["p_value"] for row in tests], method="holm")[1]
    for row, p in zip(tests, entire):
        row["p_holm_entire_battery"] = float(p)
        row["reject_entire_battery_05"] = bool(p < .05)
    for method in ("newey_west", "circular_block"):
        for window in WINDOWS:
            group = [row for row in tests if row["method"] == method and row["window_years"] == window]
            for name, subset in (
                ("all", group), ("primary", [row for row in group if row["is_primary"]]),
                ("paired", [row for row in group if row["comparison"] == "v21_minus_v1"]),
            ):
                if not subset:
                    continue
                corrected = multipletests([row["p_value"] for row in subset], method="holm")[1]
                for row, p in zip(subset, corrected):
                    row[f"p_holm_{name}"] = float(p)
                    row[f"reject_{name}_05"] = bool(p < .05)


def analyze_series(series: pd.DataFrame, chart_from: str, chart_through: str,
                   n_boot: int = N_BOOT) -> dict:
    first, last = pd.Timestamp(chart_from), pd.Timestamp(chart_through)
    first_complete = first.year + int(first.day_of_year > 7)
    last_complete = last.year - int((last + pd.Timedelta(days=7)).year == last.year)
    complete = series[series["year"].between(first_complete, last_complete)].copy()
    tests = []
    for key, group in complete.groupby(KEYS, sort=True):
        labels = dict(zip(KEYS, key))
        primary = (
            labels["sample"] == "matched_all_variants" and labels["time_basis"] == "debut_year"
            and labels["variant"] == "v21_tokens" and labels["weighting"] == "exposure"
        )
        tests.extend({**labels, "comparison": "level", "is_primary": primary, **row}
                     for row in trend_tests(group, n_boot))
    paired = complete[
        (complete["sample"] == "matched_all_variants")
        & (complete["time_basis"] == "debut_year") & (complete["weighting"] == "exposure")
    ]
    for dimension in ("valence", "arousal", "dominance"):
        part = paired[paired["dimension"] == dimension]
        old = part[part["variant"] == "v1_tokens"][["year", "mean", "n"]]
        new = part[part["variant"] == "v21_tokens"][["year", "mean", "n"]]
        joined = old.merge(new, on="year", how="outer", suffixes=("_v1", "_v21"),
                           validate="one_to_one")
        if joined.isna().any().any() or not (joined["n_v1"] == joined["n_v21"]).all():
            raise ValueError("Version contrasts require identical annual matched-song populations")
        contrast = joined.assign(mean=joined["mean_v21"] - joined["mean_v1"])
        tests.extend({
            "sample": "matched_all_variants", "time_basis": "debut_year",
            "variant": "v21_tokens_minus_v1_tokens", "dimension": dimension,
            "weighting": "exposure", "comparison": "v21_minus_v1", "is_primary": False, **row,
        } for row in trend_tests(contrast, n_boot))
    primary_dimensions = {row["dimension"] for row in tests if row["is_primary"]}
    if primary_dimensions != {"valence", "arousal", "dominance"}:
        raise ValueError("All three primary VAD dimensions are required")
    _correct_families(tests)
    return {
        "chart_from": chart_from, "chart_through": chart_through,
        "first_complete_year": first_complete, "last_complete_year": last_complete,
        "n_comparisons": len(tests) // (2 * len(WINDOWS)), "n_tests": len(tests),
        "settings": {
            "annual_weighting": "Each annual mean has equal regression weight; song exposure "
                                "weights remain inside the exposure-weighted annual means.",
            "estimand": "OLS linear change per decade on the common 0-1 scale; not the "
                        "earlier Theil-Sen estimator or a constant year-by-year change.",
            "newey_west": "Bartlett kernel; finite-sample covariance correction; t inference "
                         "with n_years-2 degrees of freedom.",
            "windows_years": list(WINDOWS), "main_window_years": 5,
            "bootstrap": "Circular fixed-length blocks of centered, detrended OLS residuals; "
                         "fixed years. Basic 95% intervals; two-sided null p-values from centered "
                         "slope errors, with plus-one Monte Carlo correction.",
            "n_boot": n_boot, "seed": SEED, "minimum_bootstrap_p": 1 / (n_boot + 1),
            "multiplicity": "Holm within each method/window: three primary v2.1 token dimensions; "
                            "three paired version contrasts; and separately all comparisons. "
                            "An additional correction covers the entire method/window test battery. "
                            "Confidence intervals are pointwise, not simultaneous.",
            "primary": "Matched-all-variants songs, debut year, exposure weighting, v2.1 tokens.",
            "selection": "Retrospective checks after viewing the trends, not preregistered tests. "
                         "All windows are reported; no window chosen for its p-value.",
        },
        "limitations": [
            "Failure to reject zero is not evidence of equivalence or no meaningful change.",
            "HAC and block inference are approximate with only about seven decades of observations.",
            "Block resampling assumes approximately stationary, weakly dependent detrended residuals; "
            "HAC allows heteroskedasticity but cannot remove arbitrary long-memory dependence.",
            "These tests condition on observed annual means and their lyric-covered population; "
            "they do not independently resample songs, correct artist clusters, repair missing "
            "lyrics, validate emotion constructs, or establish causality.",
            "An average linear trend can conceal nonlinear or period-specific changes. "
            "Sensitivity series share songs and are not independent replications.",
        ],
        "tests": tests,
    }


def run(series_path: Path, chart_from: str, chart_through: str, output_dir: Path) -> dict:
    result = analyze_series(pd.read_csv(series_path), chart_from, chart_through)
    result["input_signature"] = fingerprint(
        {"chart_from": chart_from, "chart_through": chart_through}, [series_path, Path(__file__)],
    )
    atomic_json(output_dir / "nrc_vad_statistics.json", result)
    pd.DataFrame(result["tests"]).to_csv(output_dir / "nrc_vad_statistics.csv", index=False)
    return result
