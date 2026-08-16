import numpy as np
import pytest

from pipeline.common.curves import (
    applicable_families,
    assess,
    eval_binning,
    evaluate,
    fit_binning,
    fit_bell,
    fit_linear,
    fit_log_linear,
    fit_quadratic,
    fit_s_shape,
    weighted_r_squared,
)


def test_fit_linear_recovers_known_slope_and_intercept():
    x = np.linspace(0, 10, 50)
    y = 2.0 + 0.5 * x
    w = np.ones_like(x)
    result = fit_linear(x, y, w)
    assert result.converged
    assert result.params["a0"] == pytest.approx(2.0, abs=1e-6)
    assert result.params["a1"] == pytest.approx(0.5, abs=1e-6)


def test_fit_linear_r_squared_perfect_fit_is_one():
    x = np.linspace(-5, 5, 40)
    y = -1.0 + 3.0 * x
    w = np.ones_like(x)
    result = fit_linear(x, y, w)
    pred = evaluate(result, x)
    assert weighted_r_squared(y, pred, w) == pytest.approx(1.0, abs=1e-6)


def test_fit_quadratic_recovers_u_shape():
    x = np.linspace(-5, 5, 60)
    y = 1.0 + 0.0 * x + 2.0 * x**2
    w = np.ones_like(x)
    result = fit_quadratic(x, y, w)
    assert result.params["a2"] == pytest.approx(2.0, abs=1e-4)
    assert result.params["a1"] == pytest.approx(0.0, abs=1e-4)


def test_fit_log_linear_recovers_known_relationship():
    x = np.linspace(1, 100, 50)
    y = 0.5 + 1.5 * np.log(x)
    w = np.ones_like(x)
    result = fit_log_linear(x, y, w)
    assert result.params["a0"] == pytest.approx(0.5, abs=1e-6)
    assert result.params["a1"] == pytest.approx(1.5, abs=1e-6)


def test_fit_bell_converges_on_bell_shaped_data():
    xn = np.linspace(0, 1, 60)
    true_y = (2 * np.pi * 0.15**2) ** -0.5 * np.exp(-((xn - 0.5) ** 2) / (2 * 0.15**2)) - 3.0
    w = np.ones_like(xn)
    result = fit_bell(xn, true_y, w, min_x=0.0, max_x=1.0)
    assert result.converged
    pred = evaluate(result, xn)
    assert weighted_r_squared(true_y, pred, w) > 0.99


def test_fit_s_shape_converges_on_logistic_data():
    xn = np.linspace(0, 1, 60)
    true_y = 1 / (1 + np.exp(-(xn * 8 - 4))) + 1.0
    w = np.ones_like(xn)
    result = fit_s_shape(xn, true_y, w, min_x=0.0, max_x=1.0)
    assert result.converged
    pred = evaluate(result, xn)
    assert weighted_r_squared(true_y, pred, w) > 0.99


def test_fit_binning_produces_monotonic_lookup_on_monotonic_data():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0, 100, n)
    bad_prob = np.clip(x / 100, 0.01, 0.99)
    y = rng.binomial(1, bad_prob)
    w = np.ones(n)
    result = fit_binning(x, y, w, n_bins=10, min_bin_size=50)
    assert result.converged
    log_odds = result.extra["log_odds"]
    # binning a monotonic-increasing true relationship should yield a
    # (roughly) monotonic-increasing lookup table
    assert log_odds == sorted(log_odds)


def test_eval_binning_clips_out_of_range_values_to_edge_bins():
    result = fit_binning(np.linspace(0, 100, 500), np.random.default_rng(1).binomial(1, 0.3, 500), np.ones(500))
    below = eval_binning(result, np.array([-1000.0]))
    above = eval_binning(result, np.array([1e9]))
    assert below[0] == result.extra["log_odds"][0]
    assert above[0] == result.extra["log_odds"][-1]


def test_applicable_families_excludes_bell_for_increasing_trend():
    families = applicable_families("increasing", min_x=1.0)
    assert "bell" not in families
    assert "inverted_bell" not in families
    assert "linear" in families
    assert "log_linear" in families
    assert "binning" in families


def test_applicable_families_excludes_log_linear_when_min_x_not_positive():
    families = applicable_families("any", min_x=-5.0)
    assert "log_linear" not in families


def test_applicable_families_includes_bell_only_for_any_or_inverted_u():
    assert "bell" in applicable_families("any", 1.0)
    assert "bell" in applicable_families("inverted_u_shape", 1.0)
    assert "bell" not in applicable_families("u_shape", 1.0)
    assert "bell" not in applicable_families("increasing", 1.0)


def test_assess_rejects_wrong_signed_slope_for_expected_trend():
    x = np.linspace(0, 10, 50)
    y = 5.0 - 2.0 * x  # clearly decreasing
    w = np.ones_like(x)
    result = fit_linear(x, y, w)
    verdict = assess(result, "increasing", sig_level=0.20, r2_level=0.01, r_squared=1.0, min_x=0, max_x=10)
    assert "does not follow expected trend" in verdict


def test_assess_usable_for_strong_conforming_signal():
    x = np.linspace(0, 10, 200)
    y = 1.0 + 0.8 * x
    w = np.ones_like(x)
    result = fit_linear(x, y, w)
    pred = evaluate(result, x)
    r2 = weighted_r_squared(y, pred, w)
    verdict = assess(result, "increasing", sig_level=0.20, r2_level=0.01, r_squared=r2, min_x=0, max_x=10)
    assert verdict == "usable"


def test_assess_binning_always_usable_regardless_of_trend():
    x = np.linspace(0, 100, 500)
    y = np.random.default_rng(2).binomial(1, 0.3, 500)
    w = np.ones(500)
    result = fit_binning(x, y, w)
    verdict = assess(result, "u_shape", sig_level=0.01, r2_level=0.99, r_squared=0.0, min_x=0, max_x=100)
    assert verdict == "usable"
