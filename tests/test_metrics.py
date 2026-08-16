import numpy as np
import pandas as pd
import pytest

from pipeline.common.metrics import (
    bad_rate_encode,
    fit_missing_aware_score,
    weighted_gini,
    weighted_woe_iv,
)


def test_weighted_gini_perfect_separation():
    # hand-verified: descending-score walk gives ROC_area=0.75, bad_rate=0.5
    # => gini=(0.75-0.5)/0.25=1.0
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([1, 2, 3, 4, 5, 6])
    weight = np.ones(6)
    assert weighted_gini(y, score, weight) == pytest.approx(1.0, abs=1e-9)


def test_weighted_gini_reversed_score_same_magnitude():
    y = np.array([0, 0, 0, 1, 1, 1])
    score_desc = np.array([1, 2, 3, 4, 5, 6])
    score_asc = np.array([6, 5, 4, 3, 2, 1])
    weight = np.ones(6)
    g1 = weighted_gini(y, score_desc, weight)
    g2 = weighted_gini(y, score_asc, weight)
    assert g1 == pytest.approx(g2, abs=1e-9)


def test_weighted_gini_no_signal_is_zero():
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.ones(6)  # every observation tied -> no ranking signal
    weight = np.ones(6)
    assert weighted_gini(y, score, weight) == pytest.approx(0.0, abs=1e-9)


def test_weighted_gini_scale_invariant_to_uniform_weight():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([1, 2, 3, 4, 5, 6])
    g_unit = weighted_gini(y, score, np.ones(6))
    g_scaled = weighted_gini(y, score, np.full(6, 5.0))
    assert g_unit == pytest.approx(g_scaled, abs=1e-9)


def test_weighted_gini_degenerate_all_bad_or_all_good():
    y_all_bad = np.ones(5)
    y_all_good = np.zeros(5)
    score = np.arange(5)
    weight = np.ones(5)
    assert weighted_gini(y_all_bad, score, weight) == 0.0
    assert weighted_gini(y_all_good, score, weight) == 0.0


def test_bad_rate_encode_matches_group_means():
    x = pd.Series(["a", "a", "b", "b", "b"])
    y = pd.Series([1, 0, 1, 1, 0])
    w = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    encoded = bad_rate_encode(x, y, w)
    assert encoded.iloc[0] == pytest.approx(0.5)  # category 'a': 1 bad of 2
    assert encoded.iloc[2] == pytest.approx(2 / 3)  # category 'b': 2 bad of 3


def test_weighted_woe_iv_separating_variable_beats_random():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.binomial(1, 0.3, n).astype(float)
    weight = np.ones(n)

    # a score correlated with y should have materially higher IV than pure noise
    separating_score = y * 2 + rng.normal(0, 1, n)
    random_score = rng.normal(0, 1, n)

    iv_signal = weighted_woe_iv(y, separating_score, weight, bins=10, min_bin_size=50)
    iv_noise = weighted_woe_iv(y, random_score, weight, bins=10, min_bin_size=50)

    assert iv_signal > iv_noise


def test_weighted_woe_iv_returns_zero_when_floor_unreachable():
    # too few observations to ever satisfy min_bin_size at the floor bin count
    y = np.array([0, 1, 0, 1])
    score = np.array([1, 2, 3, 4])
    weight = np.ones(4)
    assert weighted_woe_iv(y, score, weight, bins=20, min_bin_size=1000, min_bins_floor=3) == 0.0


def test_fit_missing_aware_score_shape_and_range():
    rng = np.random.default_rng(1)
    n = 500
    x = pd.Series(rng.normal(0, 1, n))
    x.iloc[:50] = np.nan
    y = pd.Series(rng.binomial(1, 0.3, n))
    w = pd.Series(np.ones(n))

    pred = fit_missing_aware_score(x, y, w)
    assert pred is not None
    assert pred.shape == (n,)
    assert np.all((pred >= 0) & (pred <= 1))


def test_fit_missing_aware_score_none_when_single_class():
    x = pd.Series([1.0, 2.0, 3.0, np.nan])
    y = pd.Series([1, 1, 1, 1])
    w = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert fit_missing_aware_score(x, y, w) is None
