"""Weighted Gini and Information Value (IV), the statistical core reused by
every stage that ranks a variable's predictive power. This is a single
Python port of what the original SAS pipeline implemented four times over
(gini_num_non_missing, gini_num_with_missing, gini_char_non_missing,
gini_char_with_missing in "1.0 initial univariate ginis.sas").

gini_with_missing/IV deviate from the SAS macros in one deliberate way: SAS
built one dummy 0/1 indicator per distinct special/dummy value and fit
`proc logistic` on (score, dum_1, dum_2, ...). Here every missing/dummy
value collapses into a single "is_missing" indicator before fitting -- for
this pipeline's `dummy_values` list (usually empty or a couple of sentinel
codes) that's a negligible simplification, and it keeps the model from
becoming rank-deficient when a variable has many missing patterns.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def weighted_gini(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    """Weighted Gini (accuracy ratio) of `score` for predicting `y`, using
    `weight` as the per-observation multiplier.

    Ports the cumulative-bad trapezoidal accumulation from the SAS gini_*
    macros exactly: sort by score descending, walk tied-score groups
    accumulating ROC_area, then rescale by the achieved bad rate so the
    result is comparable across variables/samples with different bad rates.
    Sign is direction-agnostic (abs()'d at the end) since ranking a variable
    ascending vs descending only flips the sign, never the magnitude --
    same as the SAS macros, which score whichever raw direction the
    variable happens to have and always take abs(gini).

    All three arrays must already be free of NaN/missing observations --
    callers filter before calling this.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)

    tot_bad = float(np.sum(y * weight))
    tot_pop = float(np.sum(weight))

    if tot_pop <= 0 or tot_bad <= 0 or tot_bad >= tot_pop:
        return 0.0

    df = pd.DataFrame({"score": score, "bad_wt": y * weight, "wt": weight})
    grouped = df.groupby("score", sort=True)[["bad_wt", "wt"]].sum()
    grouped = grouped.sort_index(ascending=False)  # descending score, matches SAS

    count_one = grouped["bad_wt"].to_numpy()
    i = grouped["wt"].to_numpy()
    cumm_bad_before = np.concatenate(([0.0], np.cumsum(count_one)[:-1]))

    roc_area = np.sum(i * (cumm_bad_before + count_one / 2) / (tot_bad * tot_pop))

    bad_rate = tot_bad / tot_pop
    denom = 0.5 - bad_rate / 2
    gini = (roc_area - 0.5) / denom
    return abs(float(gini))


def bad_rate_encode(x: pd.Series, y: pd.Series, weight: pd.Series) -> pd.Series:
    """Maps each category of a categorical variable to its weighted mean bad
    rate, computed over non-missing rows only -- ports the
    `proc means ... mean(bad_flag)=bad_rate` step in gini_char_non_missing /
    gini_char_with_missing, which is how SAS turns a character variable into
    something rankable for the Gini calculation."""
    df = pd.DataFrame({"x": x, "bad_wt": y * weight, "wt": weight})
    grp = df.groupby("x")[["bad_wt", "wt"]].sum()
    bad_rate = grp["bad_wt"] / grp["wt"]
    return x.map(bad_rate)


def weighted_woe_iv(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    bins: int = 20,
    min_bin_size: float = 100.0,
    min_bins_floor: int = 3,
) -> float:
    """Weighted Information Value of `score` for predicting `y`. Quantile-bins
    `score` into up to `bins` groups, shrinking the bin count until every bin
    holds at least `min_bin_size` weighted observations (or the floor is
    hit) -- ports the `num_groups` shrink-loop in gini_num_with_missing /
    gini_char_with_missing. WOE/IV per bin uses the same +1 Laplace
    smoothing as the SAS `data IV` step.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)

    tot_bad = float(np.sum(y * weight))
    tot_pop = float(np.sum(weight))
    if tot_pop <= 0:
        return 0.0

    num_bins = bins
    while num_bins >= min_bins_floor:
        try:
            bin_id = pd.qcut(score, num_bins, labels=False, duplicates="drop")
        except ValueError:
            num_bins -= 1
            continue
        bin_weights = pd.Series(weight).groupby(bin_id).sum()
        if bin_weights.min() >= min_bin_size:
            break
        num_bins -= 1
    else:
        return 0.0

    df = pd.DataFrame({"bin": bin_id, "bad_wt": y * weight, "good_wt": (1 - y) * weight})
    grp = df.groupby("bin")[["bad_wt", "good_wt"]].sum()
    n_bins_actual = len(grp)
    if n_bins_actual < 2:
        return 0.0

    tot_good = tot_pop - tot_bad
    pct_good = (grp["good_wt"] + 1) / (tot_good + n_bins_actual - tot_bad)
    pct_bad = (grp["bad_wt"] + 1) / (tot_bad + n_bins_actual)
    woe = np.log(pct_good / pct_bad)
    iv = float(np.sum((pct_good - pct_bad) * woe))
    return abs(iv)


def fit_missing_aware_score(
    x: pd.Series, y: pd.Series, weight: pd.Series, dummy_values: Optional[Iterable] = None
) -> Optional[np.ndarray]:
    """Fits `bad_flag ~ x + is_missing` (logistic, weighted) over the FULL
    population including missing/dummy rows, and returns the fitted
    probability of bad for every row. This is the "with missing" score used
    by weighted_gini/weighted_woe_iv -- ports gini_num_with_missing /
    gini_char_with_missing's use of a fitted probability (rather than the
    raw value) so that missingness itself can carry predictive signal.

    Returns None if the model can't be fit (e.g. no variance, all one
    class) -- callers should treat that as gini_with_missing = 0.
    """
    dummy_values = set(dummy_values or [])
    is_missing = x.isna() | x.isin(dummy_values) if dummy_values else x.isna()
    x_filled = x.where(~is_missing, 0.0).astype(float)

    features = pd.DataFrame({"x": x_filled, "is_missing": is_missing.astype(float)})
    if features["is_missing"].nunique() < 2:
        features = features[["x"]]

    y_arr = y.to_numpy()
    if len(np.unique(y_arr)) < 2:
        return None

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(features, y_arr, sample_weight=weight.to_numpy())
        return model.predict_proba(features)[:, 1]
    except Exception:
        return None
