"""Small weighted-statistics helpers shared by the missing-value and outlier
stages (weighted mean/median for imputation, weighted percentiles for
outlier cap suggestions)."""

from typing import Union

import numpy as np
import pandas as pd


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total_weight)


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """Weighted quantile via linear interpolation on the cumulative weight
    curve (Hazen-style: each observation's cumulative weight is centered on
    weight/2 either side, matching how proc means' default percentile
    definition behaves for evenly-weighted data)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= weights.sum()
    return float(np.interp(q, cum_weights, values))


def weighted_mode(values: pd.Series, weights: pd.Series) -> Union[str, float]:
    """The value with the largest total weight -- works for both numeric and
    categorical series."""
    df = pd.DataFrame({"v": values, "w": weights})
    totals = df.groupby("v")["w"].sum()
    return totals.idxmax()


def rebalance_weight_5050(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """50/50 rebalance: scales bad-flag=1 rows' weight so the weighted bad
    and good populations become equal -- ports SAS's
    `multiplier=(100-percent)/percent` step, used identically by Stage 8
    (multifactor model search) and Stage 10 (bootstrap validation). Stage 9
    (model selection) uses a subtly different formula of its own -- see
    pipeline.stages.s09_model_selection -- so it isn't shared here."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    bad_wt = (y * w).sum()
    good_wt = ((1 - y) * w).sum()
    if bad_wt <= 0:
        return w.copy()
    multiplier = good_wt / bad_wt
    return np.where(y == 1, w * multiplier, w)


def variance_inflation_factors(corr: pd.DataFrame) -> dict:
    """VIF per variable, via the standard inverse-correlation-matrix
    diagonal formula -- mathematically equivalent to (but far cheaper than)
    regressing each variable on all the others and taking 1/(1-R^2). Used
    by Stage 11's multicollinearity diagnostic (ports `proc reg ... /vif`).
    A single-variable "matrix" has no other predictors to correlate
    against, so its VIF is trivially 1.0."""
    cols = list(corr.columns)
    if len(cols) <= 1:
        return {c: 1.0 for c in cols}
    inv = np.linalg.pinv(corr.to_numpy())
    return {c: float(inv[i, i]) for i, c in enumerate(cols)}


def weighted_corr_matrix(df: pd.DataFrame, columns: list, weight: pd.Series) -> pd.DataFrame:
    """Weighted Pearson correlation matrix across `columns` -- ports the
    `proc corr ... weight sample_weight` step shared by "7.0"/"7.1"'s
    factor-loading and correlation-pair analysis."""
    w = np.asarray(weight, dtype=float)
    total_w = w.sum()
    standardized = {}
    for c in columns:
        x = df[c].astype(float).to_numpy()
        mean = np.average(x, weights=w)
        var = np.average((x - mean) ** 2, weights=w)
        std = np.sqrt(var) if var > 0 else 1.0
        standardized[c] = (x - mean) / std

    n = len(columns)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            value = np.average(standardized[columns[i]] * standardized[columns[j]], weights=w) if total_w > 0 else 0.0
            corr[i, j] = corr[j, i] = value
    return pd.DataFrame(corr, index=columns, columns=columns)


def weighted_trend_buckets(
    x: pd.Series, y: pd.Series, weight: pd.Series, n_buckets: int = 20, min_bucket_size: float = 100.0
) -> pd.DataFrame:
    """Buckets x into up to n_buckets weighted-quantile groups (shrinking the
    count until every bucket clears min_bucket_size, or a floor of 2 -- the
    same shrink-loop idea as pipeline.common.metrics.weighted_woe_iv) and
    returns one row per bucket: mean x, Laplace-smoothed weighted log-odds of
    y, and total bucket weight. Ports the moving-average-smoothed bad-rate
    trend "5.0 create variable trends.sas" builds per variable -- simplified
    from SAS's moving-average window to straightforward quantile bucketing,
    which is more robust and needs no extra smoothing-window parameter."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)

    num_bins = n_buckets
    bin_id = None
    while num_bins >= 2:
        try:
            candidate = pd.qcut(x, num_bins, labels=False, duplicates="drop")
        except ValueError:
            num_bins -= 1
            continue
        bucket_weight = pd.Series(weight).groupby(candidate).sum()
        if bucket_weight.min() >= min_bucket_size or len(bucket_weight) <= 2:
            bin_id = candidate
            break
        num_bins -= 1
    if bin_id is None:
        bin_id = pd.qcut(x, 2, labels=False, duplicates="drop")

    df = pd.DataFrame({"bin": bin_id, "x": x, "bad_wt": y * weight, "good_wt": (1 - y) * weight, "wt": weight})
    grp = df.groupby("bin").apply(
        lambda g: pd.Series(
            {
                "x": np.average(g["x"], weights=g["wt"]),
                "ln_odd": np.log((g["bad_wt"].sum() + 1) / (g["good_wt"].sum() + 1)),
                "weight": g["wt"].sum(),
            }
        ),
        include_groups=False,
    )
    return grp.sort_values("x").reset_index(drop=True)
