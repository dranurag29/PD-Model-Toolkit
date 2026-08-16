"""Parametric "shape" transformations fit to a variable's binned log-odds
trend, replacing the proc nlin curve-fitting in
"5.1.1 numeric transformations - test applicable transformation.sas".

SAS fits 9 named shape classes (A degree-1 polynomial in 3 clamped
variants, B degree-2+1 clamped quadratic in 3 variants, C plain degree-2
polynomial, D/E/F the same three degree-1/2/2+1 shapes on log(var), G Bell,
H Inverted Bell, I S-shape) plus J Binning as an always-available
non-parametric fallback. This port keeps every qualitatively distinct curve
shape SAS was choosing between -- linear, quadratic, log-linear, bell,
inverted bell, S-shape, binning -- and deliberately drops the clamped/
piecewise sub-variants (A2/A3, B, E, F): they're hinge-function variations
on shapes already covered by quadratic/bell/inverted_bell, and nothing
downstream depends on their exact presence.

linear/quadratic/log_linear are ordinary weighted least squares (closed
form -- no iterative solver needed). bell/inverted_bell/s_shape are
genuinely nonlinear and fit via scipy.optimize.curve_fit on x normalized to
[0, 1], mirroring how the SAS macros for G/H/I normalize var before fitting
for numerical stability. binning has no smooth functional form at all: it's
an adaptive-width quantile lookup table, fit directly on row-level data
(not the coarser trend buckets) since it needs enough resolution to place
sensible cut points.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

TREND_ANY = "any"
TREND_INCREASING = "increasing"
TREND_DECREASING = "decreasing"
TREND_U_SHAPE = "u_shape"
TREND_INVERTED_U_SHAPE = "inverted_u_shape"
ALL_TRENDS = [TREND_ANY, TREND_INCREASING, TREND_DECREASING, TREND_U_SHAPE, TREND_INVERTED_U_SHAPE]

# Which fitted parameters actually define the curve's *shape* (as opposed to
# a vertical offset) -- these are the ones whose p-value gates "significant",
# matching which parameters each SAS macro checks (e.g. G/Bell and I/S-shape
# only gate on their spread/steepness parameter "a1", never the offset "a2").
SHAPE_PARAMS = {
    "linear": ["a1"],
    "quadratic": ["a1", "a2"],
    "log_linear": ["a1"],
    "bell": ["a1"],
    "inverted_bell": ["a1"],
    "s_shape": ["a1"],
    "binning": [],
}


@dataclass
class FitResult:
    family: str
    converged: bool
    params: Dict[str, float] = field(default_factory=dict)
    dof: int = 1
    param_var: Dict[str, float] = field(default_factory=dict)  # variance of each param estimate
    extra: dict = field(default_factory=dict)  # family-specific data (e.g. binning's edges/values)


def _weighted_least_squares(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    W = np.diag(w)
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWX_inv = np.linalg.pinv(XtWX)
    beta = XtWX_inv @ XtW @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float((resid**2 * w).sum() / w.sum() * len(y) / dof) if w.sum() > 0 else 0.0
    cov = XtWX_inv * sigma2
    return beta, cov, dof


def weighted_r_squared(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Squared weighted Pearson correlation between actual and predicted --
    exactly what SAS's `proc corr ... PEARSON; R2=pred*pred` computes."""
    y, pred, w = np.asarray(y, dtype=float), np.asarray(pred, dtype=float), np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return 0.0
    wm_y, wm_p = np.average(y, weights=w), np.average(pred, weights=w)
    cov = np.average((y - wm_y) * (pred - wm_p), weights=w)
    var_y = np.average((y - wm_y) ** 2, weights=w)
    var_p = np.average((pred - wm_p) ** 2, weights=w)
    if var_y <= 0 or var_p <= 0:
        return 0.0
    return float((cov / np.sqrt(var_y * var_p)) ** 2)


def significance(result: FitResult) -> Optional[float]:
    """Max two-sided p-value among a family's shape-defining parameters --
    None if it can't be estimated (unconverged fit, or a family like binning
    with no meaningful parameter covariance)."""
    shape_params = SHAPE_PARAMS[result.family]
    if not result.converged or not shape_params:
        return None
    pvals = []
    for p in shape_params:
        var = result.param_var.get(p)
        if var is None or var <= 0:
            return None
        se = np.sqrt(var)
        t = result.params[p] / se
        pvals.append(2 * scipy_stats.t.sf(abs(t), df=max(result.dof, 1)))
    return max(pvals)


# ---- linear / quadratic / log_linear (closed-form weighted OLS) -----------


def fit_linear(x, y, w) -> FitResult:
    X = np.column_stack([np.ones_like(x), x])
    beta, cov, dof = _weighted_least_squares(X, y, w)
    return FitResult("linear", True, {"a0": beta[0], "a1": beta[1]}, dof, {"a1": cov[1, 1]})


def eval_linear(params, x):
    return params["a0"] + params["a1"] * np.asarray(x, dtype=float)


def fit_quadratic(x, y, w) -> FitResult:
    X = np.column_stack([np.ones_like(x), x, x**2])
    beta, cov, dof = _weighted_least_squares(X, y, w)
    return FitResult(
        "quadratic", True, {"a0": beta[0], "a1": beta[1], "a2": beta[2]}, dof, {"a1": cov[1, 1], "a2": cov[2, 2]}
    )


def eval_quadratic(params, x):
    x = np.asarray(x, dtype=float)
    return params["a0"] + params["a1"] * x + params["a2"] * x**2


def fit_log_linear(x, y, w) -> FitResult:
    # caller guarantees min(x) > 0
    lx = np.log(x)
    X = np.column_stack([np.ones_like(lx), lx])
    beta, cov, dof = _weighted_least_squares(X, y, w)
    return FitResult("log_linear", True, {"a0": beta[0], "a1": beta[1]}, dof, {"a1": cov[1, 1]})


def eval_log_linear(params, x):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, None)
    return params["a0"] + params["a1"] * np.log(x)


# ---- bell / inverted_bell / s_shape (nonlinear, normalized-x) -------------


def _normalize(x, min_x, max_x):
    span = max_x - min_x
    if span <= 0:
        return np.zeros_like(x, dtype=float)
    return (np.asarray(x, dtype=float) - min_x) / span


def _bell_fn(xn, a0, a1, a2):
    return (2 * np.pi * a1**2) ** -0.5 * np.exp(-((xn - a0) ** 2) / (2 * a1**2)) + a2


def _inverted_bell_fn(xn, a0, a1, a2):
    return -((2 * np.pi * a1**2) ** -0.5) * np.exp(-((xn - a0) ** 2) / (2 * a1**2)) + a2


def _s_shape_fn(xn, a0, a1, a2):
    return 1 / (1 + np.exp(-(xn * a1 - a0))) + a2


def _fit_nonlinear(family, fn, x, y, w, min_x, max_x, a0_bounds=(0.0, 1.0)) -> FitResult:
    """Multi-start nonlinear weighted least squares: curve_fit's local
    optimizer can converge to a degenerate near-zero-amplitude local optimum
    (a1 collapsing toward its lower bound, leaving an almost-flat curve)
    that -- because Pearson correlation is scale-invariant -- can still
    report a misleadingly high R square from a microscopic residual drift
    that happens to share the data's sign pattern. SAS's macros guard
    against local optima the same way, seeding proc nlin from several
    starting points (e.g. the 33%/50%/66% inflection guesses in
    check_degree_2_1_poly) rather than a single guess. Selection among
    starts uses weighted SSE (curve_fit's actual objective), NOT R square --
    SSE correctly penalizes a near-flat curve that fails to explain the
    data's real variance, where R square would reward it."""
    xn = _normalize(x, min_x, max_x)
    lo, hi = a0_bounds
    span = hi - lo if np.isfinite(hi - lo) else 1.0
    a0_inits = [lo + span * f for f in (0.25, 0.5, 0.75)]
    a1_inits = [0.5, 2.0, 5.0, 8.0]
    bounds = ([lo, 1e-3, -np.inf], [hi, 10.0, np.inf])
    sigma = 1 / np.sqrt(np.maximum(w, 1e-9))

    best, best_sse = None, np.inf
    for a0_init in a0_inits:
        for a1_init in a1_inits:
            p0 = [a0_init, a1_init, float(np.average(y, weights=w))]
            try:
                popt, pcov = curve_fit(fn, xn, y, p0=p0, sigma=sigma, absolute_sigma=False, bounds=bounds, maxfev=5000)
            except Exception:
                continue
            if not np.all(np.isfinite(pcov)):
                continue
            sse = float(np.sum(w * (y - fn(xn, *popt)) ** 2))
            if sse < best_sse:
                best, best_sse = (popt, pcov), sse
    if best is None:
        return FitResult(family, False)
    popt, pcov = best
    dof = max(len(y) - 3, 1)
    return FitResult(
        family,
        True,
        {"a0": popt[0], "a1": popt[1], "a2": popt[2]},
        dof,
        {"a1": pcov[1, 1]},
        extra={"min_x": min_x, "max_x": max_x},
    )


def fit_bell(x, y, w, min_x, max_x) -> FitResult:
    return _fit_nonlinear("bell", _bell_fn, x, y, w, min_x, max_x)


def eval_bell(params, x, min_x, max_x):
    xn = _normalize(x, min_x, max_x)
    return _bell_fn(xn, params["a0"], params["a1"], params["a2"])


def fit_inverted_bell(x, y, w, min_x, max_x) -> FitResult:
    return _fit_nonlinear("inverted_bell", _inverted_bell_fn, x, y, w, min_x, max_x)


def eval_inverted_bell(params, x, min_x, max_x):
    xn = _normalize(x, min_x, max_x)
    return _inverted_bell_fn(xn, params["a0"], params["a1"], params["a2"])


def fit_s_shape(x, y, w, min_x, max_x) -> FitResult:
    # unlike bell/inverted_bell, S-shape's a0 is a logit-space midpoint
    # shift, not a normalized peak location -- SAS's check_S_shape doesn't
    # bound it to [0, 1] either, so a generous-but-finite range instead of
    # bell's (0, 1) keeps curve_fit numerically stable without artificially
    # clamping the midpoint into the observed x range.
    return _fit_nonlinear("s_shape", _s_shape_fn, x, y, w, min_x, max_x, a0_bounds=(-20.0, 20.0))


def eval_s_shape(params, x, min_x, max_x):
    xn = _normalize(x, min_x, max_x)
    return _s_shape_fn(xn, params["a0"], params["a1"], params["a2"])


# ---- binning (adaptive-width quantile WOE lookup, always usable) ----------


def fit_binning(x, y, w, n_bins: int = 10, min_bin_size: float = 100.0) -> FitResult:
    """Adaptive quantile bins on row-level (x, bad_flag, weight) data --
    shrinks the bin count until every bin clears min_bin_size (or a floor of
    2), the same pattern as weighted_woe_iv -- then stores each bin's right
    edge and Laplace-smoothed weighted log-odds for lookup at eval time."""
    x, y, w = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    num_bins = n_bins
    bin_id = None
    while num_bins >= 2:
        try:
            candidate = pd.qcut(x, num_bins, labels=False, duplicates="drop")
        except ValueError:
            num_bins -= 1
            continue
        bin_weight = pd.Series(w).groupby(candidate).sum()
        if bin_weight.min() >= min_bin_size or len(bin_weight) <= 2:
            bin_id = candidate
            break
        num_bins -= 1
    if bin_id is None:
        return FitResult("binning", False)

    df = pd.DataFrame({"bin": bin_id, "x": x, "bad_wt": y * w, "good_wt": (1 - y) * w})
    grp = df.groupby("bin").agg(edge=("x", "max"), bad_wt=("bad_wt", "sum"), good_wt=("good_wt", "sum"))
    grp = grp.sort_values("edge")
    log_odds = np.log((grp["bad_wt"] + 1) / (grp["good_wt"] + 1))
    return FitResult(
        "binning", True, {}, dof=1, param_var={}, extra={"edges": grp["edge"].tolist(), "log_odds": log_odds.tolist()}
    )


def eval_binning(result: FitResult, x):
    edges = np.array(result.extra["edges"])
    log_odds = np.array(result.extra["log_odds"])
    idx = np.searchsorted(edges, np.asarray(x, dtype=float), side="left")
    idx = np.clip(idx, 0, len(log_odds) - 1)
    return log_odds[idx]


# ---- orchestration ----------------------------------------------------


def applicable_families(expected_trend: str, min_x: float) -> List[str]:
    """Ports the upfront model x Expected_Trend filter table in
    "5.1.1 ...".all_univariate (the `keep=0/1` block): which shapes even
    make sense to attempt for a variable's expected trend, before any
    fitting happens."""
    families = []
    if expected_trend in (TREND_ANY, TREND_INCREASING, TREND_DECREASING):
        families += ["linear", "s_shape"]
        if min_x > 0:
            families.append("log_linear")
    families.append("quadratic")  # attempted for every trend; conformity checked post-hoc
    if expected_trend in (TREND_ANY, TREND_INVERTED_U_SHAPE):
        families.append("bell")
    if expected_trend in (TREND_ANY, TREND_U_SHAPE):
        families.append("inverted_bell")
    families.append("binning")  # always applicable, always usable
    return families


def _trend_conformity_reason(result: FitResult, expected_trend: str, min_x: float, max_x: float) -> Optional[str]:
    if expected_trend == TREND_ANY or result.family == "binning":
        return None
    p = result.params
    if result.family in ("linear", "log_linear", "s_shape"):
        if expected_trend == TREND_INCREASING and p["a1"] <= 0:
            return f"does not follow expected trend '{expected_trend}'"
        if expected_trend == TREND_DECREASING and p["a1"] >= 0:
            return f"does not follow expected trend '{expected_trend}'"
        return None
    if result.family == "quadratic":
        vertex = -p["a1"] / (2 * p["a2"]) if p["a2"] != 0 else None
        vertex_in_range = vertex is not None and min_x < vertex < max_x
        if expected_trend == TREND_U_SHAPE and not (p["a2"] > 0 and vertex_in_range):
            return f"does not follow expected trend '{expected_trend}'"
        if expected_trend == TREND_INVERTED_U_SHAPE and not (p["a2"] < 0 and vertex_in_range):
            return f"does not follow expected trend '{expected_trend}'"
        return None
    if result.family in ("bell", "inverted_bell"):
        peak_in_range = 0.0 < p["a0"] < 1.0
        if not peak_in_range:
            return "no inflection point in variable range"
        return None
    return None


def assess(result: FitResult, expected_trend: str, sig_level: float, r2_level: float, r_squared: float,
           min_x: float, max_x: float) -> str:
    """Usability verdict for a fitted curve -- ports the assessment cascade
    every SAS check_* macro ends with: converged? trend-conforming?
    significant? enough R square? else "usable"."""
    if not result.converged:
        return "failed to converge"
    reason = _trend_conformity_reason(result, expected_trend, min_x, max_x)
    if reason:
        return reason
    if result.family != "binning":
        sig = significance(result)
        if sig is None or sig > sig_level:
            return "not significant"
        if r_squared < r2_level:
            return "low R square"
    return "usable"


FIT_FUNCS = {
    "linear": lambda x, y, w, min_x, max_x: fit_linear(x, y, w),
    "quadratic": lambda x, y, w, min_x, max_x: fit_quadratic(x, y, w),
    "log_linear": lambda x, y, w, min_x, max_x: fit_log_linear(x, y, w),
    "bell": fit_bell,
    "inverted_bell": fit_inverted_bell,
    "s_shape": fit_s_shape,
}

EVAL_FUNCS = {
    "linear": lambda result, x: eval_linear(result.params, x),
    "quadratic": lambda result, x: eval_quadratic(result.params, x),
    "log_linear": lambda result, x: eval_log_linear(result.params, x),
    "bell": lambda result, x: eval_bell(result.params, x, result.extra["min_x"], result.extra["max_x"]),
    "inverted_bell": lambda result, x: eval_inverted_bell(result.params, x, result.extra["min_x"], result.extra["max_x"]),
    "s_shape": lambda result, x: eval_s_shape(result.params, x, result.extra["min_x"], result.extra["max_x"]),
    "binning": lambda result, x: eval_binning(result, x),
}


def evaluate(result: FitResult, x) -> np.ndarray:
    return np.asarray(EVAL_FUNCS[result.family](result, x), dtype=float)
