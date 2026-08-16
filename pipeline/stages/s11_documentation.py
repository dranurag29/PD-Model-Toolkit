"""Stage 11: Model documentation.

Ports "11.1 model documentation 50-50 weights.sas" and
"11.2 model documentation ACTUAL weights.sas" as one stage, since 11.2
directly reuses 11.1's fitted model -- only its intercept gets
recalibrated -- rather than fitting anything new. An analyst would
naturally treat "the documentation" as one deliverable with two
calibration views, matching why SAS itself numbered these 11.1/11.2
instead of as separate top-level stages like every other X.0 file here.

Picks Stage 10's #1-ranked (by val_gini) validated model by default
(override via decisions/documentation_rules.yaml's selected_combination_id),
refits it on the full 50/50-rebalanced development sample (same
rebalancing as Stage 8/10), and produces two decile calibration tables (N
population-weighted buckets, ascending by predicted probability -- ports
the `rank_pred_prob=int(cumm/total*20)` logic):

- "50/50" view (11.1): predictions and weights exactly as fit. Useful for
  checking rank-ordering/monotonicity (does a higher predicted-risk bucket
  really have a higher actual bad rate?), NOT for absolute calibration --
  the rebalanced population's ~50% bad rate is artificial.

- "actual" view (11.2): SAS's offset-regression intercept-correction
  technique. Convert every prediction to log-odds, then refit an
  intercept-only model with that log-odds held fixed as a GLM offset
  (coefficient locked to 1.0, not estimated) against a weighting scaled to
  target_bad_rate. This separates "which variables matter and how they
  trade off" (learned better on a balanced sample, where the rare class
  doesn't get drowned out) from "what's the true baseline risk level"
  (restored via the offset correction) -- standard practice for rare-event
  logistic regression, sometimes called prior/case-control correction. The
  slope coefficients are unchanged from the 50/50 fit; only the intercept
  moves.

target_bad_rate replaces SAS's hardcoded segment_bad_rate (a specific
business constant from the original engagement -- a portfolio statistic
with no meaning outside it) with an explicit 0-1 fraction; null means "use
the development sample's own observed weighted bad rate", the only
defensible zero-config default.

Also reports VIF and a correlation matrix for the final variables (via
pipeline.common.stats.variance_inflation_factors / weighted_corr_matrix),
ports `proc reg .../vif` and `proc corr`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    read_metrics,
    write_artifact,
    write_metrics,
)
from pipeline.common.stats import (  # noqa: E402
    rebalance_weight_5050,
    variance_inflation_factors,
    weighted_corr_matrix,
)

STAGE = "11_documentation"
SOURCE_STAGE = "10_validation"
DECISIONS_NAME = "documentation_rules"


def _select_model(validation_metrics: dict, decisions: dict):
    models = validation_metrics.get("models", [])
    if not models:
        return None
    selected_id = decisions.get("selected_combination_id")
    if selected_id:
        for m in models:
            if m["combination_id"] == selected_id:
                return m
    return models[0]  # already sorted by val_gini, descending


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y, sample_weight=w)
    return float(model.intercept_[0]), model.coef_[0].astype(float)


def _calibration_table(pred_prob: np.ndarray, y: np.ndarray, w: np.ndarray, n_buckets: int) -> list:
    """Population-weighted percentile buckets, ascending by predicted
    probability -- ports rank_pred_prob=int(cumm_weight/total*n_buckets)."""
    order = np.argsort(pred_prob)
    pred_prob, y, w = pred_prob[order], y[order], w[order]
    cum_w = np.cumsum(w)
    total = cum_w[-1]
    bucket = np.minimum((cum_w / (total + 1e-5) * n_buckets).astype(int), n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        mask = bucket == b
        if not mask.any():
            continue
        wb = w[mask]
        rows.append(
            {
                "bucket": b,
                "count_good": round(float(((1 - y[mask]) * wb).sum()), 2),
                "count_bad": round(float((y[mask] * wb).sum()), 2),
                "pred_bad_rate": round(float(np.average(pred_prob[mask], weights=wb)), 4),
                "act_bad_rate": round(float(np.average(y[mask], weights=wb)), 4),
            }
        )
    return rows


def _vif_and_corr(dev: pd.DataFrame, var_list: list, weight: np.ndarray):
    if len(var_list) > 1:
        corr = weighted_corr_matrix(dev, var_list, pd.Series(weight))
        vif = variance_inflation_factors(corr)
        corr_out = {a: {b: round(float(corr.loc[a, b]), 4) for b in corr.columns} for a in corr.index}
    else:
        vif = {var_list[0]: 1.0}
        corr_out = {}
    return {k: round(v, 3) for k, v in vif.items()}, corr_out


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    dp = params["documentation"]
    bad_flag, sample_weight = g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    validation_metrics = read_metrics(SOURCE_STAGE)

    chosen = _select_model(validation_metrics, decisions)
    if chosen is None:
        write_artifact(STAGE, "development", dev)
        write_artifact(STAGE, "validation", val)
        metrics = {"selected": None}
        write_metrics(STAGE, metrics)
        return metrics

    var_list = chosen["variables"]
    n_buckets = dp["n_calibration_buckets"]

    y = dev[bad_flag].astype(float).to_numpy()
    w_orig = dev[sample_weight].astype(float).to_numpy()
    w_5050 = rebalance_weight_5050(y, w_orig)
    X = dev[var_list].astype(float).to_numpy()

    intercept_5050, coefs_5050 = _fit_logistic(X, y, w_5050)
    pred_5050 = 1 / (1 + np.exp(-(intercept_5050 + X @ coefs_5050)))
    calibration_5050 = _calibration_table(pred_5050, y, w_5050, n_buckets)
    vif_5050, corr_5050 = _vif_and_corr(dev, var_list, w_5050)

    # --- 11.2: recalibrate the intercept to target_bad_rate -----------
    observed_bad_rate = float(np.average(y, weights=w_orig))
    target_bad_rate = dp["target_bad_rate"] if dp["target_bad_rate"] is not None else observed_bad_rate

    # bad rows: original weight * 50/50 multiplier * an extra target-rate
    # factor; good rows: original weight, unchanged -- ports SAS restoring
    # `sample_weight=wt` before applying `multiplier*(segment_bad_rate/(100-segment_bad_rate))`
    # (derived here in 0-1 fractional terms instead of SAS's 0-100 percent scale)
    extra_factor = target_bad_rate / (1 - target_bad_rate)
    w_actual = np.where(y == 1, w_5050 * extra_factor, w_orig)

    ln_odd = np.log(pred_5050 / (1 - pred_5050))
    offset_result = sm.GLM(
        y, np.ones((len(y), 1)), family=sm.families.Binomial(), freq_weights=w_actual, offset=ln_odd
    ).fit()
    intercept_actual = float(offset_result.params[0])
    pred_actual = 1 / (1 + np.exp(-(intercept_actual + ln_odd)))

    calibration_actual = _calibration_table(pred_actual, y, w_actual, n_buckets)
    vif_actual, corr_actual = _vif_and_corr(dev, var_list, w_actual)

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {
        "selected": {
            "combination_id": chosen["combination_id"],
            "variables": var_list,
            "dev_gini": chosen["dev_gini"],
            "val_gini": chosen["val_gini"],
        },
        "observed_bad_rate": round(observed_bad_rate, 4),
        "target_bad_rate": round(target_bad_rate, 4),
        "model_5050": {
            "intercept": round(intercept_5050, 6),
            "coefficients": {v: round(float(c), 6) for v, c in zip(var_list, coefs_5050)},
            "calibration": calibration_5050,
            "vif": vif_5050,
            "correlation_matrix": corr_5050,
        },
        "model_actual": {
            "intercept": round(intercept_actual, 6),
            "intercept_shift": round(intercept_actual - intercept_5050, 6),
            # slope coefficients are unchanged from the 50/50 fit -- the
            # offset regression only ever refits the intercept
            "coefficients": {v: round(float(c), 6) for v, c in zip(var_list, coefs_5050)},
            "calibration": calibration_actual,
            "vif": vif_actual,
            "correlation_matrix": corr_actual,
        },
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _decisions = load_decisions(DECISIONS_NAME)
    _result = run(_params, _decisions)
    if _result["selected"] is None:
        print("Stage 11: no validated model to document (Stage 10 has nothing)")
    else:
        print(f"Stage 11: documenting {_result['selected']['combination_id']}")
        print(f"  observed bad rate={_result['observed_bad_rate']:.4f} target={_result['target_bad_rate']:.4f}")
        print(f"  50/50 intercept={_result['model_5050']['intercept']:.4f}")
        print(
            f"  actual intercept={_result['model_actual']['intercept']:.4f} "
            f"(shift={_result['model_actual']['intercept_shift']:.4f})"
        )
        actual_overall_bad_rate = sum(r["count_bad"] for r in _result["model_actual"]["calibration"]) / sum(
            r["count_bad"] + r["count_good"] for r in _result["model_actual"]["calibration"]
        )
        print(f"  actual-view weighted bad rate check: {actual_overall_bad_rate:.4f} (should ~= target)")
