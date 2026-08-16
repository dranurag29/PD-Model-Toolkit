"""Stage 8: Multifactor model search.

Ports "8.0 model development - multifactor model.sas". Brute-force fits a
weighted logistic regression for every combination of "skip this factor
group, or include exactly one of its winning variables" across Stage 7's
factor groups -- i.e. every candidate model with at most one variable per
correlated cluster, avoiding multicollinearity by construction. No
threshold filtering happens here (that's a future model-selection stage's
job, matching the original SAS file's boundary) -- every combination gets
recorded with its Gini, coefficient significance, and sign.

Fitting (and Gini) use a 50/50-rebalanced sample weight, ports SAS's
`multiplier=(100-percent)/percent` reweighting of bad-flag=1 rows so the
weighted bad and good populations become equal -- a standard rare-event
technique for stabilizing logistic regression on an imbalanced target.
Final calibration back to the true population bad rate belongs to a later
stage (SAS's "11.2 ... ACTUAL weights"), not this one.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_params,
    read_artifact,
    read_metrics,
    write_artifact,
    write_metrics,
)
from pipeline.common.metrics import weighted_gini  # noqa: E402
from pipeline.common.stats import rebalance_weight_5050  # noqa: E402

STAGE = "08_multifactor_model"
SOURCE_STAGE = "07_variable_grouping"


def _fit_multivariate_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted multivariate logistic regression bad_flag ~ X (with
    intercept). Returns (params, pvalues, predicted_prob) with the
    intercept first, or None if the fit fails (e.g. perfect separation)."""
    X_design = sm.add_constant(X)
    try:
        result = sm.GLM(y, X_design, family=sm.families.Binomial(), freq_weights=w).fit()
    except Exception:
        return None
    if not result.converged or not np.all(np.isfinite(result.params)):
        return None
    return result.params, result.pvalues, np.asarray(result.predict(X_design))


def _enumerate_combinations(factor_groups: dict, winners: dict, max_combinations: int):
    """Every combination of "skip group, or include one of its winning
    variables", excluding the all-skip combination -- ports SAS's nested
    odometer-counter loop over group_1..group_n (each group_i always offers
    a 0=skip choice, even a singleton group)."""
    group_ids = sorted(factor_groups.keys())
    choices_per_group = [[None] + [winners[v] for v in factor_groups[gid]] for gid in group_ids]

    total_possible = 1
    for choices in choices_per_group:
        total_possible *= len(choices)
    total_possible -= 1  # exclude the all-skip combination

    combos = []
    for combo in itertools.product(*choices_per_group):
        selected = [c for c in combo if c is not None]
        if not selected:
            continue
        combos.append(selected)
        if len(combos) >= max_combinations:
            break
    return combos, total_possible


def run(params: dict) -> dict:
    g = params["global"]
    mm = params["multifactor_model"]
    bad_flag, sample_weight = g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    grouping_metrics = read_metrics(SOURCE_STAGE)
    factor_groups = grouping_metrics["factor_groups"]
    winners = grouping_metrics["winners"]

    y = dev[bad_flag].astype(float).to_numpy()
    w = dev[sample_weight].astype(float).to_numpy()
    rebalanced_w = rebalance_weight_5050(y, w)

    combos, total_possible = _enumerate_combinations(factor_groups, winners, mm["max_combinations"])
    truncated = total_possible > mm["max_combinations"]

    results = []
    for selected_cols in combos:
        X = dev[selected_cols].astype(float).to_numpy()
        fit = _fit_multivariate_logit(X, y, rebalanced_w)
        combination_id = "+".join(sorted(selected_cols))

        if fit is None:
            results.append(
                {
                    "combination_id": combination_id,
                    "variables": selected_cols,
                    "num_factors": len(selected_cols),
                    "gini": 0.0,
                    "significance": None,
                    "sign_correct": False,
                    "converged": False,
                    "coefficients": {},
                }
            )
            continue

        coef_params, pvalues, pred_prob = fit
        gini = weighted_gini(y, pred_prob, rebalanced_w)
        var_estimates = {col: float(coef_params[i + 1]) for i, col in enumerate(selected_cols)}
        var_pvalues = {col: float(pvalues[i + 1]) for i, col in enumerate(selected_cols)}
        significance = max(var_pvalues.values())
        sign_correct = all(v > 0 for v in var_estimates.values())

        results.append(
            {
                "combination_id": combination_id,
                "variables": selected_cols,
                "num_factors": len(selected_cols),
                "gini": round(float(gini), 4),
                "significance": round(float(significance), 4),
                "sign_correct": bool(sign_correct),
                "converged": True,
                "coefficients": {
                    "Intercept": {"estimate": round(float(coef_params[0]), 6), "p_value": round(float(pvalues[0]), 6)},
                    **{
                        col: {"estimate": round(var_estimates[col], 6), "p_value": round(var_pvalues[col], 6)}
                        for col in selected_cols
                    },
                },
            }
        )

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {
        "n_combinations": len(results),
        "n_combinations_possible": total_possible,
        "truncated": truncated,
        "max_combinations": mm["max_combinations"],
        "combinations": results,
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _result = run(_params)
    print(
        f"Stage 8: {_result['n_combinations']}/{_result['n_combinations_possible']} combinations fit"
        + (" (TRUNCATED)" if _result["truncated"] else "")
    )
    _best = sorted(
        (c for c in _result["combinations"] if c["converged"] and c["sign_correct"]),
        key=lambda c: c["gini"],
        reverse=True,
    )[:5]
    for _c in _best:
        print(f"  gini={_c['gini']:.3f} sig={_c['significance']:.3f} vars={_c['variables']}")
