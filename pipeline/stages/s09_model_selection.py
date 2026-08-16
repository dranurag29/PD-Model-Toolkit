"""Stage 9: Model selection.

Ports "9.0 model development - select models.sas". Filters Stage 8's
brute-force combination leaderboard down to real finalists: sign_correct,
significance <= level_significant, num_factors >= level_num_factors, and
gini > level_gini. For each survivor, computes every variable's
"contribution" -- its coefficient times the difference between its
weighted mean among bads and weighted mean among goods, both computed on a
freshly 50/50-rebalanced sample -- and normalizes those into a
factor_weight per variable that sums to 1 within a combination: a rough
measure of how much of the model's discriminating power each variable is
pulling. Combinations where the max per-variable weight exceeds
level_max_wt (one variable dominating, or a sign flip elsewhere from
Simpson's-paradox-style suppression pushing another variable's share past
100%) or the min weight falls below level_min_wt (statistically
significant but practically negligible) are dropped too.

The 50/50 rebalancing formula here is SAS's exact wording from this file,
which differs subtly from Stage 8's: bad rows get weight set literally to
the multiplier (not multiplied by their own original weight), good rows
get weight set literally to 1 (not kept at their original weight). For
this pipeline's synthetic data (sample_weight always 1) the two formulas
are numerically identical; ported as written for fidelity to a general
weighted sample where they could differ.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_params,
    read_artifact,
    read_metrics,
    write_artifact,
    write_metrics,
)

STAGE = "09_model_selection"
SOURCE_STAGE = "08_multifactor_model"


def _rebalanced_weight(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """SAS's exact stage-9 rebalancing: bad rows -> the multiplier (a
    constant), good rows -> 1 (a constant) -- see module docstring."""
    bad_wt = (y * w).sum()
    good_wt = ((1 - y) * w).sum()
    if bad_wt <= 0:
        return np.ones_like(y)
    multiplier = good_wt / bad_wt
    return np.where(y == 1, multiplier, 1.0)


def _weighted_mean_by_class(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    mask0, mask1 = y == 0, y == 1
    mean0 = float(np.average(x[mask0], weights=w[mask0])) if mask0.any() else 0.0
    mean1 = float(np.average(x[mask1], weights=w[mask1])) if mask1.any() else 0.0
    return mean0, mean1


def _passes_thresholds(c: dict, ms: dict) -> bool:
    return (
        c["converged"]
        and c["sign_correct"]
        and c["significance"] is not None
        and c["significance"] <= ms["level_significant"]
        and c["num_factors"] >= ms["level_num_factors"]
        and c["gini"] > ms["level_gini"]
    )


def run(params: dict) -> dict:
    g = params["global"]
    ms = params["model_selection"]
    bad_flag, sample_weight = g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    mf_metrics = read_metrics(SOURCE_STAGE)
    combinations = mf_metrics["combinations"]

    survivors = [c for c in combinations if _passes_thresholds(c, ms)]

    models = []
    if survivors:
        all_vars = sorted({v for c in survivors for v in c["variables"]})

        y = dev[bad_flag].astype(float).to_numpy()
        w = dev[sample_weight].astype(float).to_numpy()
        rebalanced_w = _rebalanced_weight(y, w)

        means = {v: _weighted_mean_by_class(dev[v].astype(float).to_numpy(), y, rebalanced_w) for v in all_vars}

        for c in survivors:
            rows = []
            total_contribution = 0.0
            for v in c["variables"]:
                estimate = c["coefficients"][v]["estimate"]
                mean0, mean1 = means[v]
                contribution = estimate * (mean1 - mean0)
                rows.append(
                    {
                        "variable": v,
                        "estimate": estimate,
                        "p_value": c["coefficients"][v]["p_value"],
                        "contribution": contribution,
                    }
                )
                total_contribution += contribution

            intercept = c["coefficients"].get("Intercept")
            if intercept is not None:
                # contribution is 0 by convention (mean_0=mean_1=0 for the
                # intercept), so it never affects total_contribution/weights
                rows.append(
                    {"variable": "Intercept", "estimate": intercept["estimate"], "p_value": intercept["p_value"], "contribution": 0.0}
                )

            if total_contribution == 0:
                continue  # degenerate: can't normalize into weights

            for r in rows:
                r["factor_weight"] = r["contribution"] / total_contribution

            non_intercept_weights = [r["factor_weight"] for r in rows if r["variable"] != "Intercept"]
            max_wt, min_wt = max(non_intercept_weights), min(non_intercept_weights)

            if max_wt <= ms["level_max_wt"] and min_wt >= ms["level_min_wt"]:
                models.append(
                    {
                        "combination_id": c["combination_id"],
                        "variables": c["variables"],
                        "num_factors": c["num_factors"],
                        "gini": c["gini"],
                        "significance": c["significance"],
                        "max_wt": round(float(max_wt), 4),
                        "min_wt": round(float(min_wt), 4),
                        "rows": [
                            {
                                "variable": r["variable"],
                                "estimate": round(float(r["estimate"]), 6),
                                "p_value": r["p_value"],
                                "contribution": round(float(r["contribution"]), 6),
                                "factor_weight": round(float(r["factor_weight"]), 4),
                            }
                            for r in rows
                        ],
                    }
                )

    models.sort(key=lambda m: m["gini"], reverse=True)

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {
        "n_candidates": len(combinations),
        "n_after_thresholds": len(survivors),
        "n_selected": len(models),
        "models": models,
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _result = run(_params)
    print(
        f"Stage 9: {_result['n_candidates']} candidates -> {_result['n_after_thresholds']} passed thresholds "
        f"-> {_result['n_selected']} selected"
    )
    for _m in _result["models"][:5]:
        print(f"  gini={_m['gini']:.3f} max_wt={_m['max_wt']:.2f} min_wt={_m['min_wt']:.2f} vars={_m['variables']}")
