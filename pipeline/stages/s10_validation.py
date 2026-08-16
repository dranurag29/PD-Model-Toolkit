"""Stage 10: Bootstrap validation.

Ports "10.0 model validation.sas". For every Stage 9 finalist: fits the
model on a 50/50-rebalanced development sample (same rebalancing formula
as Stage 8), then SCORES the untouched validation set -- the first stage
in this pipeline to use validation data for anything beyond passthrough.
Draws num_bootstrap_samples independent bootstrap replicates from the
scored validation set (each an unweighted simple random sample of
sample_rate% of validation rows, no replacement, matching
`proc surveyselect method=srs`), and for each replicate: re-derives a
fresh 50/50 rebalancing weight specific to that replicate's resampled bad
rate, computes Gini using the already-scored predictions under that
weighting, and refits the model from scratch on just that replicate's rows
to get a fresh set of coefficient estimates. Averaging Gini and each
coefficient across all replicates (and their coefficient of variation)
measures how stable the model actually is outside the exact development
sample it was fit on -- a materially more meaningful validation signal
than any single train/test split.

Bootstrap refits use scikit-learn's LogisticRegression rather than
statsmodels: only point estimates are needed for stability statistics (not
p-values), and sklearn's solver is meaningfully faster across the hundreds
of refits this stage performs per candidate model.

random_seed is not in the SAS original -- makes the bootstrap draw
reproducible given a fixed params.yaml, consistent with this being a DVC
pipeline (an unseeded draw would make `comparison` un-reproducible run to
run, undermining the point of tracking it as a DVC output).
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

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

STAGE = "10_validation"
SOURCE_STAGE = "09_model_selection"


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted logistic regression via sklearn -- point estimates only, no
    p-values (this stage doesn't need them). Returns (intercept, coefs), or
    None if the fit fails (e.g. perfect separation, a degenerate replicate
    with only one class present)."""
    if len(np.unique(y)) < 2:
        return None
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y, sample_weight=w)
    except Exception:
        return None
    return float(model.intercept_[0]), model.coef_[0].astype(float)


def _validate_model(m: dict, dev, val, y_dev, dev_rebalanced_w, y_val, w_val, rng, n_samples, replicate_size):
    var_list = m["variables"]
    X_dev = dev[var_list].astype(float).to_numpy()
    fit = _fit_logistic(X_dev, y_dev, dev_rebalanced_w)
    if fit is None:
        return None
    intercept, coefs = fit

    X_val = val[var_list].astype(float).to_numpy()
    pred_prob_val = 1 / (1 + np.exp(-(intercept + X_val @ coefs)))
    n_val = len(val)

    replicate_ginis = []
    replicate_coefs = {v: [] for v in var_list}
    replicate_intercepts = []
    for _ in range(n_samples):
        idx = rng.choice(n_val, size=replicate_size, replace=False)
        y_r, w_r, pred_r = y_val[idx], w_val[idx], pred_prob_val[idx]
        rw_r = rebalance_weight_5050(y_r, w_r)

        replicate_ginis.append(weighted_gini(y_r, pred_r, rw_r))

        refit = _fit_logistic(X_val[idx], y_r, rw_r)
        if refit is None:
            continue
        r_intercept, r_coefs = refit
        replicate_intercepts.append(r_intercept)
        for i, v in enumerate(var_list):
            replicate_coefs[v].append(r_coefs[i])

    gini_arr = np.array(replicate_ginis)
    val_gini = float(gini_arr.mean())
    coeff_of_var_gini = float(abs(gini_arr.std() / val_gini)) if val_gini != 0 else None

    rows = []
    sign_correct = True
    max_cv = 0.0
    for r in m["rows"]:
        v = r["variable"]
        arr = np.array(replicate_intercepts if v == "Intercept" else replicate_coefs[v])
        if len(arr) == 0:
            rows.append(
                {"variable": v, "dev_estimate": r["estimate"], "dev_factor_weight": r["factor_weight"], "val_estimate": None, "coeff_of_var_estimate": None}
            )
            continue
        mean_estimate = float(arr.mean())
        cv = float(abs(arr.std() / mean_estimate)) if mean_estimate != 0 else None
        if v != "Intercept":
            if mean_estimate < 0:
                sign_correct = False
            if cv is not None:
                max_cv = max(max_cv, cv)
        rows.append(
            {
                "variable": v,
                "dev_estimate": r["estimate"],
                "dev_factor_weight": r["factor_weight"],
                "val_estimate": round(mean_estimate, 6),
                "coeff_of_var_estimate": round(cv, 4) if cv is not None else None,
            }
        )

    return {
        "combination_id": m["combination_id"],
        "variables": var_list,
        "num_factors": m["num_factors"],
        "dev_gini": m["gini"],
        "dev_significance": m["significance"],
        "val_gini": round(val_gini, 4),
        "coeff_of_var_gini": round(coeff_of_var_gini, 4) if coeff_of_var_gini is not None else None,
        "val_sign_correct": sign_correct,
        "max_cv": round(max_cv, 4),
        "rows": rows,
    }


def run(params: dict) -> dict:
    g = params["global"]
    vp = params["validation"]
    bad_flag, sample_weight = g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    selection_metrics = read_metrics(SOURCE_STAGE)
    models = selection_metrics["models"]

    results = []
    if models:
        rng = np.random.default_rng(vp["random_seed"])
        n_samples = vp["num_bootstrap_samples"]
        replicate_size = int(round(len(val) * vp["sample_rate"] / 100))

        y_dev = dev[bad_flag].astype(float).to_numpy()
        w_dev = dev[sample_weight].astype(float).to_numpy()
        dev_rebalanced_w = rebalance_weight_5050(y_dev, w_dev)

        y_val = val[bad_flag].astype(float).to_numpy()
        w_val = val[sample_weight].astype(float).to_numpy()

        for m in models:
            result = _validate_model(m, dev, val, y_dev, dev_rebalanced_w, y_val, w_val, rng, n_samples, replicate_size)
            if result is not None:
                results.append(result)

    results.sort(key=lambda r: r["val_gini"], reverse=True)

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {
        "num_bootstrap_samples": vp["num_bootstrap_samples"],
        "sample_rate": vp["sample_rate"],
        "n_models_validated": len(results),
        "models": results,
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _result = run(_params)
    print(f"Stage 10: {_result['n_models_validated']} models bootstrap-validated")
    for _m in _result["models"][:5]:
        print(
            f"  dev_gini={_m['dev_gini']:.3f} val_gini={_m['val_gini']:.3f} "
            f"cv_gini={_m['coeff_of_var_gini']} max_cv={_m['max_cv']} vars={_m['variables']}"
        )
