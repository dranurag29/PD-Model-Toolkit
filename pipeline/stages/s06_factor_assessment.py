"""Stage 6: Factor assessment.

Ports "6.0 factor assessment.sas". For every NUMERIC column in the Stage 5
output -- both the original untransformed variables and every
"<var>__<family>" transformed candidate Stage 5 materialized -- fits a
weighted univariate logistic regression (bad_flag ~ variable) and records
its Gini, coefficient p-value, and sign. This mirrors SAS's `type=1` filter
on development_num_final; char-recoded categorical columns are string-typed
and are not carried into the numeric factor-assessment/model-building
stages (this pipeline doesn't WOE-numericize them anywhere, matching what
"6.0" itself does -- it only ever looks at type=1 variables).

Combined with Stage 5's R square (looked up directly from its metrics.json
-- no need for SAS's code-string-parsing "add_R2" step, since Stage 5
already stores R square as a plain number), each candidate gets an
adjusted_gini (a small bonus for simpler/more standard transformation
classes -- linear, S-shape, and binning -- mirroring the SAS 1.05/1.04/1.03
multipliers for transformation classes A/I/J) and a transformation_usable
verdict (wrong-signed coefficient, weak significance, or low Gini all
disqualify it).

This stage only SCORES every candidate; picking the single best
transformation per base variable (via correlation-based grouping) is
"7.0"/"7.1"'s job, not this one -- matching the original SAS file boundary.
The output dataset is an unchanged passthrough of Stage 5's -- like the SAS
original, this stage only produces a diagnostics table, not a new dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

STAGE = "06_factor_assessment"
SOURCE_STAGE = "05_transformations"

ADJUSTED_GINI_BONUS = {"linear": 1.05, "s_shape": 1.04, "binning": 1.03}


def _split_variable_name(column: str):
    """"<var>__<family>" -> (var, family); a raw untransformed column has no
    "__" separator and returns (column, None)."""
    if "__" in column:
        var, family = column.rsplit("__", 1)
        return var, family
    return column, None


def _fit_univariate_logit(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted univariate logistic regression bad_flag ~ x. Ports SAS's
    `proc logistic ... model bad_flag(event='1')=var; weight sample_weight;`
    -- using statsmodels' freq_weights rather than replicating SAS's WEIGHT
    statement, which doesn't rescale standard errors for the effective
    sample size, so this gives statistically correct p-values for a
    weighted fit instead of SAS's known quirk there.

    Drops the intercept when the variable has <=2 distinct values (a 0/1
    flag), matching SAS's `/ noint` branch. Returns
    (coefficient, p_value, predicted_prob), or None if the fit fails (e.g.
    perfect separation)."""
    n_unique = pd.unique(x).size
    X = x.reshape(-1, 1)
    if n_unique > 2:
        X = sm.add_constant(X)
    try:
        result = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()
    except Exception:
        return None
    if not result.converged or not np.all(np.isfinite(result.params)):
        return None
    return float(result.params[-1]), float(result.pvalues[-1]), np.asarray(result.predict(X))


def _lookup_r_squared(transformations_metrics: dict, var: str, family):
    if family is None:
        return None
    attempts = transformations_metrics.get("variables", {}).get(var, {}).get("attempts", [])
    for a in attempts:
        if a["family"] == family:
            return a["r_squared"]
    return None


def run(params: dict) -> dict:
    g = params["global"]
    fa = params["factor_assessment"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    transformations_metrics = read_metrics(SOURCE_STAGE)

    candidate_cols = [
        c
        for c in dev.columns
        if c not in (application_no, bad_flag, sample_weight) and pd.api.types.is_numeric_dtype(dev[c])
    ]

    y = dev[bad_flag].astype(float).to_numpy()
    w = dev[sample_weight].astype(float).to_numpy()

    per_candidate = {}
    for col in candidate_cols:
        var, family = _split_variable_name(col)
        x = dev[col].astype(float).to_numpy()

        fit = _fit_univariate_logit(x, y, w)
        if fit is None:
            gini, significance, sign_correct = 0.0, None, False
        else:
            coef, significance, pred_prob = fit
            gini = weighted_gini(y, pred_prob, w)
            sign_correct = coef > 0

        r_squared = _lookup_r_squared(transformations_metrics, var, family)
        bonus = ADJUSTED_GINI_BONUS.get(family, 1.0)
        adjusted_gini = min(np.ceil(gini * bonus * 100) / 100, 1.0)
        adjusted_r_squared = min(np.ceil(r_squared * 100) / 100, 1.0) if r_squared is not None else None

        usable = True
        if not sign_correct:
            usable = False
        if significance is None or np.floor(significance * 100) / 100 >= fa["significance_level"]:
            usable = False
        if np.floor(gini * 100) / 100 <= fa["gini_level"]:
            usable = False

        per_candidate[col] = {
            "actual_variable": var,
            "transformation_family": family,
            "gini": round(float(gini), 4),
            "significance": round(float(significance), 4) if significance is not None else None,
            "sign_correct": bool(sign_correct),
            "r_squared": r_squared,
            "adjusted_gini": round(float(adjusted_gini), 4),
            "adjusted_r_squared": round(float(adjusted_r_squared), 4) if adjusted_r_squared is not None else None,
            "usable": usable,
        }

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {"candidates": per_candidate}
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _result = run(_params)
    _candidates = _result["candidates"]
    _usable = sum(1 for c in _candidates.values() if c["usable"])
    print(f"Stage 6: {_usable}/{len(_candidates)} candidate columns usable")
    _base_vars = sorted({c["actual_variable"] for c in _candidates.values()})
    for _v in _base_vars:
        _rows = [(k, c) for k, c in _candidates.items() if c["actual_variable"] == _v and c["usable"]]
        _rows.sort(key=lambda kv: kv[1]["adjusted_gini"], reverse=True)
        _best = _rows[0] if _rows else None
        _label = f"{_best[0]} (adj_gini={_best[1]['adjusted_gini']:.3f})" if _best else "(none usable)"
        print(f"  {_v:24s} best={_label}")
