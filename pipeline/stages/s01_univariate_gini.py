"""Stage 1: Univariate Gini/IV screening.

Ports "1.0 initial univariate ginis.sas": for every candidate variable,
computes a non-missing Gini (raw value, or bad-rate-encoded value for
categoricals, as the score) and a with-missing Gini + IV (fitted
missing-aware score -- see pipeline.common.metrics.fit_missing_aware_score),
then screens variables with the same OR-of-conditions rule as the SAS
var_select macro: any (min_gini AND min_fill_rate) pair from
params.univariate.gini_fillrate_conditions, OR IV >= min_iv. A
missing-gini-uplift sanity check (condition_5 in the SAS macro) can still
veto a pass. The SAS macro's final gate -- an analyst-set "expected_trend"
column reviewed in Univariate analysis I.xls -- becomes a per-variable
override in decisions/univariate_review.yaml (status: auto | keep | drop),
editable from the Streamlit page.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_raw,
    write_artifact,
    write_metrics,
)
from pipeline.common.metrics import (  # noqa: E402
    bad_rate_encode,
    fit_missing_aware_score,
    weighted_gini,
    weighted_woe_iv,
)

STAGE = "01_univariate"
DECISIONS_NAME = "univariate_review"


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _missing_mask(series: pd.Series, dummy_values) -> pd.Series:
    mask = series.isna()
    if dummy_values and not _is_categorical(series):
        mask = mask | series.isin(dummy_values)
    return mask


def compute_variable_metrics(
    x: pd.Series, y: pd.Series, weight: pd.Series, dummy_values: list, uni_params: dict
) -> dict:
    """One variable's screening stats, computed on the development sample."""
    missing = _missing_mask(x, dummy_values)
    population_total = float(weight.sum())
    population_non_missing = float(weight[~missing].sum())
    fill_rate = population_non_missing / population_total if population_total > 0 else 0.0

    is_cat = _is_categorical(x)

    if int((~missing).sum()) >= 2:
        x_nm, y_nm, w_nm = x[~missing], y[~missing], weight[~missing]
        score_nm = bad_rate_encode(x_nm, y_nm, w_nm) if is_cat else x_nm.astype(float)
        gini_non_missing = weighted_gini(y_nm.to_numpy(), score_nm.to_numpy(), w_nm.to_numpy())
    else:
        gini_non_missing = 0.0

    # with-missing score: bad-rate-encode categoricals (missing -> NaN),
    # cast numerics to float with missing/dummy -> NaN, then let
    # fit_missing_aware_score add its own is_missing indicator and fit.
    if is_cat:
        x_encoded = x.astype(object).copy()
        x_encoded[~missing] = bad_rate_encode(x[~missing], y[~missing], weight[~missing])
        x_encoded[missing] = np.nan
        x_for_fit = pd.to_numeric(x_encoded, errors="coerce")
    else:
        x_for_fit = x.astype(float).where(~missing, np.nan)

    pred = fit_missing_aware_score(x_for_fit, y, weight)
    if pred is not None:
        gini_with_missing = weighted_gini(y.to_numpy(), pred, weight.to_numpy())
        iv = weighted_woe_iv(
            y.to_numpy(),
            pred,
            weight.to_numpy(),
            bins=uni_params["iv_bin_count"],
            min_bin_size=uni_params["iv_min_bin_size"],
            min_bins_floor=uni_params["iv_min_bins_floor"],
        )
    else:
        gini_with_missing = 0.0
        iv = 0.0

    return {
        "dtype": "categorical" if is_cat else "numeric",
        "population_total": population_total,
        "population_non_missing": population_non_missing,
        "fill_rate": round(fill_rate, 4),
        "gini_non_missing": round(float(gini_non_missing), 4),
        "gini_with_missing": round(float(gini_with_missing), 4),
        "iv": round(float(iv), 4),
    }


def _auto_pass(stats: dict, uni_params: dict) -> bool:
    max_gini = max(stats["gini_non_missing"], stats["gini_with_missing"])
    fill_rate = stats["fill_rate"]

    condition_pass = any(
        max_gini >= c["min_gini"] and fill_rate >= c["min_fill_rate"]
        for c in uni_params["gini_fillrate_conditions"]
    )
    condition_pass = condition_pass or stats["iv"] >= uni_params["min_iv"]

    uplift = max(stats["gini_with_missing"] - stats["gini_non_missing"], 0.0)
    condition_5 = not (
        uplift > uni_params["max_missing_gini_uplift"]
        and stats["gini_non_missing"] < uni_params["missing_gini_uplift_ceiling"]
    )

    return bool(condition_pass and condition_5)


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    uni_params = params["univariate"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]
    dummy_values = g.get("dummy_values") or []

    dev = read_raw("development")
    val = read_raw("validation")

    candidate_vars = [c for c in dev.columns if c not in (application_no, bad_flag, sample_weight)]
    overrides = decisions.get("variables", {})

    per_variable = {}
    selected_vars = []
    for var in candidate_vars:
        stats = compute_variable_metrics(
            dev[var], dev[bad_flag], dev[sample_weight], dummy_values, uni_params
        )
        auto_pass = _auto_pass(stats, uni_params)
        status = overrides.get(var, {}).get("status", "auto")
        if status == "keep":
            final_pass = True
        elif status == "drop":
            final_pass = False
        else:
            final_pass = auto_pass

        stats["auto_pass"] = auto_pass
        stats["decision_status"] = status
        stats["selected"] = final_pass
        per_variable[var] = stats
        if final_pass:
            selected_vars.append(var)

    keep_cols = [application_no, bad_flag, sample_weight] + selected_vars
    write_artifact(STAGE, "development", dev[keep_cols])
    write_artifact(STAGE, "validation", val[keep_cols])

    metrics = {
        "n_candidate_variables": len(candidate_vars),
        "n_selected_variables": len(selected_vars),
        "selected_variables": selected_vars,
        "variables": per_variable,
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _decisions = load_decisions(DECISIONS_NAME)
    _result = run(_params, _decisions)
    print(f"Stage 1: {_result['n_selected_variables']}/{_result['n_candidate_variables']} variables selected")
    for _v in _result["selected_variables"]:
        _s = _result["variables"][_v]
        print(f"  {_v:24s} gini={_s['gini_with_missing']:.3f} iv={_s['iv']:.3f} fill={_s['fill_rate']:.2f}")
