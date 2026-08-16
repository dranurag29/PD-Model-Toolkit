"""Stage 2: Missing-value treatment.

Ports "2.2 basic missing value imputation.sas" -- in the original SAS
pipeline this stage was a stub left for a human to fill in by hand
(`/* PUT MISSING VALUE TREATMENT HERE */`). Here every variable gets an
actual, configurable strategy: mean / median / mode / constant /
flag_category, chosen per-variable in decisions/missing_value_rules.yaml
(falling back to params.missing_value.default_strategy). Fill values are
always computed from the DEVELOPMENT sample only and applied identically to
validation, so validation-sample statistics never leak into the fit.

flag_category on a numeric variable also adds a "<var>__was_missing"
0/1 indicator column alongside the filled value -- the same idea as the
is_missing indicator pipeline.common.metrics.fit_missing_aware_score uses,
so missingness itself can still carry signal downstream instead of being
silently erased by imputation.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    write_artifact,
    write_metrics,
)
from pipeline.common.stats import weighted_mean, weighted_mode, weighted_quantile  # noqa: E402

STAGE = "02_missing"
SOURCE_STAGE = "01_univariate"
DECISIONS_NAME = "missing_value_rules"


def _is_categorical(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def _missing_mask(series: pd.Series, dummy_values: list, is_cat: bool) -> pd.Series:
    mask = series.isna()
    if dummy_values and not is_cat:
        mask = mask | series.isin(dummy_values)
    return mask


def _resolve_strategy(is_cat: bool, requested: str) -> str:
    # mean/median are numeric-only concepts; a categorical falls back to mode.
    if is_cat and requested in ("mean", "median"):
        return "mode"
    return requested


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    mv_params = params["missing_value"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]
    dummy_values = g.get("dummy_values") or []

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    dev_out, val_out = dev.copy(), val.copy()

    candidate_vars = [c for c in dev.columns if c not in (application_no, bad_flag, sample_weight)]
    overrides = decisions.get("variables", {})

    per_variable = {}
    for var in candidate_vars:
        is_cat = _is_categorical(dev[var])
        missing_dev = _missing_mask(dev[var], dummy_values, is_cat)
        missing_val = _missing_mask(val[var], dummy_values, is_cat)
        pct_missing_dev = float(missing_dev.mean())
        pct_missing_val = float(missing_val.mean())

        cfg = overrides.get(var, {})
        strategy = _resolve_strategy(is_cat, cfg.get("strategy", mv_params["default_strategy"]))
        indicator_added = False
        fill_value = None

        if pct_missing_dev == 0 and pct_missing_val == 0:
            strategy = "none"
        elif strategy == "mean":
            fill_value = weighted_mean(dev.loc[~missing_dev, var].astype(float), dev.loc[~missing_dev, sample_weight])
        elif strategy == "median":
            fill_value = weighted_quantile(
                dev.loc[~missing_dev, var].astype(float), dev.loc[~missing_dev, sample_weight], 0.5
            )
        elif strategy == "mode":
            fill_value = weighted_mode(dev.loc[~missing_dev, var], dev.loc[~missing_dev, sample_weight])
        elif strategy == "constant":
            fill_value = cfg.get("constant_value", mv_params["default_constant"])
        elif strategy == "flag_category":
            if is_cat:
                fill_value = "missing"
            else:
                fill_value = cfg.get("constant_value", mv_params["default_constant"])
                indicator_added = True

        if strategy != "none":
            if is_cat:
                dev_out[var] = dev_out[var].astype(object)
                val_out[var] = val_out[var].astype(object)
            else:
                dev_out[var] = dev_out[var].astype(float)
                val_out[var] = val_out[var].astype(float)
            dev_out.loc[missing_dev, var] = fill_value
            val_out.loc[missing_val, var] = fill_value

            if indicator_added:
                ind_col = f"{var}__was_missing"
                dev_out[ind_col] = missing_dev.astype(int)
                val_out[ind_col] = missing_val.astype(int)

        per_variable[var] = {
            "dtype": "categorical" if is_cat else "numeric",
            "strategy": strategy,
            "fill_value": fill_value,
            "pct_missing_dev": round(pct_missing_dev, 4),
            "pct_missing_val": round(pct_missing_val, 4),
            "indicator_added": indicator_added,
        }

    write_artifact(STAGE, "development", dev_out)
    write_artifact(STAGE, "validation", val_out)

    metrics = {"variables": per_variable}
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _decisions = load_decisions(DECISIONS_NAME)
    _result = run(_params, _decisions)
    for _v, _s in _result["variables"].items():
        print(f"  {_v:24s} strategy={_s['strategy']:14s} missing_dev={_s['pct_missing_dev']:.2%}")
