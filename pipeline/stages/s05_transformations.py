"""Stage 5: Numeric variable transformations.

Ports "5.1.1 numeric transformations - test applicable transformation.sas"
(curve-shape testing) and the spirit of "5.2.x" (special/dummy-value
handling). See pipeline.common.curves for the full rationale on which of
SAS's 9 shape classes are kept (linear, quadratic, log_linear, bell,
inverted_bell, s_shape) plus the always-available binning fallback, and
which clamped/piecewise sub-variants were dropped as a scope
simplification.

For every numeric variable: build its weighted log-odds trend (excluding
dummy/special values), fit every shape family applicable to the variable's
expected trend (decisions/transformation_rules.yaml, default "any"), assess
each fit (converged? trend-conforming? significant? enough R square?), and
materialize a new "<var>__<family>" column for every "usable" fit -- in
BOTH development and validation, using only development-derived fit
parameters (and, for dummy-value rows, development's own dummy-subgroup
mean log-odds -- ports 5.2.x's separate treatment of special codes without
needing its generate/generated/implement code-string dance). The original
variable column is kept unchanged; transformations add candidate columns,
they don't replace anything. Categorical columns pass through untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.curves import (  # noqa: E402
    FIT_FUNCS,
    FitResult,
    applicable_families,
    assess,
    evaluate,
    fit_binning,
    significance,
    weighted_r_squared,
)
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    write_artifact,
    write_metrics,
)
from pipeline.common.stats import weighted_trend_buckets  # noqa: E402

STAGE = "05_transformations"
SOURCE_STAGE = "04_char_recoding"
DECISIONS_NAME = "transformation_rules"


def _fit_family(family: str, bucket_x, bucket_y, bucket_w, raw_x, raw_y, raw_w, min_x, max_x, tp: dict) -> FitResult:
    if family == "binning":
        return fit_binning(raw_x, raw_y, raw_w, n_bins=tp["binning_n_bins"], min_bin_size=tp["binning_min_bin_size"])
    return FIT_FUNCS[family](bucket_x, bucket_y, bucket_w, min_x, max_x)


def _process_variable(dev: pd.DataFrame, val: pd.DataFrame, var: str, bad_flag: str, sample_weight: str,
                       dummy_values: list, tp: dict, cfg: dict) -> dict:
    dev_x = dev[var].astype(float)
    dummy_mask = dev_x.isin(dummy_values) if dummy_values else pd.Series(False, index=dev_x.index)
    normal = ~dummy_mask

    raw_x = dev_x[normal].to_numpy()
    raw_y = dev[bad_flag][normal].astype(float).to_numpy()
    raw_w = dev[sample_weight][normal].astype(float).to_numpy()
    min_x, max_x = float(raw_x.min()), float(raw_x.max())

    dummy_log_odds = None
    n_dummy = int(dummy_mask.sum())
    if n_dummy > 0:
        d_bad = dev[bad_flag][dummy_mask].astype(float)
        d_wt = dev[sample_weight][dummy_mask].astype(float)
        dummy_log_odds = float(np.log(((d_bad * d_wt).sum() + 1) / (((1 - d_bad) * d_wt).sum() + 1)))

    buckets = weighted_trend_buckets(
        raw_x, raw_y, raw_w, n_buckets=tp["n_trend_buckets"], min_bucket_size=tp["min_trend_bucket_size"]
    )
    bucket_x, bucket_y, bucket_w = buckets["x"].to_numpy(), buckets["ln_odd"].to_numpy(), buckets["weight"].to_numpy()

    expected_trend = cfg.get("expected_trend", "any")
    families = applicable_families(expected_trend, min_x)
    if cfg.get("allowed_models"):
        families = [f for f in families if f in cfg["allowed_models"]]

    attempts = []
    val_x = val[var].astype(float)
    val_dummy_mask = val_x.isin(dummy_values) if dummy_values else pd.Series(False, index=val_x.index)

    for family in families:
        result = _fit_family(family, bucket_x, bucket_y, bucket_w, raw_x, raw_y, raw_w, min_x, max_x, tp)
        if result.converged:
            pred_at_buckets = evaluate(result, bucket_x)
            r_squared = weighted_r_squared(bucket_y, pred_at_buckets, bucket_w)
        else:
            r_squared = 0.0
        verdict = assess(result, expected_trend, tp["significance_level"], tp["r_square_level"], r_squared, min_x, max_x)
        usable = verdict == "usable"

        extra = {}
        if family == "binning" and result.converged:
            extra = {"edges": result.extra["edges"], "log_odds": result.extra["log_odds"]}

        attempts.append(
            {
                "family": family,
                "converged": result.converged,
                "r_squared": round(float(r_squared), 4),
                "significance": (lambda s: round(float(s), 4) if s is not None else None)(significance(result)),
                "assessment": verdict,
                "usable": usable,
                "params": {k: round(float(v), 6) for k, v in result.params.items()},
                "extra": extra,
            }
        )

        if usable:
            col = f"{var}__{family}"
            dev_col = evaluate(result, dev_x.to_numpy())
            val_col = evaluate(result, val_x.to_numpy())
            if n_dummy > 0:
                dev_col = np.where(dummy_mask.to_numpy(), dummy_log_odds, dev_col)
                val_col = np.where(val_dummy_mask.to_numpy(), dummy_log_odds, val_col)
            dev[col] = dev_col
            val[col] = val_col

    return {
        "min_x": min_x,
        "max_x": max_x,
        "expected_trend": expected_trend,
        "n_dummy_dev": n_dummy,
        "dummy_log_odds": dummy_log_odds,
        "trend_buckets": {
            "x": [round(float(v), 4) for v in bucket_x],
            "ln_odd": [round(float(v), 4) for v in bucket_y],
            "weight": [round(float(v), 2) for v in bucket_w],
        },
        "attempts": attempts,
        "materialized_columns": [a["family"] for a in attempts if a["usable"]],
    }


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    tp = params["transformations"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]
    dummy_values = g.get("dummy_values") or []

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    dev_out, val_out = dev.copy(), val.copy()

    candidate_vars = [
        c
        for c in dev.columns
        if c not in (application_no, bad_flag, sample_weight) and pd.api.types.is_numeric_dtype(dev[c])
    ]
    overrides = decisions.get("variables", {})

    per_variable = {}
    for var in candidate_vars:
        cfg = overrides.get(var, {})
        per_variable[var] = _process_variable(dev_out, val_out, var, bad_flag, sample_weight, dummy_values, tp, cfg)

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
        _usable = [a["family"] for a in _s["attempts"] if a["usable"]]
        print(f"  {_v:24s} usable={','.join(_usable) or '(none)'}")
