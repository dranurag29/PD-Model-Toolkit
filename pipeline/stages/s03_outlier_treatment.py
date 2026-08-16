"""Stage 3: Outlier treatment.

Ports "3.0 descriptives for outlier treatment.sas" + the
generate-rules/generated-rules/implement-rules chain in
"3.1.1"/"3.1.2"/"3.1.3 num outlier treatment ....sas". The SAS version
computed descriptive percentiles, had an analyst set cap/drop rules in the
"Num Outlier" sheet of Univariate analysis I.xls, then generated and ran SAS
code from those rules. Here the percentile suggestion and the rule
application happen in the same script; the analyst's decision lives in
decisions/outlier_rules.yaml (status: none | cap | drop per variable,
editable from the Streamlit page), and if `cap` is chosen without an
explicit lower_cap/upper_cap the suggested percentile bounds
(params.outlier.suggest_lower_percentile/suggest_upper_percentile) are used
automatically.
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
from pipeline.common.stats import weighted_quantile  # noqa: E402

STAGE = "03_outlier"
SOURCE_STAGE = "02_missing"
DECISIONS_NAME = "outlier_rules"


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    out_params = params["outlier"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]
    dummy_values = g.get("dummy_values") or []

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    dev_out, val_out = dev.copy(), val.copy()

    candidate_vars = [c for c in dev.columns if c not in (application_no, bad_flag, sample_weight)]
    overrides = decisions.get("variables", {})

    per_variable = {}
    dropped_vars = []

    for var in candidate_vars:
        is_numeric = pd.api.types.is_numeric_dtype(dev[var])
        cfg = overrides.get(var, {})
        status = cfg.get("status", out_params["default_strategy"]) if is_numeric else "none"

        suggested_lower = suggested_upper = None
        if is_numeric:
            values = dev[var].astype(float)
            weights = dev[sample_weight].astype(float)
            mask = ~(values.isna() | values.isin(dummy_values))
            if int(mask.sum()) > 0:
                suggested_lower = weighted_quantile(
                    values[mask], weights[mask], out_params["suggest_lower_percentile"] / 100
                )
                suggested_upper = weighted_quantile(
                    values[mask], weights[mask], out_params["suggest_upper_percentile"] / 100
                )

        applied_lower = applied_upper = None
        if status == "drop":
            dropped_vars.append(var)
        elif status == "cap":
            applied_lower = cfg.get("lower_cap", suggested_lower)
            applied_upper = cfg.get("upper_cap", suggested_upper)
            if applied_lower is not None:
                dev_out[var] = dev_out[var].clip(lower=applied_lower)
                val_out[var] = val_out[var].clip(lower=applied_lower)
            if applied_upper is not None:
                dev_out[var] = dev_out[var].clip(upper=applied_upper)
                val_out[var] = val_out[var].clip(upper=applied_upper)

        per_variable[var] = {
            "dtype": "numeric" if is_numeric else "categorical",
            "status": status,
            "suggested_lower_cap": round(suggested_lower, 2) if suggested_lower is not None else None,
            "suggested_upper_cap": round(suggested_upper, 2) if suggested_upper is not None else None,
            "applied_lower_cap": round(applied_lower, 2) if applied_lower is not None else None,
            "applied_upper_cap": round(applied_upper, 2) if applied_upper is not None else None,
        }

    keep_cols = [c for c in dev_out.columns if c not in dropped_vars]
    write_artifact(STAGE, "development", dev_out[keep_cols])
    write_artifact(STAGE, "validation", val_out[keep_cols])

    metrics = {"variables": per_variable, "dropped_variables": dropped_vars}
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _decisions = load_decisions(DECISIONS_NAME)
    _result = run(_params, _decisions)
    for _v, _s in _result["variables"].items():
        print(f"  {_v:24s} status={_s['status']:6s} lower={_s['applied_lower_cap']} upper={_s['applied_upper_cap']}")
    if _result["dropped_variables"]:
        print(f"  dropped: {_result['dropped_variables']}")
