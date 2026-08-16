"""Stage 4: Categorical variable recoding.

Ports "4.0 char recoding etc.sas". In the original SAS file the frequency-
table and bad-rate diagnostic macros (freqtbl, bad_rate) are commented out,
and the "real" category grouping was done externally via a CHAID export
(also commented out) plus manual analyst review -- as written, the SAS
stage is a straight pass-through from development_num_out to
development_char_final. This port keeps that diagnostic intent alive
directly instead of leaving it dead code: every categorical variable's
per-level weighted population share and bad rate is always computed
(replacing freqtbl/bad_rate), and by default any category below
params.char_recoding.min_category_share gets auto-consolidated into a
single rare_category_bucket_name -- a common, CHAID-adjacent technique for
stabilizing bad-rate estimates on sparse categories. Per-variable overrides
in decisions/char_recoding_rules.yaml (mode: auto | none | custom) replace
the "analyst decided in Excel/CHAID" step: 'none' reproduces the SAS file's
literal pass-through behavior, 'custom' takes an explicit level -> group
name map (see the Streamlit page's drill-in editor).

By this point in the pipeline (after Stage 2's missing-value treatment)
every categorical column is already fully filled, so recoding here is
purely about consolidating existing levels, not handling missingness.
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

STAGE = "04_char_recoding"
SOURCE_STAGE = "03_outlier"
DECISIONS_NAME = "char_recoding_rules"


def _level_diagnostics(x: pd.Series, y: pd.Series, weight: pd.Series) -> list:
    """Weighted population share + bad rate per level, sorted largest-share
    first -- ports the commented-out freqtbl (frequency table) and bad_rate
    (mean bad_flag by class) SAS macros into one always-on diagnostic."""
    df = pd.DataFrame({"level": x.astype(object), "bad_wt": y.to_numpy() * weight.to_numpy(), "wt": weight.to_numpy()})
    grp = df.groupby("level", dropna=False)[["bad_wt", "wt"]].sum()
    total_weight = grp["wt"].sum()
    out = []
    for level, row in grp.sort_values("wt", ascending=False).iterrows():
        out.append(
            {
                "level": level,
                "population_share": round(row["wt"] / total_weight, 4) if total_weight else 0.0,
                "bad_rate": round(row["bad_wt"] / row["wt"], 4) if row["wt"] else 0.0,
            }
        )
    return out


def _auto_group_map(diagnostics: list, min_share: float, bucket_name: str) -> dict:
    return {d["level"]: bucket_name for d in diagnostics if d["population_share"] < min_share}


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    cr_params = params["char_recoding"]
    application_no, bad_flag, sample_weight = g["application_no"], g["bad_flag"], g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    dev_out, val_out = dev.copy(), val.copy()

    candidate_vars = [
        c
        for c in dev.columns
        if c not in (application_no, bad_flag, sample_weight) and not pd.api.types.is_numeric_dtype(dev[c])
    ]
    overrides = decisions.get("variables", {})

    per_variable = {}
    for var in candidate_vars:
        before = _level_diagnostics(dev[var], dev[bad_flag], dev[sample_weight])
        cfg = overrides.get(var, {})
        mode = cfg.get("mode", "auto")

        if mode == "none":
            group_map = {}
        elif mode == "custom":
            group_map = cfg.get("groups", {}) or {}
        else:  # auto
            group_map = _auto_group_map(before, cr_params["min_category_share"], cr_params["rare_category_bucket_name"])

        if group_map:
            dev_out[var] = dev_out[var].astype(object).replace(group_map)
            val_out[var] = val_out[var].astype(object).replace(group_map)

        after = _level_diagnostics(dev_out[var], dev_out[bad_flag], dev_out[sample_weight])

        per_variable[var] = {
            "mode": mode,
            "n_levels_before": len(before),
            "n_levels_after": len(after),
            "group_map": group_map,
            "before": before,
            "after": after,
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
        print(f"  {_v:24s} mode={_s['mode']:8s} levels {_s['n_levels_before']} -> {_s['n_levels_after']}")
