"""Stage 7: Variable grouping.

Ports "7.0 model development - variable selection and grouping.sas" and
"7.1 final variable grouping - with var list provided.sas".

Step 1 (7.0's factor_list macro, re-derived here rather than reused from
Stage 6 since 6.0 deliberately only *scores* candidates -- matching the
original SAS file boundary): pick the single best USABLE transformed column
per base variable, ranked by adjusted_gini then adjusted_r_squared, same
sort key SAS uses. An analyst can force-exclude a base variable entirely
via decisions/variable_grouping_rules.yaml (mode: exclude), replacing the
manually-curated var_list "7.1" expects to already be given.

Step 2 (7.0's get_factor_loading + 7.1's create_factor_groups): build a
weighted correlation matrix across the winners, then group them into
non-overlapping clusters via connected components of the correlation graph
(edge = |corr| >= corr_threshold). SAS instead iteratively re-runs
PROC FACTOR with a shrinking factor count until the resulting loadings
happen to keep every correlated pair in the same rotated factor -- a
trial-and-error search for exactly the invariant connected components
gives directly and provably minimally, so this port skips the search
loop and the factor-rotation machinery entirely.

A future Stage 8 (multifactor model search) is expected to pick at most one
variable per factor group when building candidate models, avoiding
multicollinearity -- this stage only produces that grouping, it doesn't
build any model itself, matching the original SAS file boundary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    read_metrics,
    write_artifact,
    write_metrics,
)
from pipeline.common.stats import weighted_corr_matrix  # noqa: E402

STAGE = "07_variable_grouping"
SOURCE_STAGE = "06_factor_assessment"
DECISIONS_NAME = "variable_grouping_rules"


def _pick_winners(candidates: dict, excluded: set) -> dict:
    """One winning column per base variable -- ports 7.0/7.1's
    selected_factors derivation: among usable candidates for a base
    variable, the one with the highest adjusted_gini (ties broken by
    adjusted_r_squared), matching the SAS sort key order."""
    by_var = {}
    for col, c in candidates.items():
        if not c["usable"] or c["actual_variable"] in excluded:
            continue
        by_var.setdefault(c["actual_variable"], []).append((col, c))

    winners = {}
    for var, rows in by_var.items():
        rows.sort(key=lambda kv: (kv[1]["adjusted_gini"], kv[1]["adjusted_r_squared"] or 0.0), reverse=True)
        winners[var] = rows[0][0]
    return winners


def _group_by_correlation(corr: pd.DataFrame, threshold: float) -> dict:
    cols = list(corr.columns)
    if len(cols) <= 1:
        return {0: cols} if cols else {}
    adjacency = (corr.abs().to_numpy() >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    groups: dict = {}
    for col, label in zip(cols, labels):
        groups.setdefault(int(label), []).append(col)
    return groups


def run(params: dict, decisions: dict) -> dict:
    g = params["global"]
    vg = params["variable_grouping"]
    sample_weight = g["sample_weight"]

    dev = read_artifact(SOURCE_STAGE, "development")
    val = read_artifact(SOURCE_STAGE, "validation")
    factor_metrics = read_metrics(SOURCE_STAGE)
    candidates = factor_metrics["candidates"]

    overrides = decisions.get("variables", {})
    excluded = {var for var, cfg in overrides.items() if cfg.get("mode") == "exclude"}

    winners = _pick_winners(candidates, excluded)
    winning_cols = list(winners.values())

    corr = weighted_corr_matrix(dev, winning_cols, dev[sample_weight]) if winning_cols else pd.DataFrame()
    groups_by_col = _group_by_correlation(corr, vg["corr_threshold"])

    col_to_var = {col: var for var, col in winners.items()}
    groups_by_var = {
        str(gid): sorted(col_to_var[col] for col in cols) for gid, cols in groups_by_col.items()
    }
    var_to_group = {var: gid for gid, vars_ in groups_by_var.items() for var in vars_}

    write_artifact(STAGE, "development", dev)
    write_artifact(STAGE, "validation", val)

    metrics = {
        "corr_threshold": vg["corr_threshold"],
        "excluded_variables": sorted(excluded),
        "winners": winners,
        "variable_to_group": var_to_group,
        "factor_groups": groups_by_var,
        "correlation_matrix": {a: {b: round(float(corr.loc[a, b]), 4) for b in corr.columns} for a in corr.index},
    }
    write_metrics(STAGE, metrics)
    return metrics


if __name__ == "__main__":
    _params = load_params()
    _decisions = load_decisions(DECISIONS_NAME)
    _result = run(_params, _decisions)
    print(f"Stage 7: {len(_result['winners'])} winning variables -> {len(_result['factor_groups'])} factor groups")
    for _gid, _vars in _result["factor_groups"].items():
        print(f"  group {_gid}: {_vars}")
    if _result["excluded_variables"]:
        print(f"  excluded: {_result['excluded_variables']}")
