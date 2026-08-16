import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import correlation_heatmap, run_stage  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Variable Grouping", layout="wide")
st.title("Stage 7 -- Variable Grouping")
st.caption(
    "Ports \"7.0\"/\"7.1\". Picks the single best usable transformed column per "
    "base variable (same ranking Stage 6 scores by), then groups the winners into "
    "non-overlapping clusters via connected components of the correlation graph -- "
    "variables in the same group are mutually correlated and shouldn't both enter "
    "a model together. A later stage picks at most one variable per group."
)

params = load_params()
vg = params["variable_grouping"]

st.subheader("Parameters")
vg["corr_threshold"] = st.number_input(
    "corr_threshold", value=float(vg["corr_threshold"]), min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
    help="Two winning variables are placed in the same factor group if |correlation| >= this.",
)

if st.button("Save parameters"):
    params["variable_grouping"] = vg
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro variable_grouping..."):
        result = run_stage("variable_grouping")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("07_variable_grouping")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

decisions = load_decisions("variable_grouping_rules")
overrides = decisions.get("variables", {})

m1, m2 = st.columns(2)
m1.metric("winning variables", len(metrics["winners"]))
m2.metric("factor groups", len(metrics["factor_groups"]))

st.subheader("Winners and groups")
rows = [
    {
        "variable": var,
        "winning_column": col,
        "group": metrics["variable_to_group"].get(var, "(excluded)"),
        "excluded": overrides.get(var, {}).get("mode") == "exclude",
    }
    for var, col in metrics["winners"].items()
]
for var in metrics["excluded_variables"]:
    rows.append({"variable": var, "winning_column": "(excluded)", "group": "(excluded)", "excluded": True})
results_df = pd.DataFrame(rows).sort_values(["group", "variable"])

st.markdown("Check `excluded` to force-drop a variable from grouping entirely (its next-best "
            "transformation won't be considered either -- the variable is dropped outright).")
edited = st.data_editor(
    results_df,
    key="variable_grouping_editor",
    width="stretch",
    disabled=["variable", "winning_column", "group"],
    column_config={"excluded": st.column_config.CheckboxColumn()},
)

if st.button("Save decisions"):
    new_overrides = {}
    for row in edited.to_dict("records"):
        if row["excluded"]:
            new_overrides[row["variable"]] = {"mode": "exclude"}
    save_decisions("variable_grouping_rules", {"variables": new_overrides})
    st.success("Saved to decisions/variable_grouping_rules.yaml. Re-run the stage to apply.")

st.divider()
st.subheader("Correlation matrix")
if metrics["correlation_matrix"]:
    st.plotly_chart(correlation_heatmap(metrics["correlation_matrix"]), width="stretch", key="corr_heatmap")
else:
    st.info("No winning variables to correlate.")
