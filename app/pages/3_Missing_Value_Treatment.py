import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage, variable_trend_chart  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Missing-Value Treatment", layout="wide")
st.title("Stage 2 -- Missing-Value Treatment")
st.caption(
    "Ports \"2.2 basic missing value imputation.sas\" -- a stub in the original "
    "pipeline, left for a human to fill in by hand. Here every variable gets a "
    "real strategy, chosen per-variable below (falling back to the default). "
    "Fill values are computed from development only and applied identically to "
    "validation."
)

params = load_params()
mv = params["missing_value"]

st.subheader("Default parameters")
p1, p2 = st.columns(2)
with p1:
    mv["default_strategy"] = st.selectbox(
        "default_strategy",
        options=["mean", "median", "mode", "constant", "flag_category"],
        index=["mean", "median", "mode", "constant", "flag_category"].index(mv["default_strategy"]),
    )
with p2:
    mv["default_constant"] = st.number_input("default_constant", value=float(mv["default_constant"]))

if st.button("Save parameters"):
    params["missing_value"] = mv
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro missing_value_treatment..."):
        result = run_stage("missing_value_treatment")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("02_missing")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

st.subheader("Per-variable strategy")
decisions = load_decisions("missing_value_rules")
overrides = decisions.get("variables", {})

rows = []
for var, stats in metrics["variables"].items():
    rows.append(
        {
            "variable": var,
            "dtype": stats["dtype"],
            "pct_missing_dev": stats["pct_missing_dev"],
            "pct_missing_val": stats["pct_missing_val"],
            "applied_strategy": stats["strategy"],
            "fill_value": stats["fill_value"],
            "override_strategy": overrides.get(var, {}).get("strategy", "(use default)"),
        }
    )
results_df = pd.DataFrame(rows).sort_values("pct_missing_dev", ascending=False).reset_index(drop=True)

st.markdown("Set `override_strategy` to force a specific strategy for a variable; leave as "
            "`(use default)` to fall back to `default_strategy` above.")
edited = st.data_editor(
    results_df,
    key="missing_value_editor",
    width="stretch",
    disabled=["variable", "dtype", "pct_missing_dev", "pct_missing_val", "applied_strategy", "fill_value"],
    column_config={
        "override_strategy": st.column_config.SelectboxColumn(
            options=["(use default)", "mean", "median", "mode", "constant", "flag_category"]
        ),
    },
)

if st.button("Save decisions"):
    new_overrides = {}
    for row in edited.to_dict("records"):
        if row["override_strategy"] != "(use default)":
            new_overrides[row["variable"]] = {"strategy": row["override_strategy"]}
    save_decisions("missing_value_rules", {"variables": new_overrides})
    st.success("Saved to decisions/missing_value_rules.yaml. Re-run the stage to apply.")

st.divider()
st.subheader("Before / after")
dev_before = read_artifact("01_univariate", "development")
dev_after = read_artifact("02_missing", "development")
chosen_var = st.selectbox("Variable", options=list(metrics["variables"].keys()))
if chosen_var:
    bad_flag = params["global"]["bad_flag"]
    weight_col = params["global"]["sample_weight"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.plotly_chart(
            variable_trend_chart(dev_before, chosen_var, bad_flag, weight_col, params["global"].get("dummy_values")),
            width="stretch",
            key="chart_before",
        )
    with c2:
        st.markdown("**After**")
        st.plotly_chart(
            variable_trend_chart(dev_after, chosen_var, bad_flag, weight_col, []),
            width="stretch",
            key="chart_after",
        )
