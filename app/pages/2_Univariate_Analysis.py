import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage, variable_trend_chart  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_metrics,
    read_raw,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Univariate Analysis", layout="wide")
st.title("Stage 1 -- Univariate Gini/IV Screening")
st.caption(
    "Ports \"1.0 initial univariate ginis.sas\". Every candidate variable gets a "
    "non-missing Gini, a with-missing Gini + Information Value, and passes if it "
    "clears any Gini/fill-rate pair below OR the IV bar -- same OR-of-conditions "
    "rule as the original SAS var_select macro."
)

params = load_params()
uni = params["univariate"]

st.subheader("Parameters")
p1, p2 = st.columns(2)
with p1:
    uni["min_iv"] = st.number_input("min_iv", value=float(uni["min_iv"]), step=0.01, format="%.2f")
    uni["max_missing_gini_uplift"] = st.number_input(
        "max_missing_gini_uplift", value=float(uni["max_missing_gini_uplift"]), step=0.01, format="%.2f"
    )
    uni["missing_gini_uplift_ceiling"] = st.number_input(
        "missing_gini_uplift_ceiling", value=float(uni["missing_gini_uplift_ceiling"]), step=0.01, format="%.2f"
    )
with p2:
    uni["iv_bin_count"] = st.number_input("iv_bin_count", value=int(uni["iv_bin_count"]), step=1)
    uni["iv_min_bin_size"] = st.number_input("iv_min_bin_size", value=int(uni["iv_min_bin_size"]), step=10)
    uni["iv_min_bins_floor"] = st.number_input("iv_min_bins_floor", value=int(uni["iv_min_bins_floor"]), step=1)

st.markdown("**Gini / fill-rate pass conditions** -- a variable passes if it clears ANY row below")
conditions_df = pd.DataFrame(
    [{"min_gini": c["min_gini"], "min_fill_rate": c["min_fill_rate"]} for c in uni["gini_fillrate_conditions"]]
)
edited_conditions = st.data_editor(conditions_df, num_rows="dynamic", key="conditions_editor")

if st.button("Save parameters"):
    uni["gini_fillrate_conditions"] = [
        {"min_gini": float(r["min_gini"]), "min_fill_rate": float(r["min_fill_rate"])}
        for r in edited_conditions.to_dict("records")
    ]
    params["univariate"] = uni
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()

run_col, _ = st.columns([1, 3])
with run_col:
    if st.button("Run stage", type="primary"):
        with st.spinner("Running dvc repro univariate_gini..."):
            result = run_stage("univariate_gini")
        if result.returncode == 0:
            st.success("Stage complete.")
        else:
            st.error("Stage failed -- see log below.")
        with st.expander("Log", expanded=result.returncode != 0):
            st.code((result.stdout or "") + (result.stderr or ""), language="text")
        st.rerun()

metrics = read_metrics("01_univariate")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

st.subheader(f"Results -- {metrics['n_selected_variables']}/{metrics['n_candidate_variables']} variables selected")

decisions = load_decisions("univariate_review")
overrides = decisions.get("variables", {})

rows = []
for var, stats in metrics["variables"].items():
    rows.append(
        {
            "variable": var,
            "dtype": stats["dtype"],
            "fill_rate": stats["fill_rate"],
            "gini_non_missing": stats["gini_non_missing"],
            "gini_with_missing": stats["gini_with_missing"],
            "iv": stats["iv"],
            "auto_pass": stats["auto_pass"],
            "decision_status": overrides.get(var, {}).get("status", "auto"),
            "selected": stats["selected"],
        }
    )
results_df = pd.DataFrame(rows).sort_values("gini_with_missing", ascending=False).reset_index(drop=True)

st.markdown("Edit `decision_status` to force-include/exclude a variable regardless of the auto screen "
            "(replaces the analyst-reviewed `expected_trend` column from Univariate analysis I.xls).")
edited_results = st.data_editor(
    results_df,
    key="univariate_results_editor",
    width="stretch",
    disabled=["variable", "dtype", "fill_rate", "gini_non_missing", "gini_with_missing", "iv", "auto_pass", "selected"],
    column_config={
        "decision_status": st.column_config.SelectboxColumn(options=["auto", "keep", "drop"]),
    },
)

if st.button("Save decisions"):
    new_overrides = {
        row["variable"]: {"status": row["decision_status"]}
        for row in edited_results.to_dict("records")
    }
    save_decisions("univariate_review", {"variables": new_overrides})
    st.success("Saved to decisions/univariate_review.yaml. Re-run the stage to apply.")

st.divider()
st.subheader("Drill in")
dev = read_raw("development")
chosen_var = st.selectbox("Variable", options=list(metrics["variables"].keys()))
if chosen_var:
    fig = variable_trend_chart(
        dev,
        chosen_var,
        params["global"]["bad_flag"],
        params["global"]["sample_weight"],
        dummy_values=params["global"].get("dummy_values"),
    )
    st.plotly_chart(fig, width="stretch")
    st.json(metrics["variables"][chosen_var])
