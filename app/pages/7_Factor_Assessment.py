import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage  # noqa: E402
from pipeline.common.io import load_params, read_metrics, save_params  # noqa: E402

st.set_page_config(page_title="Factor Assessment", layout="wide")
st.title("Stage 6 -- Factor Assessment")
st.caption(
    "Ports \"6.0 factor assessment.sas\". Every numeric candidate column from "
    "Stage 5 (raw variables and every transformed derivative) gets its own "
    "weighted univariate logistic regression, scored by Gini, coefficient "
    "significance, and sign. An adjusted Gini (a small bonus for simpler shapes "
    "-- linear, S-shape, binning) determines usability. This stage only scores "
    "candidates; picking one winner per variable is a later stage's job."
)

params = load_params()
fa = params["factor_assessment"]

st.subheader("Parameters")
p1, p2 = st.columns(2)
with p1:
    fa["gini_level"] = st.number_input("gini_level", value=float(fa["gini_level"]), step=0.01, format="%.2f")
with p2:
    fa["significance_level"] = st.number_input("significance_level", value=float(fa["significance_level"]), step=0.01, format="%.2f")

if st.button("Save parameters"):
    params["factor_assessment"] = fa
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro factor_assessment..."):
        result = run_stage("factor_assessment")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("06_factor_assessment")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

candidates = metrics["candidates"]
rows = [
    {
        "column": col,
        "actual_variable": c["actual_variable"],
        "transformation_family": c["transformation_family"] or "(raw)",
        "gini": c["gini"],
        "significance": c["significance"],
        "sign_correct": c["sign_correct"],
        "r_squared": c["r_squared"],
        "adjusted_gini": c["adjusted_gini"],
        "usable": c["usable"],
    }
    for col, c in candidates.items()
]
results_df = pd.DataFrame(rows).sort_values(["actual_variable", "adjusted_gini"], ascending=[True, False])

base_vars = sorted(results_df["actual_variable"].unique())
n_with_usable = sum(1 for v in base_vars if results_df[(results_df["actual_variable"] == v) & results_df["usable"]].shape[0] > 0)
m1, m2, m3 = st.columns(3)
m1.metric("candidate columns", len(results_df))
m2.metric("usable columns", int(results_df["usable"].sum()))
m3.metric("base variables with a usable candidate", f"{n_with_usable}/{len(base_vars)}")

st.subheader("All candidates")
st.dataframe(results_df, width="stretch")

st.divider()
st.subheader("Drill in -- best transformation per variable")
chosen_var = st.selectbox("Base variable", options=base_vars)
if chosen_var:
    var_rows = results_df[results_df["actual_variable"] == chosen_var].copy()
    var_rows = var_rows.sort_values("adjusted_gini", ascending=False).reset_index(drop=True)
    usable_rows = var_rows[var_rows["usable"]]
    if len(usable_rows) > 0:
        winner = usable_rows.iloc[0]
        st.success(
            f"Current winner: **{winner['column']}** "
            f"(family={winner['transformation_family']}, adjusted_gini={winner['adjusted_gini']:.3f})"
        )
    else:
        st.warning(f"No usable transformation for {chosen_var}.")

    def _highlight_winner(row):
        is_winner = len(usable_rows) > 0 and row["column"] == usable_rows.iloc[0]["column"]
        return ["background-color: #d4edda" if is_winner else "" for _ in row]

    st.dataframe(var_rows.style.apply(_highlight_winner, axis=1), width="stretch")
