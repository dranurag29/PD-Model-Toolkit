import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import factor_weight_chart, run_stage  # noqa: E402
from pipeline.common.io import load_params, read_metrics, save_params  # noqa: E402

st.set_page_config(page_title="Model Selection", layout="wide")
st.title("Stage 9 -- Model Selection")
st.caption(
    "Ports \"9.0 model development - select models.sas\". Filters Stage 8's "
    "combination leaderboard to real finalists (sign-correct, significant, enough "
    "factors, enough Gini), then computes each surviving variable's share of the "
    "model's discriminating power (its coefficient times the bad/good mean "
    "difference, normalized to sum to 1). Combinations where one variable "
    "dominates or another contributes too little are dropped too."
)

params = load_params()
ms = params["model_selection"]

st.subheader("Parameters")
p1, p2, p3 = st.columns(3)
with p1:
    ms["level_gini"] = st.number_input("level_gini", value=float(ms["level_gini"]), step=0.01, format="%.2f")
    ms["level_significant"] = st.number_input("level_significant", value=float(ms["level_significant"]), step=0.01, format="%.2f")
with p2:
    ms["level_num_factors"] = st.number_input("level_num_factors", value=int(ms["level_num_factors"]), step=1, min_value=1)
    ms["level_max_wt"] = st.number_input("level_max_wt", value=float(ms["level_max_wt"]), step=0.05, format="%.2f")
with p3:
    ms["level_min_wt"] = st.number_input("level_min_wt", value=float(ms["level_min_wt"]), step=0.01, format="%.2f")

if st.button("Save parameters"):
    params["model_selection"] = ms
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro model_selection..."):
        result = run_stage("model_selection")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("09_model_selection")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

st.subheader("Selection funnel")
f1, f2, f3 = st.columns(3)
f1.metric("candidates (Stage 8)", metrics["n_candidates"])
f2.metric("passed gini/significance/sign", metrics["n_after_thresholds"])
f3.metric("selected (passed weight balance)", metrics["n_selected"])

models = metrics["models"]
if not models:
    st.warning(
        "No combination survived every filter. This can be a genuine finding, not a bug -- "
        "e.g. if only one variable in the dataset carries real signal, every "
        "num_factors>=2 combination necessarily pairs it with a weak variable whose "
        "contribution share falls below level_min_wt. Try relaxing level_min_wt or "
        "level_num_factors above, or check the Multifactor Model page's leaderboard."
    )
    st.stop()

st.subheader("Selected models")
rows = [
    {
        "combination_id": m["combination_id"],
        "variables": ", ".join(v.replace("__", " / ") for v in m["variables"]),
        "num_factors": m["num_factors"],
        "gini": m["gini"],
        "significance": m["significance"],
        "max_wt": m["max_wt"],
        "min_wt": m["min_wt"],
    }
    for m in models
]
st.dataframe(pd.DataFrame(rows), width="stretch")

st.divider()
st.subheader("Drill in -- factor weights")
chosen_id = st.selectbox("Model", options=[m["combination_id"] for m in models])
if chosen_id:
    chosen = next(m for m in models if m["combination_id"] == chosen_id)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(factor_weight_chart(chosen["rows"]), width="stretch", key="factor_weight_chart")
    with c2:
        st.metric("Gini", f"{chosen['gini']:.4f}")
        st.metric("Significance", f"{chosen['significance']:.4f}")
        st.metric("Max / Min weight", f"{chosen['max_wt']:.2f} / {chosen['min_wt']:.2f}")
    st.dataframe(pd.DataFrame(chosen["rows"]), width="stretch")
