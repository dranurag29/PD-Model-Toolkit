import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import dev_vs_val_estimate_chart, run_stage  # noqa: E402
from pipeline.common.io import load_params, read_metrics, save_params  # noqa: E402

st.set_page_config(page_title="Bootstrap Validation", layout="wide")
st.title("Stage 10 -- Bootstrap Validation")
st.caption(
    "Ports \"10.0 model validation.sas\". For every Stage 9 finalist, fits on "
    "development and scores the untouched validation set -- the first stage to use "
    "it for anything beyond passthrough. Draws many bootstrap replicates from the "
    "scored validation set, refitting the model on each, to see how stable Gini and "
    "every coefficient are outside the exact sample the model was built on."
)

params = load_params()
vp = params["validation"]

st.subheader("Parameters")
p1, p2, p3 = st.columns(3)
with p1:
    vp["num_bootstrap_samples"] = st.number_input("num_bootstrap_samples", value=int(vp["num_bootstrap_samples"]), step=50, min_value=10)
with p2:
    vp["sample_rate"] = st.number_input("sample_rate (%)", value=int(vp["sample_rate"]), step=5, min_value=10, max_value=100)
with p3:
    vp["random_seed"] = st.number_input("random_seed", value=int(vp["random_seed"]), step=1)

if st.button("Save parameters"):
    params["validation"] = vp
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro validation... (bootstraps every finalist, may take a while)"):
        result = run_stage("validation")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("10_validation")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

models = metrics["models"]
st.metric("models bootstrap-validated", metrics["n_models_validated"])

if not models:
    st.warning(
        "No models to validate -- Stage 9 (Model Selection) selected zero finalists "
        "with the current thresholds. Relax its thresholds first, or check its page."
    )
    st.stop()

st.subheader("Validated models")
rows = [
    {
        "combination_id": m["combination_id"],
        "variables": ", ".join(v.replace("__", " / ") for v in m["variables"]),
        "num_factors": m["num_factors"],
        "dev_gini": m["dev_gini"],
        "val_gini": m["val_gini"],
        "coeff_of_var_gini": m["coeff_of_var_gini"],
        "val_sign_correct": m["val_sign_correct"],
        "max_cv": m["max_cv"],
    }
    for m in models
]
st.dataframe(pd.DataFrame(rows), width="stretch")

st.divider()
st.subheader("Drill in -- development vs. bootstrap-validated coefficients")
chosen_id = st.selectbox("Model", options=[m["combination_id"] for m in models])
if chosen_id:
    chosen = next(m for m in models if m["combination_id"] == chosen_id)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(dev_vs_val_estimate_chart(chosen["rows"]), width="stretch", key="dev_vs_val_chart")
    with c2:
        st.metric("Dev Gini", f"{chosen['dev_gini']:.4f}")
        st.metric("Bootstrap-mean val Gini", f"{chosen['val_gini']:.4f}")
        st.metric("Coeff. of var (Gini)", f"{chosen['coeff_of_var_gini']:.2%}" if chosen["coeff_of_var_gini"] is not None else "n/a")
        st.metric("Max coeff. of var (any variable)", f"{chosen['max_cv']:.2%}")
    st.dataframe(pd.DataFrame(chosen["rows"]), width="stretch")
