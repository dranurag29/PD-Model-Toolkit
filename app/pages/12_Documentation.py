import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import calibration_chart, correlation_heatmap, run_stage  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Model Documentation", layout="wide")
st.title("Stage 11 -- Model Documentation")
st.caption(
    "Ports \"11.1\"/\"11.2\" as one stage (11.2 reuses 11.1's fitted model, only "
    "recalibrating its intercept). Refits Stage 10's #1-ranked model on the full "
    "50/50-rebalanced development sample, then documents it two ways: as fit "
    "(50/50 view -- trust the rank-ordering, not the absolute rate) and "
    "recalibrated to a target bad rate via an offset-regression intercept "
    "correction (actual view -- trust the absolute predicted probabilities)."
)

params = load_params()
dp = params["documentation"]
decisions = load_decisions("documentation_rules")

st.subheader("Parameters")
p1, p2 = st.columns(2)
with p1:
    dp["n_calibration_buckets"] = st.number_input("n_calibration_buckets", value=int(dp["n_calibration_buckets"]), step=1, min_value=2)
with p2:
    auto_rate = st.checkbox("Auto target_bad_rate (use development sample's observed rate)", value=dp["target_bad_rate"] is None)
    if auto_rate:
        dp["target_bad_rate"] = None
    else:
        current_pct = (dp["target_bad_rate"] or 0.05) * 100
        new_pct = st.number_input("target_bad_rate (%)", value=float(current_pct), min_value=0.1, max_value=99.0, step=0.1)
        dp["target_bad_rate"] = round(new_pct / 100, 6)

if st.button("Save parameters"):
    params["documentation"] = dp
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
st.subheader("Model choice")
validation_metrics = read_metrics("10_validation")
available_models = validation_metrics.get("models", [])
if available_models:
    options = ["(auto -- Stage 10's #1 by val_gini)"] + [m["combination_id"] for m in available_models]
    current = decisions.get("selected_combination_id") or options[0]
    chosen_option = st.selectbox("selected_combination_id", options=options, index=options.index(current) if current in options else 0)
    if st.button("Save model choice"):
        decisions["selected_combination_id"] = None if chosen_option == options[0] else chosen_option
        save_decisions("documentation_rules", decisions)
        st.success("Saved to decisions/documentation_rules.yaml. Re-run the stage to apply.")
else:
    st.info("No validated models available yet -- run Stage 10 (Bootstrap Validation) first.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro documentation..."):
        result = run_stage("documentation")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("11_documentation")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

if metrics["selected"] is None:
    st.warning("No validated model to document -- Stage 9/10 have nothing selected/validated yet.")
    st.stop()

sel = metrics["selected"]
st.subheader(f"Documenting: {sel['combination_id']}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dev Gini", f"{sel['dev_gini']:.4f}")
m2.metric("Val Gini", f"{sel['val_gini']:.4f}")
m3.metric("Observed bad rate", f"{metrics['observed_bad_rate']:.2%}")
m4.metric("Target bad rate", f"{metrics['target_bad_rate']:.2%}")

tab_5050, tab_actual = st.tabs(["50/50 view (rank-ordering)", "Actual view (calibrated probabilities)"])

for tab, key, label in [(tab_5050, "model_5050", "50/50"), (tab_actual, "model_actual", "actual")]:
    with tab:
        view = metrics[key]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"**Intercept**: {view['intercept']:.4f}" + (f" (shift {view.get('intercept_shift', 0):+.4f})" if "intercept_shift" in view else ""))
            coef_df = pd.DataFrame(
                [{"variable": v, "coefficient": c, "vif": view["vif"].get(v)} for v, c in view["coefficients"].items()]
            )
            st.dataframe(coef_df, width="stretch")
        with c2:
            st.plotly_chart(
                calibration_chart(view["calibration"], f"{label} view -- predicted vs. actual bad rate"),
                width="stretch",
                key=f"calibration_chart_{key}",
            )
        st.dataframe(pd.DataFrame(view["calibration"]), width="stretch")

        if view["correlation_matrix"]:
            st.plotly_chart(
                correlation_heatmap(view["correlation_matrix"], f"{label} view -- variable correlation"),
                width="stretch",
                key=f"corr_heatmap_{key}",
            )
