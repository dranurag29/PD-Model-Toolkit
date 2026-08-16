import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_artifact,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Outlier Treatment", layout="wide")
st.title("Stage 3 -- Outlier Treatment")
st.caption(
    "Ports the descriptives -> analyst rules -> generated/applied code chain "
    "in \"3.0\"/\"3.1.1\"/\"3.1.2\"/\"3.1.3 ... outlier treatment.sas\". Suggested "
    "cap bounds are the development sample's weighted percentiles; choosing "
    "'cap' without explicit bounds below uses them automatically."
)

params = load_params()
out_p = params["outlier"]

st.subheader("Default parameters")
p1, p2, p3 = st.columns(3)
with p1:
    out_p["default_strategy"] = st.selectbox(
        "default_strategy", options=["none", "cap", "drop"],
        index=["none", "cap", "drop"].index(out_p["default_strategy"]),
    )
with p2:
    out_p["suggest_lower_percentile"] = st.number_input(
        "suggest_lower_percentile", value=float(out_p["suggest_lower_percentile"]), min_value=0.0, max_value=49.0
    )
with p3:
    out_p["suggest_upper_percentile"] = st.number_input(
        "suggest_upper_percentile", value=float(out_p["suggest_upper_percentile"]), min_value=51.0, max_value=100.0
    )

if st.button("Save parameters"):
    params["outlier"] = out_p
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro outlier_treatment..."):
        result = run_stage("outlier_treatment")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("03_outlier")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

st.subheader("Per-variable treatment")
decisions = load_decisions("outlier_rules")
overrides = decisions.get("variables", {})

rows = []
for var, stats in metrics["variables"].items():
    if stats["dtype"] != "numeric":
        continue
    cfg = overrides.get(var, {})
    rows.append(
        {
            "variable": var,
            "suggested_lower_cap": stats["suggested_lower_cap"],
            "suggested_upper_cap": stats["suggested_upper_cap"],
            "applied_status": stats["status"],
            "override_status": cfg.get("status", "(use default)"),
            "override_lower_cap": cfg.get("lower_cap"),
            "override_upper_cap": cfg.get("upper_cap"),
        }
    )
results_df = pd.DataFrame(rows)

st.markdown(
    "Set `override_status` to force cap/drop/none for a variable. Leave "
    "`override_lower_cap`/`override_upper_cap` blank to use the suggested "
    "percentile bounds when capping."
)
edited = st.data_editor(
    results_df,
    key="outlier_editor",
    width="stretch",
    disabled=["variable", "suggested_lower_cap", "suggested_upper_cap", "applied_status"],
    column_config={
        "override_status": st.column_config.SelectboxColumn(options=["(use default)", "none", "cap", "drop"]),
    },
)

if metrics["dropped_variables"]:
    st.warning(f"Dropped this run: {', '.join(metrics['dropped_variables'])}")

if st.button("Save decisions"):
    new_overrides = {}
    for row in edited.to_dict("records"):
        if row["override_status"] == "(use default)":
            continue
        entry = {"status": row["override_status"]}
        if pd.notna(row.get("override_lower_cap")):
            entry["lower_cap"] = float(row["override_lower_cap"])
        if pd.notna(row.get("override_upper_cap")):
            entry["upper_cap"] = float(row["override_upper_cap"])
        new_overrides[row["variable"]] = entry
    save_decisions("outlier_rules", {"variables": new_overrides})
    st.success("Saved to decisions/outlier_rules.yaml. Re-run the stage to apply.")

st.divider()
st.subheader("Distribution")
dev = read_artifact("03_outlier", "development")
numeric_vars = [v for v, s in metrics["variables"].items() if s["dtype"] == "numeric"]
chosen_var = st.selectbox("Variable", options=numeric_vars)
if chosen_var:
    fig = go.Figure()
    fig.add_histogram(x=dev[chosen_var].dropna(), nbinsx=40)
    fig.update_layout(title=f"{chosen_var} -- post-treatment distribution", height=380, margin=dict(t=60, b=40))
    st.plotly_chart(fig, width="stretch", key="outlier_hist")
