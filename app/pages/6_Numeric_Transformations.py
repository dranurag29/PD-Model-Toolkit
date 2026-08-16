import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage, transformation_overlay_chart  # noqa: E402
from pipeline.common.curves import ALL_TRENDS, applicable_families  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Numeric Transformations", layout="wide")
st.title("Stage 5 -- Numeric Variable Transformations")
st.caption(
    "Ports \"5.1.1 numeric transformations - test applicable transformation.sas\". "
    "For every numeric variable, fits several candidate curve shapes (linear, "
    "quadratic, log-linear, bell, inverted bell, S-shape, plus an always-available "
    "adaptive binning fallback) against its binned log-odds trend, and keeps every "
    "shape that's trend-conforming, statistically significant, and has enough R square. "
    "Usable shapes are materialized as new `<variable>__<family>` columns."
)

params = load_params()
tp = params["transformations"]

st.subheader("Parameters")
p1, p2, p3 = st.columns(3)
with p1:
    tp["significance_level"] = st.number_input("significance_level", value=float(tp["significance_level"]), step=0.01, format="%.2f")
    tp["r_square_level"] = st.number_input("r_square_level", value=float(tp["r_square_level"]), step=0.01, format="%.2f")
with p2:
    tp["n_trend_buckets"] = st.number_input("n_trend_buckets", value=int(tp["n_trend_buckets"]), step=1)
    tp["min_trend_bucket_size"] = st.number_input("min_trend_bucket_size", value=int(tp["min_trend_bucket_size"]), step=10)
with p3:
    tp["binning_n_bins"] = st.number_input("binning_n_bins", value=int(tp["binning_n_bins"]), step=1)
    tp["binning_min_bin_size"] = st.number_input("binning_min_bin_size", value=int(tp["binning_min_bin_size"]), step=10)

if st.button("Save parameters"):
    params["transformations"] = tp
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro transformations... (fits every shape for every numeric variable, may take a moment)"):
        result = run_stage("transformations")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("05_transformations")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

decisions = load_decisions("transformation_rules")
overrides = decisions.get("variables", {})

st.subheader("Per-variable overview")
rows = [
    {
        "variable": var,
        "expected_trend": stats["expected_trend"],
        "n_dummy_dev": stats["n_dummy_dev"],
        "n_usable": len(stats["materialized_columns"]),
        "usable_families": ", ".join(stats["materialized_columns"]) or "(none)",
    }
    for var, stats in metrics["variables"].items()
]
st.dataframe(pd.DataFrame(rows), width="stretch")

st.divider()
st.subheader("Drill in")
chosen_var = st.selectbox("Variable", options=list(metrics["variables"].keys()))
if chosen_var:
    var_metrics = metrics["variables"][chosen_var]

    st.plotly_chart(
        transformation_overlay_chart(var_metrics, f"{chosen_var} -- actual vs fitted curves"),
        width="stretch",
        key="transform_overlay_chart",
    )

    attempts_df = pd.DataFrame(
        [
            {
                "family": a["family"],
                "converged": a["converged"],
                "r_squared": a["r_squared"],
                "significance": a["significance"],
                "assessment": a["assessment"],
                "usable": a["usable"],
            }
            for a in var_metrics["attempts"]
        ]
    )
    st.dataframe(attempts_df, width="stretch")

    st.markdown("**Per-variable decision** (replaces the analyst-set \"expected trend and adjustment\" "
                "Excel sheet -- controls which shapes are even attempted, see `pipeline.common.curves.applicable_families`)")
    current_cfg = overrides.get(chosen_var, {})
    c1, c2 = st.columns(2)
    with c1:
        new_trend = st.selectbox(
            "expected_trend", options=ALL_TRENDS,
            index=ALL_TRENDS.index(current_cfg.get("expected_trend", "any")),
        )
    with c2:
        candidate_families = applicable_families(new_trend, var_metrics["min_x"])
        current_allowed = current_cfg.get("allowed_models") or candidate_families
        new_allowed = st.multiselect(
            "allowed_models", options=candidate_families,
            default=[f for f in current_allowed if f in candidate_families],
        )

    if st.button(f"Save decision for {chosen_var}"):
        entry = {"expected_trend": new_trend}
        if set(new_allowed) != set(candidate_families):
            entry["allowed_models"] = new_allowed
        overrides[chosen_var] = entry
        save_decisions("transformation_rules", {"variables": overrides})
        st.success("Saved to decisions/transformation_rules.yaml. Re-run the stage to apply.")
        st.rerun()
