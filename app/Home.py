import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ui_common import dvc_status_text, run_full_pipeline  # noqa: E402
from pipeline.common.io import ARTIFACTS_DIR, read_metrics  # noqa: E402

st.set_page_config(page_title="PD Toolkit", layout="wide")

st.title("PD Toolkit")
st.caption(
    "Python/DVC rebuild of the SAS credit-scorecard pipeline. Use the pages "
    "in the sidebar to configure each stage, review/edit variable-level "
    "decisions, and run stages -- no terminal or SAS session needed."
)

STAGES = [
    ("Data Ingestion", None),
    ("Univariate Gini/IV", "01_univariate"),
    ("Missing-Value Treatment", "02_missing"),
    ("Outlier Treatment", "03_outlier"),
    ("Char Recoding", "04_char_recoding"),
    ("Numeric Transformations", "05_transformations"),
    ("Factor Assessment", "06_factor_assessment"),
    ("Variable Grouping", "07_variable_grouping"),
    ("Multifactor Model", "08_multifactor_model"),
    ("Model Selection", "09_model_selection"),
    ("Bootstrap Validation", "10_validation"),
    ("Model Documentation", "11_documentation"),
]

st.subheader("Pipeline status")
PER_ROW = 5
for row_start in range(0, len(STAGES), PER_ROW):
    row_stages = STAGES[row_start : row_start + PER_ROW]
    cols = st.columns(PER_ROW)
    for col, (label, stage_dir) in zip(cols, row_stages):
        with col:
            st.markdown(f"**{label}**")
            if stage_dir is None:
                st.caption("see Data Ingestion page")
                continue
            metrics = read_metrics(stage_dir)
            if not metrics:
                st.caption("not yet run")
            elif stage_dir == "01_univariate":
                st.metric("variables selected", f"{metrics['n_selected_variables']}/{metrics['n_candidate_variables']}")
            elif stage_dir == "06_factor_assessment":
                n_usable = sum(1 for c in metrics["candidates"].values() if c["usable"])
                st.metric("usable candidates", f"{n_usable}/{len(metrics['candidates'])}")
            elif stage_dir == "07_variable_grouping":
                st.metric("factor groups", f"{len(metrics['factor_groups'])} from {len(metrics['winners'])} vars")
            elif stage_dir == "08_multifactor_model":
                st.metric("combinations fit", metrics["n_combinations"])
            elif stage_dir == "09_model_selection":
                st.metric("models selected", f"{metrics['n_selected']}/{metrics['n_after_thresholds']}")
            elif stage_dir == "10_validation":
                st.metric("models validated", metrics["n_models_validated"])
            elif stage_dir == "11_documentation":
                st.caption("documented" if metrics.get("selected") else "no model to document")
            else:
                st.caption(f"{len(metrics.get('variables', {}))} variables processed")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("DVC DAG status")
    if st.button("Refresh status"):
        st.rerun()
    status = dvc_status_text()
    st.code(status or "Data and pipelines are up to date.", language="text")

with right:
    st.subheader("Run everything")
    st.caption("Runs `dvc repro` -- only stale stages actually re-execute.")
    if st.button("Run full pipeline", type="primary"):
        with st.spinner("Running dvc repro..."):
            result = run_full_pipeline()
        if result.returncode == 0:
            st.success("Pipeline up to date.")
        else:
            st.error("dvc repro failed -- see log below.")
        with st.expander("Log", expanded=result.returncode != 0):
            st.code((result.stdout or "") + (result.stderr or ""), language="text")

st.divider()
st.caption(f"Artifacts written under `{ARTIFACTS_DIR.relative_to(ARTIFACTS_DIR.parents[1])}`.")
