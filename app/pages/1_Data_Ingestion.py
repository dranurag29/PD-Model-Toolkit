import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage  # noqa: E402
from pipeline.common.io import DATA_RAW_DIR, load_params  # noqa: E402

st.set_page_config(page_title="Data Ingestion", layout="wide")
st.title("Data Ingestion")
st.caption(
    "The pipeline's root input: development_base (used to fit/select variables) "
    "and validation_base (held out, used from Stage 10 onward -- see the Phase 2 "
    "roadmap in params.yaml). Same role as the SAS `inp.development_base` / "
    "`inp.validation_base` datasets."
)

params = load_params()
required_cols = [
    params["global"]["application_no"],
    params["global"]["bad_flag"],
    params["global"]["sample_weight"],
]


def _sample_summary(name: str):
    path = DATA_RAW_DIR / f"{name}_base.parquet"
    if not path.exists():
        st.warning(f"`{path.name}` not found.")
        return
    df = pd.read_parquet(path)
    bad_col = params["global"]["bad_flag"]
    c1, c2, c3 = st.columns(3)
    c1.metric("rows", f"{len(df):,}")
    c2.metric("columns", len(df.columns))
    if bad_col in df.columns:
        c3.metric("bad rate", f"{df[bad_col].mean():.2%}")
    with st.expander("Preview"):
        st.dataframe(df.head(20), width="stretch")


tab_dev, tab_val = st.tabs(["development_base", "validation_base"])
with tab_dev:
    _sample_summary("development")
with tab_val:
    _sample_summary("validation")

st.divider()
st.subheader("Bootstrap with synthetic data")
st.caption(
    "No real bank export is wired up yet -- this generates a stand-in dataset "
    "(see `scripts/make_synthetic_data.py`) so every stage is runnable today."
)
if st.button("(Re)generate synthetic data"):
    with st.spinner("Running dvc repro generate_synthetic_data..."):
        result = run_stage("generate_synthetic_data")
    if result.returncode == 0:
        st.success("Synthetic data regenerated.")
        st.rerun()
    else:
        st.error("Failed -- see log below.")
        st.code((result.stdout or "") + (result.stderr or ""), language="text")

st.divider()
st.subheader("Upload real data")
st.caption(
    f"Replaces `data/raw/<sample>_base.parquet`. Must contain the columns "
    f"configured in params.yaml: `{'`, `'.join(required_cols)}`."
)
upload_col1, upload_col2 = st.columns(2)
for col, sample in ((upload_col1, "development"), (upload_col2, "validation")):
    with col:
        st.markdown(f"**{sample}_base**")
        uploaded = st.file_uploader(
            f"Parquet or CSV for {sample}_base", type=["parquet", "csv"], key=f"upload_{sample}"
        )
        if uploaded is not None:
            df = pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.error(f"Missing required column(s): {', '.join(missing_cols)}")
            else:
                st.success(f"{len(df):,} rows, {len(df.columns)} columns -- schema OK.")
                if st.button(f"Save as data/raw/{sample}_base.parquet", key=f"save_{sample}"):
                    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(DATA_RAW_DIR / f"{sample}_base.parquet", index=False)
                    st.success("Saved. Run downstream stages from their pages, or Home > Run full pipeline.")
                    st.rerun()
