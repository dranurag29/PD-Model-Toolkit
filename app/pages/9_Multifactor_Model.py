import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import run_stage  # noqa: E402
from pipeline.common.io import load_params, read_metrics, save_params  # noqa: E402

st.set_page_config(page_title="Multifactor Model", layout="wide")
st.title("Stage 8 -- Multifactor Model Search")
st.caption(
    "Ports \"8.0 model development - multifactor model.sas\". Brute-force fits a "
    "weighted logistic regression for every combination of at-most-one-variable-per-"
    "factor-group from Stage 7 -- every candidate model avoids multicollinearity by "
    "construction. Fitting uses a 50/50-rebalanced sample weight (equal weighted bad/"
    "good populations), a standard rare-event stabilization technique. No threshold "
    "filtering happens here -- every combination is recorded; picking finalists is a "
    "later stage's job."
)

params = load_params()
mm = params["multifactor_model"]

st.subheader("Parameters")
mm["max_combinations"] = st.number_input(
    "max_combinations", value=int(mm["max_combinations"]), step=100,
    help="Safety cap on the brute-force search, not present in the SAS original.",
)
if st.button("Save parameters"):
    params["multifactor_model"] = mm
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro multifactor_model... (fits one logistic regression per combination)"):
        result = run_stage("multifactor_model")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("08_multifactor_model")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

if metrics["truncated"]:
    st.warning(
        f"Search space truncated: fit {metrics['n_combinations']} of "
        f"{metrics['n_combinations_possible']} possible combinations (max_combinations cap)."
    )

m1, m2, m3 = st.columns(3)
m1.metric("combinations fit", metrics["n_combinations"])
combos = metrics["combinations"]
n_sign_correct = sum(1 for c in combos if c["sign_correct"] and c["converged"])
m2.metric("sign-correct combinations", n_sign_correct)
best_gini = max((c["gini"] for c in combos if c["converged"]), default=0.0)
m3.metric("best Gini", f"{best_gini:.3f}")

st.subheader("Leaderboard")
rows = [
    {
        "combination_id": c["combination_id"],
        "variables": ", ".join(v.replace("__", " / ") for v in c["variables"]),
        "num_factors": c["num_factors"],
        "gini": c["gini"],
        "significance": c["significance"],
        "sign_correct": c["sign_correct"],
        "converged": c["converged"],
    }
    for c in combos
]
leaderboard_df = pd.DataFrame(rows).sort_values("gini", ascending=False).reset_index(drop=True)

c1, c2 = st.columns(2)
with c1:
    only_sign_correct = st.checkbox("Only sign-correct combinations", value=True)
with c2:
    min_factors = st.slider("Minimum num_factors", 1, int(leaderboard_df["num_factors"].max()), 1)

filtered = leaderboard_df[leaderboard_df["num_factors"] >= min_factors]
if only_sign_correct:
    filtered = filtered[filtered["sign_correct"]]
st.dataframe(filtered, width="stretch")

st.divider()
st.subheader("Drill in -- combination coefficients")
top_ids = filtered["combination_id"].head(50).tolist() or leaderboard_df["combination_id"].head(50).tolist()
chosen_id = st.selectbox("Combination (top 50 shown)", options=top_ids)
if chosen_id:
    chosen = next(c for c in combos if c["combination_id"] == chosen_id)
    st.metric("Gini", f"{chosen['gini']:.4f}")
    coef_rows = [
        {"variable": v, "estimate": info["estimate"], "p_value": info["p_value"]}
        for v, info in chosen["coefficients"].items()
    ]
    st.dataframe(pd.DataFrame(coef_rows), width="stretch")
