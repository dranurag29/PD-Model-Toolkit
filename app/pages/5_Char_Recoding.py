import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui_common import level_diagnostics_chart, run_stage  # noqa: E402
from pipeline.common.io import (  # noqa: E402
    load_decisions,
    load_params,
    read_metrics,
    save_decisions,
    save_params,
)

st.set_page_config(page_title="Char Recoding", layout="wide")
st.title("Stage 4 -- Categorical Variable Recoding")
st.caption(
    "Ports \"4.0 char recoding etc.sas\". In the SAS file the frequency-table "
    "and bad-rate macros are commented out, and grouping was decided externally "
    "via CHAID + manual review -- as written, the stage is a pass-through. Here "
    "the diagnostics run unconditionally, sparse categories are auto-consolidated "
    "by default, and you can define custom groupings per variable below."
)

params = load_params()
cr = params["char_recoding"]

st.subheader("Default parameters")
p1, p2 = st.columns(2)
with p1:
    cr["min_category_share"] = st.number_input(
        "min_category_share", value=float(cr["min_category_share"]), min_value=0.0, max_value=1.0, step=0.01,
        format="%.2f",
        help="Categories below this weighted population share get auto-consolidated into the rare bucket.",
    )
with p2:
    cr["rare_category_bucket_name"] = st.text_input("rare_category_bucket_name", value=cr["rare_category_bucket_name"])

if st.button("Save parameters"):
    params["char_recoding"] = cr
    save_params(params)
    st.success("Saved to params.yaml.")

st.divider()
if st.button("Run stage", type="primary"):
    with st.spinner("Running dvc repro char_recoding..."):
        result = run_stage("char_recoding")
    if result.returncode == 0:
        st.success("Stage complete.")
    else:
        st.error("Stage failed -- see log below.")
    with st.expander("Log", expanded=result.returncode != 0):
        st.code((result.stdout or "") + (result.stderr or ""), language="text")
    st.rerun()

metrics = read_metrics("04_char_recoding")
if not metrics:
    st.info("Stage hasn't been run yet -- click 'Run stage' above.")
    st.stop()

decisions = load_decisions("char_recoding_rules")
overrides = decisions.get("variables", {})

st.subheader("Per-variable mode")
rows = [
    {
        "variable": var,
        "n_levels_before": stats["n_levels_before"],
        "n_levels_after": stats["n_levels_after"],
        "applied_mode": stats["mode"],
        "override_mode": overrides.get(var, {}).get("mode", "(use auto)"),
    }
    for var, stats in metrics["variables"].items()
]
results_df = pd.DataFrame(rows)

st.markdown(
    "`custom` mode is driven by the group mapping you set in the drill-in section below -- "
    "changing `override_mode` here only switches which mode applies, it doesn't clear a "
    "previously-saved custom mapping."
)
edited = st.data_editor(
    results_df,
    key="char_recoding_mode_editor",
    width="stretch",
    disabled=["variable", "n_levels_before", "n_levels_after", "applied_mode"],
    column_config={
        "override_mode": st.column_config.SelectboxColumn(options=["(use auto)", "auto", "none", "custom"]),
    },
)

if st.button("Save modes"):
    for row in edited.to_dict("records"):
        var = row["variable"]
        entry = dict(overrides.get(var, {}))
        if row["override_mode"] == "(use auto)":
            entry.pop("mode", None)
        else:
            entry["mode"] = row["override_mode"]
        if entry:
            overrides[var] = entry
        else:
            overrides.pop(var, None)
    save_decisions("char_recoding_rules", {"variables": overrides})
    st.success("Saved to decisions/char_recoding_rules.yaml. Re-run the stage to apply.")

st.divider()
st.subheader("Drill in -- build a custom grouping")
chosen_var = st.selectbox("Variable", options=list(metrics["variables"].keys()))
if chosen_var:
    stats = metrics["variables"][chosen_var]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            level_diagnostics_chart(stats["before"], f"{chosen_var} -- before"),
            width="stretch", key="chart_before_levels",
        )
    with c2:
        st.plotly_chart(
            level_diagnostics_chart(stats["after"], f"{chosen_var} -- after"),
            width="stretch", key="chart_after_levels",
        )

    existing_groups = overrides.get(chosen_var, {}).get("groups", {})
    grouping_df = pd.DataFrame(
        [
            {
                "level": d["level"],
                "population_share": d["population_share"],
                "bad_rate": d["bad_rate"],
                "target_group": existing_groups.get(d["level"], d["level"]),
            }
            for d in stats["before"]
        ]
    )
    st.markdown(
        "Edit `target_group` to rename/merge levels -- give two levels the same "
        "`target_group` name to collapse them together. Leave unchanged to keep a level as-is."
    )
    edited_groups = st.data_editor(
        grouping_df,
        key="char_recoding_grouping_editor",
        width="stretch",
        disabled=["level", "population_share", "bad_rate"],
    )

    if st.button(f"Save grouping for {chosen_var}"):
        group_map = {
            row["level"]: row["target_group"]
            for row in edited_groups.to_dict("records")
            if row["target_group"] != row["level"]
        }
        entry = dict(overrides.get(chosen_var, {}))
        entry["mode"] = "custom"
        entry["groups"] = group_map
        overrides[chosen_var] = entry
        save_decisions("char_recoding_rules", {"variables": overrides})
        st.success(f"Saved custom grouping for {chosen_var}. Re-run the stage to apply.")
        st.rerun()
