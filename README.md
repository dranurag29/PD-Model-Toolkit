# PD Toolkit — Python/DVC/Streamlit pipeline

A Python + DVC + Streamlit rebuild of the original SAS credit-scorecard
(Probability of Default) build pipeline (the `*.sas` files in this
directory, kept as reference). DVC orchestrates stage-by-stage execution
and tracks staleness when a parameter or a per-variable analyst decision
changes; a Streamlit app is the control panel — configure parameters,
review/edit variable-level decisions, run stages, and inspect results,
all without a terminal or a SAS session.

The full pipeline is implemented end to end, all 11 stages: Data Ingestion
→ Univariate Gini/IV → Missing-Value Treatment → Outlier Treatment → Char
Recoding → Numeric Transformations → Factor Assessment → Variable Grouping
→ Multifactor Model Search → Model Selection → Bootstrap Validation →
Model Documentation. Every `*.sas` file in this directory (0.0 through
11.2) has a corresponding Python stage; see each `pipeline/stages/sNN_*.py`
module's docstring for exactly what it ports and where it deliberately
simplifies or deviates from the SAS original (always documented, never
silent).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

- `params.yaml` — scalar thresholds for every stage (mirrors the SAS
  scripts' `%let` values). DVC tracks this as the `params:` dependency of
  each stage, so changing a threshold marks the right stages stale.
- `decisions/*.yaml` — per-variable analyst calls (outlier caps, missing-
  value overrides) that used to live in the reviewed Excel workbooks
  (`Univariate analysis I/II/III.xls`). Edited via the Streamlit UI, or by
  hand; DVC-tracked as a stage dependency.
- `data/raw/` — `development_base.parquet` / `validation_base.parquet`,
  the pipeline's root input (same role as the SAS `inp.development_base` /
  `inp.validation_base` datasets). `scripts/make_synthetic_data.py`
  generates a stand-in dataset so the pipeline is runnable before real
  data is wired up.
- `pipeline/common/` — shared building blocks used across stages:
  `metrics.py` (`weighted_gini`, `weighted_iv` — one implementation
  replacing four near-duplicate SAS macros), `stats.py` (weighted mean/
  median/mode, quantile bucketing, correlation matrix, VIF, the 50/50
  rebalancing formula), `curves.py` (the Stage 5 shape-fitting library),
  `io.py` (params/decisions/artifact read-write conventions).
- `pipeline/stages/sNN_*.py` — one script per DVC stage. Each exposes a
  plain `run(params, decisions) -> dict` function (importable directly, so
  Streamlit can call it in-process) and a `__main__` entrypoint (what
  `dvc repro` calls).
- `artifacts/<stage>/` — each stage's `development.parquet` /
  `validation.parquet` + `metrics.json` (DVC `outs:`/`metrics:`).
- `app/` — the Streamlit app (`streamlit run app/Home.py`), one page per
  stage under `app/pages/`.
- `dvc.yaml` — the pipeline DAG.

## Running

```bash
source .venv/bin/activate

# run the whole pipeline once
dvc repro

# see what's stale after editing params.yaml or a decisions/*.yaml file
dvc status

# see what a threshold change actually did
dvc metrics diff

# the GUI
streamlit run app/Home.py
```

## Extending the pipeline

Every stage follows the same pattern (see `pipeline/stages/s01_univariate_gini.py`
for the simplest example):
1. A section in `params.yaml` for its scalar thresholds.
2. A `decisions/<stage>.yaml` file if it needs per-variable (or per-model)
   analyst input.
3. `pipeline/stages/sNN_name.py` with `run(params, decisions) -> dict` +
   a `__main__` entrypoint.
4. An entry in `dvc.yaml` (`deps`/`params`/`outs`/`metrics`).
5. `app/pages/N_Name.py` (param widgets + a results table + a drill-in
   chart + a "Run stage" button, mirroring an existing page).

Some stages further downstream (5+) can legitimately produce nothing to
show -- e.g. Model Selection at strict default thresholds, or Documentation
before any model has been selected. Handle that gracefully (an informative
`st.info`/`st.warning`, not a crash) rather than assuming every stage
always has output, the same way `s09_model_selection.py` and
`s11_documentation.py` do.
