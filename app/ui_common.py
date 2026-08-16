"""Shared helpers for the Streamlit pages: running DVC as a subprocess (the
actual reproducibility mechanism -- Streamlit is a thin control panel over
it, not a second orchestrator), and a generic on-the-fly bad-rate-by-bin
chart used by every stage's drill-in panel."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common.curves import (  # noqa: E402
    eval_bell,
    eval_inverted_bell,
    eval_linear,
    eval_log_linear,
    eval_quadratic,
    eval_s_shape,
)
from pipeline.common.io import PROJECT_ROOT  # noqa: E402


def run_dvc(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["dvc", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
    )


def run_stage(stage_name: str) -> subprocess.CompletedProcess:
    return run_dvc("repro", stage_name)


def run_full_pipeline() -> subprocess.CompletedProcess:
    return run_dvc("repro")


def dvc_status_text() -> str:
    result = run_dvc("status")
    return (result.stdout or "") + (result.stderr or "")


def bad_rate_population_chart(labels: list, bad_rate: list, pop_share: list, title: str) -> go.Figure:
    """Shared layout for every "bad rate + population share by group" chart
    in the app -- a bar for population share on a secondary axis, a line for
    bad rate on the primary axis, both as percentages."""
    fig = go.Figure()
    fig.add_bar(x=labels, y=pop_share, name="population share", yaxis="y2", opacity=0.35, marker_color="#888")
    fig.add_scatter(x=labels, y=bad_rate, name="bad rate", mode="lines+markers", yaxis="y1")
    fig.update_layout(
        title=title,
        yaxis=dict(title="bad rate", tickformat=".0%"),
        yaxis2=dict(title="population share", overlaying="y", side="right", tickformat=".0%"),
        legend=dict(orientation="h", y=1.1),
        height=380,
        margin=dict(t=60, b=40),
    )
    return fig


def variable_trend_chart(
    df: pd.DataFrame,
    var: str,
    bad_flag: str,
    weight_col: str,
    dummy_values: list = None,
    max_bins: int = 10,
) -> go.Figure:
    """Weighted bad-rate (and population share) by bin, for one variable --
    the Streamlit drill-in replacement for the SAS `proc gplot` trend charts
    in "5.0 create variable trends.sas". Numeric variables are quantile-
    binned; categorical variables are grouped by category as-is."""
    dummy_values = dummy_values or []
    x = df[var]
    y = df[bad_flag].astype(float)
    w = df[weight_col].astype(float)

    is_numeric = pd.api.types.is_numeric_dtype(x)
    missing = x.isna() | (x.isin(dummy_values) if (dummy_values and is_numeric) else False)

    work = pd.DataFrame({"x": x, "y": y, "w": w})
    work_nm = work[~missing]

    if is_numeric:
        try:
            work_nm = work_nm.assign(bin=pd.qcut(work_nm["x"], max_bins, duplicates="drop"))
            group_key = "bin"
        except ValueError:
            group_key = "x"
    else:
        group_key = "x"

    grp = work_nm.groupby(group_key, observed=True).apply(
        lambda g: pd.Series(
            {
                "bad_rate": (g["y"] * g["w"]).sum() / g["w"].sum() if g["w"].sum() else 0.0,
                "population": g["w"].sum(),
            }
        ),
        include_groups=False,
    )
    grp = grp.sort_index()
    labels = [str(i) for i in grp.index]
    pop_share = grp["population"] / grp["population"].sum() if grp["population"].sum() else grp["population"]

    n_missing = int(missing.sum())
    title = f"{var} -- bad rate by bin" + (f" ({n_missing} missing/dummy excluded)" if n_missing else "")
    return bad_rate_population_chart(labels, grp["bad_rate"].tolist(), pop_share.tolist(), title)


def evaluate_transformation_curve(family: str, params: dict, extra: dict, x: np.ndarray, min_x: float, max_x: float) -> np.ndarray:
    """Evaluates a fitted transformation family from its JSON-serialized
    metrics (plain params dict, plus binning's edges/log_odds lookup table
    in `extra`) -- used to draw each candidate curve over the actual x range
    in the Stage 5 drill-in overlay chart."""
    if family == "linear":
        return eval_linear(params, x)
    if family == "quadratic":
        return eval_quadratic(params, x)
    if family == "log_linear":
        return eval_log_linear(params, x)
    if family == "bell":
        return eval_bell(params, x, min_x, max_x)
    if family == "inverted_bell":
        return eval_inverted_bell(params, x, min_x, max_x)
    if family == "s_shape":
        return eval_s_shape(params, x, min_x, max_x)
    if family == "binning":
        edges = np.array(extra["edges"])
        log_odds = np.array(extra["log_odds"])
        idx = np.clip(np.searchsorted(edges, x, side="left"), 0, len(log_odds) - 1)
        return log_odds[idx]
    raise ValueError(f"unknown family: {family}")


def transformation_overlay_chart(variable_metrics: dict, title: str) -> go.Figure:
    """Actual binned log-odds trend (markers, sized by bucket population)
    overlaid with every USABLE fitted curve -- the drill-in replacement for
    the SAS `proc gplot` "actual vs predicted" curve-fit charts in
    "5.1.1 ...".check_*'s per-model jpeg output."""
    buckets = variable_metrics["trend_buckets"]
    min_x, max_x = variable_metrics["min_x"], variable_metrics["max_x"]
    x_grid = np.linspace(min_x, max_x, 200)

    fig = go.Figure()
    weights = np.array(buckets["weight"])
    sizes = 8 + 20 * (weights / weights.max() if weights.max() > 0 else weights)
    fig.add_scatter(
        x=buckets["x"], y=buckets["ln_odd"], mode="markers", name="actual (binned)",
        marker=dict(size=sizes, color="#888"),
    )
    for attempt in variable_metrics["attempts"]:
        if not attempt["usable"]:
            continue
        y_grid = evaluate_transformation_curve(
            attempt["family"], attempt["params"], attempt.get("extra", {}), x_grid, min_x, max_x
        )
        fig.add_scatter(x=x_grid, y=y_grid, mode="lines", name=f"{attempt['family']} (R²={attempt['r_squared']:.2f})")

    fig.update_layout(
        title=title,
        xaxis_title="variable value",
        yaxis_title="log-odds",
        legend=dict(orientation="h", y=1.15),
        height=420,
        margin=dict(t=80, b=40),
    )
    return fig


def level_diagnostics_chart(diagnostics: list, title: str) -> go.Figure:
    """Same bad-rate/population-share chart, but from the per-level
    diagnostics list pipeline.stages.s04_char_recoding._level_diagnostics
    produces (already-aggregated, no raw dataframe needed) -- used by the
    Char Recoding page's before/after and drill-in views."""
    labels = [str(d["level"]) for d in diagnostics]
    bad_rate = [d["bad_rate"] for d in diagnostics]
    pop_share = [d["population_share"] for d in diagnostics]
    return bad_rate_population_chart(labels, bad_rate, pop_share, title)


def correlation_heatmap(correlation_matrix: dict, title: str = "Correlation matrix") -> go.Figure:
    """Diverging heatmap of a {a: {b: corr}} matrix -- the visual backing
    for Stage 7's connected-components variable grouping: variables that
    visibly cluster into dark red blocks are exactly the ones the pipeline
    grouped together."""
    labels = list(correlation_matrix.keys())
    z = [[correlation_matrix[a][b] for b in labels] for a in labels]
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
            text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
            colorbar=dict(title="corr"),
        )
    )
    fig.update_layout(title=title, height=max(360, 40 * len(labels)), margin=dict(t=60, b=40))
    return fig


def factor_weight_chart(rows: list, title: str = "Factor weight by variable") -> go.Figure:
    """Horizontal bar of each variable's factor_weight share within a
    selected Stage 9 model -- how much of the model's discriminating power
    (coefficient x bad/good mean difference) each variable is pulling."""
    ordered = sorted(rows, key=lambda r: r["factor_weight"])
    labels = [r["variable"] for r in ordered]
    values = [r["factor_weight"] for r in ordered]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=[f"{v:.1%}" for v in values], textposition="outside"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="factor weight", tickformat=".0%"),
        height=max(280, 45 * len(labels)),
        margin=dict(t=60, b=40, l=10, r=10),
    )
    return fig


def dev_vs_val_estimate_chart(rows: list, title: str = "Development vs. bootstrap-validated coefficient") -> go.Figure:
    """Grouped bar comparing each variable's single-fit development
    coefficient against its mean coefficient across Stage 10's bootstrap
    replicates -- a big gap between the two bars is itself a stability
    warning, on top of the explicit coeff_of_var_estimate number."""
    labels = [r["variable"] for r in rows]
    dev_vals = [r["dev_estimate"] for r in rows]
    val_vals = [r["val_estimate"] for r in rows]
    fig = go.Figure()
    fig.add_bar(x=labels, y=dev_vals, name="development estimate")
    fig.add_bar(x=labels, y=val_vals, name="bootstrap mean estimate")
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis_title="coefficient estimate",
        legend=dict(orientation="h", y=1.15),
        height=380,
        margin=dict(t=80, b=40),
    )
    return fig


def calibration_chart(rows: list, title: str) -> go.Figure:
    """Predicted vs. actual bad rate by population-weighted decile -- the
    classic scorecard calibration/reliability plot. Close agreement between
    the two lines means the model's predicted probabilities can be trusted
    at face value, not just used for rank-ordering."""
    buckets = [r["bucket"] for r in rows]
    fig = go.Figure()
    fig.add_scatter(x=buckets, y=[r["pred_bad_rate"] for r in rows], name="predicted", mode="lines+markers")
    fig.add_scatter(x=buckets, y=[r["act_bad_rate"] for r in rows], name="actual", mode="lines+markers")
    fig.update_layout(
        title=title,
        xaxis_title="bucket (0=lowest predicted risk)",
        yaxis=dict(title="bad rate", tickformat=".0%"),
        legend=dict(orientation="h", y=1.15),
        height=380,
        margin=dict(t=80, b=40),
    )
    return fig
