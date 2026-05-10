from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rlva.src.config import ANALYSIS_FEATURE_COLUMNS, ENV_LABELS, SUMMARY_FEATURE_COLUMNS


def embed_summary(summary_df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    cols = feature_columns if feature_columns is not None else SUMMARY_FEATURE_COLUMNS
    x = summary_df[cols].astype(float).to_numpy()
    scaled = StandardScaler().fit_transform(x) if len(summary_df) >= 2 else np.zeros_like(x)
    if len(summary_df) >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)
    else:
        coords = np.zeros((len(summary_df), 2), dtype=float)
    out = summary_df.copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    return out, coords


def build_behavior_space_figure(summary_df: pd.DataFrame, env_name: str, color_by: str) -> go.Figure:
    emb_df, _coords = embed_summary(summary_df)
    fig = px.scatter(
        emb_df,
        x="pc1",
        y="pc2",
        color=color_by,
        symbol="is_regime_shift",
        hover_data=[
            "k",
            "t_start",
            "t_end",
            "r_bar",
            "H_bar",
            "switch_rate",
            "anomaly_score",
            "regime_shift_score",
        ],
        color_continuous_scale="Turbo",
        title="{0} Behavior Space".format(ENV_LABELS.get(env_name, env_name)),
    )
    fig.update_traces(marker={"size": 12, "opacity": 0.9, "line": {"width": 0.5, "color": "#1f1f1f"}})
    fig.update_traces(
        customdata=emb_df[["k", "anomaly_score", "regime_shift_score", "switch_rate", "r_bar"]].to_numpy(),
        hovertemplate=(
            "k=%{customdata[0]}<br>anomaly=%{customdata[1]:.4f}<br>regime=%{customdata[2]:.4f}"
            "<br>switch=%{customdata[3]:.4f}<br>r_bar=%{customdata[4]:.4f}<extra></extra>"
        ),
    )
    fig.update_layout(template="plotly_white", xaxis_title="PC1", yaxis_title="PC2")
    return fig


def build_temporal_figure(summary_df: pd.DataFrame, selected_k: Optional[int]) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=summary_df["k"], y=summary_df["r_bar"], mode="lines+markers", name="r_bar"), secondary_y=False)
    fig.add_trace(go.Scatter(x=summary_df["k"], y=summary_df["H_bar"], mode="lines+markers", name="H_bar"), secondary_y=True)
    fig.add_trace(go.Scatter(x=summary_df["k"], y=summary_df["switch_rate"], mode="lines+markers", name="switch_rate"), secondary_y=False)
    fig.add_trace(go.Scatter(x=summary_df["k"], y=summary_df["anomaly_score"], mode="lines+markers", name="anomaly_score"), secondary_y=True)
    fig.add_trace(go.Scatter(x=summary_df["k"], y=summary_df["regime_shift_score"], mode="lines+markers", name="regime_shift"), secondary_y=True)

    anomaly_points = summary_df[summary_df["is_anomaly"]]
    if not anomaly_points.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_points["k"],
                y=anomaly_points["r_bar"],
                mode="markers",
                name="anomaly windows",
                marker={"size": 12, "symbol": "x", "color": "#b22222"},
            ),
            secondary_y=False,
        )

    shift_points = summary_df[summary_df["is_regime_shift"]]
    for shift_k in shift_points["k"].tolist():
        fig.add_vline(x=float(shift_k), line_dash="dot", line_color="#333", opacity=0.45)
    if selected_k is not None:
        fig.add_vline(x=float(selected_k), line_dash="dash", line_color="#111", opacity=0.85)

    fig.update_layout(template="plotly_white", xaxis_title="Window k", legend={"orientation": "h", "y": 1.1})
    fig.update_yaxes(title_text="reward / switch", secondary_y=False)
    fig.update_yaxes(title_text="entropy / anomaly / regime", secondary_y=True)
    return fig


def build_comparison_figure(compare_df: pd.DataFrame, env_name: str, p1: str, p2: str) -> go.Figure:
    view = compare_df[compare_df["js_div"].notna()].copy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cluster JS Divergence", "Cluster State Map"))
    fig.add_trace(
        go.Bar(
            x=view["cluster"].astype(str),
            y=view["js_div"],
            marker={"color": view["js_div"], "colorscale": "Turbo"},
            customdata=view[["n_total", "reward_gap", "q_mean", "mu_mean", "sigma_mean", "rho_mean"]],
            hovertemplate=(
                "cluster=%{x}<br>js=%{y:.4f}<br>n=%{customdata[0]:.0f}"
                "<br>reward_gap=%{customdata[1]:.4f}"
                "<br>state=(%{customdata[2]:.2f}, %{customdata[3]:.2f}, %{customdata[4]:.2f}, %{customdata[5]:.2f})<extra></extra>"
            ),
            name="js_div",
        ),
        row=1,
        col=1,
    )

    if len(view) >= 2:
        state_x = view[["q_mean", "mu_mean", "sigma_mean", "rho_mean", "reward_gap"]].astype(float).to_numpy()
        proj = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(state_x))
    else:
        proj = np.zeros((len(view), 2), dtype=float)
    fig.add_trace(
        go.Scatter(
            x=proj[:, 0],
            y=proj[:, 1],
            mode="markers+text",
            text=view["cluster"].astype(str),
            textposition="top center",
            marker={"size": 12, "color": view["js_div"], "colorscale": "Turbo", "line": {"width": 0.5, "color": "#222"}},
            customdata=view[["cluster", "js_div", "reward_gap"]],
            hovertemplate="cluster=%{customdata[0]}<br>js=%{customdata[1]:.4f}<br>reward_gap=%{customdata[2]:.4f}<extra></extra>",
            name="clusters",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        title="{0}: {1} vs {2}".format(ENV_LABELS.get(env_name, env_name), p1, p2),
        showlegend=False,
    )
    return fig


def build_benchmark_overview_figure(benchmark_df: pd.DataFrame, metric: str) -> go.Figure:
    view = benchmark_df.copy()
    fig = px.bar(
        view,
        x="env",
        y="{0}_mean".format(metric),
        color="policy",
        barmode="group",
        error_y="{0}_std".format(metric),
        title="Benchmark Overview: {0}".format(metric.replace("_", " ").title()),
        category_orders={"env": sorted(view["env"].unique().tolist())},
    )
    fig.update_layout(template="plotly_white", xaxis_title="Environment", yaxis_title=metric)
    return fig


def build_ablation_heatmap(ablation_df: pd.DataFrame, env_name: str, policy_name: str, metric: str) -> go.Figure:
    view = ablation_df[(ablation_df["env"] == env_name) & (ablation_df["policy"] == policy_name)].copy()
    feature_order = ["full", "reward_only", "action_only", "state_only"]
    if view.empty:
        return go.Figure()
    view["feature_group"] = pd.Categorical(view["feature_group"], categories=feature_order, ordered=True)
    view = view.sort_values("feature_group")
    fig = go.Figure(
        data=
        [
            go.Heatmap(
                z=[view["{0}_mean".format(metric)].astype(float).tolist()],
                x=view["feature_group"].astype(str).tolist(),
                y=["metric"],
                colorscale="Turbo",
                colorbar={"title": metric},
                text=["{0:.3f}".format(value) for value in view["{0}_mean".format(metric)].astype(float).tolist()],
                texttemplate="%{text}",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        title="{0} Ablation: {1} / {2}".format(metric.replace("_", " ").title(), ENV_LABELS.get(env_name, env_name), policy_name),
        xaxis_title="Feature group",
        yaxis_title="",
    )
    return fig


def build_feature_export_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    export_cols = ["env", "k", "t_start", "t_end"] + ANALYSIS_FEATURE_COLUMNS + ["is_anomaly", "is_regime_shift"]
    return summary_df[export_cols].copy()
