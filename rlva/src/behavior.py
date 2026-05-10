from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rlva.src.config import SUMMARY_FEATURE_COLUMNS


REQUIRED_COLUMNS = ["env", "t", "q", "mu", "sigma", "rho", "a", "p0", "p1", "p2", "r", "state_delta"]


def _validate_trace(trace: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in trace.columns]
    if missing:
        raise ValueError("Trace is missing required columns: {0}".format(missing))


def _step_entropy(df: pd.DataFrame) -> np.ndarray:
    probs = df[["p0", "p1", "p2"]].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs), axis=1)


def _window_reward_trend(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values.astype(float), 1)[0]
    return float(slope)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, 1.0)
    q = np.clip(q.astype(float), 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def _action_switch_rate(actions: pd.Series) -> float:
    if len(actions) <= 1:
        return 0.0
    changed = actions.to_numpy()[1:] != actions.to_numpy()[:-1]
    return float(np.mean(changed.astype(float)))


def _dominance_gap(action_freq: pd.Series) -> float:
    probs = sorted([float(action_freq.get(0, 0.0)), float(action_freq.get(1, 0.0)), float(action_freq.get(2, 0.0))], reverse=True)
    return float(probs[0] - probs[1])


def annotate_regimes(summary_df: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.copy()
    columns = feature_columns if feature_columns is not None else SUMMARY_FEATURE_COLUMNS
    feature_matrix = out[columns].astype(float).to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix) if len(out) >= 2 else np.zeros_like(feature_matrix)
    out["anomaly_score"] = np.sqrt((scaled ** 2).sum(axis=1))

    shift_scores: List[float] = [0.0]
    for idx in range(1, len(out)):
        prev = out.iloc[idx - 1]
        curr = out.iloc[idx]
        prev_probs = np.array([prev["freq0"], prev["freq1"], prev["freq2"]], dtype=float)
        curr_probs = np.array([curr["freq0"], curr["freq1"], curr["freq2"]], dtype=float)
        js_score = _js_divergence(prev_probs, curr_probs)
        state_jump = float(np.linalg.norm(scaled[idx] - scaled[idx - 1])) if len(out) >= 2 else 0.0
        reward_jump = abs(float(curr["r_bar"]) - float(prev["r_bar"]))
        shift_scores.append(js_score + 0.35 * state_jump + 0.15 * reward_jump)
    out["regime_shift_score"] = shift_scores

    anomaly_cutoff = float(np.percentile(out["anomaly_score"], 85)) if len(out) >= 5 else float(out["anomaly_score"].max())
    shift_cutoff = float(np.percentile(out["regime_shift_score"], 85)) if len(out) >= 5 else float(out["regime_shift_score"].max())
    out["is_anomaly"] = out["anomaly_score"] >= anomaly_cutoff
    out["is_regime_shift"] = out["regime_shift_score"] >= shift_cutoff
    return out


def summarize_trace(trace: pd.DataFrame, L: int) -> pd.DataFrame:
    if L <= 0:
        raise ValueError("L must be positive")
    _validate_trace(trace)

    df = trace.copy().reset_index(drop=True)
    df["window_k"] = df.index // L
    df["step_entropy"] = _step_entropy(df)

    rows: List[dict] = []
    env_name = str(df["env"].iloc[0]) if not df.empty else "unknown"
    for k, group in df.groupby("window_k", sort=True):
        action_freq = group["a"].value_counts(normalize=True)
        reward_values = group["r"].to_numpy(dtype=float)
        has_intervention = "intervention_active" in group.columns
        has_shift_label = "intervention_start" in group.columns
        rows.append(
            {
                "env": env_name,
                "policy": str(group["policy"].iloc[0]) if "policy" in group.columns else "unknown",
                "seed": int(group["seed"].iloc[0]) if "seed" in group.columns else -1,
                "k": int(k),
                "t_start": int(group["t"].iloc[0]),
                "t_end": int(group["t"].iloc[-1]),
                "q_bar": float(group["q"].mean()),
                "mu_bar": float(group["mu"].mean()),
                "sigma_bar": float(group["sigma"].mean()),
                "rho_bar": float(group["rho"].mean()),
                "freq0": float(action_freq.get(0, 0.0)),
                "freq1": float(action_freq.get(1, 0.0)),
                "freq2": float(action_freq.get(2, 0.0)),
                "r_bar": float(group["r"].mean()),
                "H_bar": float(group["step_entropy"].mean()),
                "switch_rate": _action_switch_rate(group["a"]),
                "state_delta_bar": float(group["state_delta"].mean()),
                "state_delta_std": float(group["state_delta"].std(ddof=0)),
                "r_std": float(group["r"].std(ddof=0)),
                "reward_trend": _window_reward_trend(reward_values),
                "dominance_gap": _dominance_gap(action_freq),
                "gt_intervention": bool(group["intervention_active"].max() > 0.0) if has_intervention else False,
                "gt_shift_start": bool(group["intervention_start"].max() > 0.0) if has_shift_label else False,
                "gt_intervention_frac": float(group["intervention_active"].mean()) if has_intervention else 0.0,
            }
        )

    columns = [
        "env",
        "policy",
        "seed",
        "k",
        "t_start",
        "t_end",
        "q_bar",
        "mu_bar",
        "sigma_bar",
        "rho_bar",
        "freq0",
        "freq1",
        "freq2",
        "r_bar",
        "H_bar",
        "switch_rate",
        "state_delta_bar",
        "state_delta_std",
        "r_std",
        "reward_trend",
        "dominance_gap",
        "gt_intervention",
        "gt_shift_start",
        "gt_intervention_frac",
    ]
    summary_df = pd.DataFrame(rows, columns=columns)
    return annotate_regimes(summary_df)
