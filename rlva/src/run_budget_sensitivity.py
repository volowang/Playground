from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_TRACE_DIR, SUMMARY_FEATURE_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budget sensitivity sweeps over trace length, window size, and seed count")
    parser.add_argument("--trace-root", type=Path, default=BENCHMARK_TRACE_DIR)
    parser.add_argument("--pairs", type=str, default="inventory:dqn,traffic:dqn,lunarlander:dqn")
    parser.add_argument("--t-values", type=str, default="300,600")
    parser.add_argument("--l-values", type=str, default="25,50,100")
    parser.add_argument("--seed-counts", type=str, default="1,3,5")
    parser.add_argument(
        "--detail-out",
        type=Path,
        default=BENCHMARK_REPORT_DIR / "budget_sensitivity.csv",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=BENCHMARK_REPORT_DIR / "budget_sensitivity_table.csv",
    )
    return parser.parse_args()


def _parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_pairs(value: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        env_name, policy_name = item.split(":", 1)
        pairs.append((env_name.strip(), policy_name.strip()))
    return pairs


def _best_f1(labels: pd.Series, scores: pd.Series) -> float:
    positives = labels.astype(bool)
    if positives.nunique() < 2:
        return float("nan")
    thresholds = sorted(set(float(value) for value in scores.tolist()))
    best = 0.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int((pred & positives).sum())
        fp = int((pred & ~positives).sum())
        fn = int((~pred & positives).sum())
        precision = tp / float(max(1, tp + fp))
        recall = tp / float(max(1, tp + fn))
        if precision + recall > 0.0:
            best = max(best, 2.0 * precision * recall / (precision + recall))
    return float(best)


def _roc_auc(labels: pd.Series, scores: pd.Series) -> float:
    y = labels.astype(int).to_numpy()
    s = scores.astype(float).to_numpy()
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    pos_ranks = ranks[pos_mask].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _average_precision(labels: pd.Series, scores: pd.Series) -> float:
    y = labels.astype(int).to_numpy()
    s = scores.astype(float).to_numpy()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision * y_sorted).sum() / n_pos)


def _safe_auc(labels: pd.Series, scores: pd.Series) -> Tuple[float, float]:
    if labels.astype(int).nunique() < 2:
        return float("nan"), float("nan")
    return _roc_auc(labels, scores), _average_precision(labels, scores)


def _localization_error(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    labels = frame[label_col].astype(bool).reset_index(drop=True)
    if not labels.any():
        return float("nan")
    pred_idx = int(frame[score_col].astype(float).reset_index(drop=True).idxmax())
    true_indices = labels[labels].index.tolist()
    return float(min(abs(pred_idx - true_idx) for true_idx in true_indices))


def _step_entropy(df: pd.DataFrame) -> np.ndarray:
    probs = df[["p0", "p1", "p2"]].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs), axis=1)


def _window_reward_trend(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), 1)[0])


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
    probs = sorted(
        [
            float(action_freq.get(0, 0.0)),
            float(action_freq.get(1, 0.0)),
            float(action_freq.get(2, 0.0)),
        ],
        reverse=True,
    )
    return float(probs[0] - probs[1])


def _annotate_regimes(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    x = out[SUMMARY_FEATURE_COLUMNS].astype(float).to_numpy()
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = (x - mean) / std
    out["anomaly_score"] = np.sqrt((scaled**2).sum(axis=1))

    shift_scores: List[float] = [0.0]
    for idx in range(1, len(out)):
        prev = out.iloc[idx - 1]
        curr = out.iloc[idx]
        prev_probs = np.array([prev["freq0"], prev["freq1"], prev["freq2"]], dtype=float)
        curr_probs = np.array([curr["freq0"], curr["freq1"], curr["freq2"]], dtype=float)
        js_score = _js_divergence(prev_probs, curr_probs)
        state_jump = float(np.linalg.norm(scaled[idx] - scaled[idx - 1]))
        reward_jump = abs(float(curr["r_bar"]) - float(prev["r_bar"]))
        shift_scores.append(js_score + 0.35 * state_jump + 0.15 * reward_jump)
    out["regime_shift_score"] = shift_scores
    return out


def _summarize_trace(trace: pd.DataFrame, L: int) -> pd.DataFrame:
    df = trace.copy().reset_index(drop=True)
    df["window_k"] = df.index // L
    df["step_entropy"] = _step_entropy(df)
    rows: List[Dict[str, object]] = []
    for k, group in df.groupby("window_k", sort=True):
        action_freq = group["a"].value_counts(normalize=True)
        reward_values = group["r"].to_numpy(dtype=float)
        rows.append(
            {
                "env": str(group["env"].iloc[0]),
                "policy": str(group["policy"].iloc[0]),
                "seed": int(group["seed"].iloc[0]),
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
                "gt_intervention": bool(group["intervention_active"].max() > 0.0),
                "gt_shift_start": bool(group["intervention_start"].max() > 0.0),
            }
        )
    return _annotate_regimes(pd.DataFrame(rows))


def _evaluate_summary(summary_df: pd.DataFrame) -> Dict[str, float]:
    anomaly_auc, anomaly_ap = _safe_auc(summary_df["gt_intervention"], summary_df["anomaly_score"])
    shift_auc, shift_ap = _safe_auc(summary_df["gt_shift_start"], summary_df["regime_shift_score"])
    return {
        "reward_mean": float(summary_df["r_bar"].mean()),
        "anomaly_auc": anomaly_auc,
        "anomaly_ap": anomaly_ap,
        "anomaly_best_f1": _best_f1(summary_df["gt_intervention"], summary_df["anomaly_score"]),
        "shift_auc": shift_auc,
        "shift_ap": shift_ap,
        "shift_best_f1": _best_f1(summary_df["gt_shift_start"], summary_df["regime_shift_score"]),
        "shift_localization_error": _localization_error(
            summary_df,
            score_col="regime_shift_score",
            label_col="gt_shift_start",
        ),
        "num_windows": float(len(summary_df)),
    }


def _trace_paths(trace_root: Path, env_name: str, policy_name: str) -> List[Path]:
    return sorted((trace_root / env_name / policy_name).glob("seed_*.csv"))


def _format_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    return frame.apply(
        lambda row: "{0:.3f} +/- {1:.3f}".format(float(row[mean_col]), float(row[std_col])),
        axis=1,
    )


def main() -> None:
    args = parse_args()
    pairs = _parse_pairs(args.pairs)
    t_values = _parse_csv_ints(args.t_values)
    l_values = _parse_csv_ints(args.l_values)
    seed_counts = _parse_csv_ints(args.seed_counts)

    detail_rows: List[Dict[str, float]] = []
    for env_name, policy_name in pairs:
        trace_paths = _trace_paths(args.trace_root, env_name=env_name, policy_name=policy_name)
        if not trace_paths:
            raise FileNotFoundError(
                "No benchmark traces found for env={0} policy={1} under {2}".format(
                    env_name,
                    policy_name,
                    args.trace_root,
                )
            )
        trace_frames = []
        for path in trace_paths:
            frame = pd.read_csv(path)
            frame["source_path"] = str(path)
            trace_frames.append(frame)
        trace_frames = sorted(trace_frames, key=lambda frame: int(frame["seed"].iloc[0]))

        for t_value in t_values:
            valid_frames = [frame[frame["t"] < t_value].copy() for frame in trace_frames if len(frame[frame["t"] < t_value]) > 0]
            if not valid_frames:
                continue
            for l_value in l_values:
                for seed_count in seed_counts:
                    selected = valid_frames[: min(seed_count, len(valid_frames))]
                    if len(selected) < seed_count:
                        continue
                    for frame in selected:
                        summary_df = _summarize_trace(frame, L=l_value)
                        row: Dict[str, float] = {
                            "env": env_name,
                            "policy": policy_name,
                            "seed": int(frame["seed"].iloc[0]),
                            "T": int(t_value),
                            "L": int(l_value),
                            "seed_count": int(seed_count),
                        }
                        row.update(_evaluate_summary(summary_df))
                        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows).sort_values(
        ["env", "policy", "T", "L", "seed_count", "seed"]
    ).reset_index(drop=True)

    metric_cols = ["reward_mean", "anomaly_auc", "shift_auc", "shift_localization_error", "num_windows"]
    agg_rows: List[Dict[str, float]] = []
    for (env_name, policy_name, t_value, l_value, seed_count), frame in detail_df.groupby(
        ["env", "policy", "T", "L", "seed_count"],
        as_index=False,
    ):
        row: Dict[str, float] = {
            "env": env_name,
            "policy": policy_name,
            "T": int(t_value),
            "L": int(l_value),
            "seed_count": int(seed_count),
        }
        for metric in metric_cols:
            row["{0}_mean".format(metric)] = float(frame[metric].mean())
            row["{0}_std".format(metric)] = float(frame[metric].std(ddof=0))
        agg_rows.append(row)

    table_df = pd.DataFrame(agg_rows).sort_values(["env", "policy", "T", "L", "seed_count"]).reset_index(drop=True)
    for metric in metric_cols:
        table_df["{0}_summary".format(metric)] = _format_mean_std(table_df, metric)

    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.detail_out, index=False)
    table_df.to_csv(args.table_out, index=False)

    display_cols = [
        "env",
        "policy",
        "T",
        "L",
        "seed_count",
        "anomaly_auc_summary",
        "shift_auc_summary",
        "shift_localization_error_summary",
    ]
    print(table_df[display_cols].to_string(index=False))
    print("\nSaved budget sensitivity details to: {0}".format(args.detail_out))
    print("Saved budget sensitivity table to: {0}".format(args.table_out))


if __name__ == "__main__":
    main()
