from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR, SUMMARY_FEATURE_COLUMNS


DETECTOR_METHODS = ["reward_collapse", "reward_jump", "mahalanobis", "knn_distance", "pca_recon"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simple external baselines on benchmark summaries")
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument(
        "--seed-out",
        type=Path,
        default=BENCHMARK_REPORT_DIR / "baseline_detector_seed_metrics.csv",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=BENCHMARK_REPORT_DIR / "baseline_detector_table.csv",
    )
    parser.add_argument("--features", type=str, default="summary")
    return parser.parse_args()


def _collect_summaries(summary_root: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(summary_root.glob("*/*/seed_*.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_path"] = str(csv_path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No benchmark summaries found under {0}".format(summary_root))
    return frames


def _best_f1(labels: pd.Series, scores: pd.Series) -> float:
    positives = labels.astype(bool)
    if positives.nunique() < 2:
        return float("nan")

    thresholds = sorted(set(float(value) for value in scores.tolist()))
    if not thresholds:
        return float("nan")

    best = 0.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int((pred & positives).sum())
        fp = int((pred & ~positives).sum())
        fn = int((~pred & positives).sum())
        precision = tp / float(max(1, tp + fp))
        recall = tp / float(max(1, tp + fn))
        if precision + recall == 0.0:
            continue
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
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


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
    y = labels.astype(int)
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return _roc_auc(labels, scores), _average_precision(labels, scores)


def _localization_error(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    labels = frame[label_col].astype(bool).reset_index(drop=True)
    if not labels.any() or score_col not in frame.columns:
        return float("nan")

    scores = frame[score_col].astype(float).reset_index(drop=True)
    pred_idx = int(scores.idxmax())
    true_indices = labels[labels].index.tolist()
    if not true_indices:
        return float("nan")
    return float(min(abs(pred_idx - true_idx) for true_idx in true_indices))


def _reward_jump_score(frame: pd.DataFrame) -> pd.Series:
    reward = frame["r_bar"].astype(float).reset_index(drop=True)
    return reward.diff().abs().fillna(0.0)


def _standardize(frame: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    x = frame[feature_columns].astype(float).to_numpy()
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def _mahalanobis_scores(x_scaled: np.ndarray) -> np.ndarray:
    centered = x_scaled - x_scaled.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + 1e-6 * np.eye(cov.shape[0], dtype=float)
    inv_cov = np.linalg.pinv(cov)
    return np.einsum("ij,jk,ik->i", centered, inv_cov, centered)


def _knn_distance_scores(x_scaled: np.ndarray) -> np.ndarray:
    distances = np.sqrt(((x_scaled[:, None, :] - x_scaled[None, :, :]) ** 2).sum(axis=2))
    sorted_distances = np.sort(distances, axis=1)
    k = max(1, min(5, x_scaled.shape[0] - 1))
    neighbor_slice = sorted_distances[:, 1 : k + 1]
    return neighbor_slice.mean(axis=1)


def _pca_reconstruction_scores(x_scaled: np.ndarray) -> np.ndarray:
    centered = x_scaled - x_scaled.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = max(1, min(2, vt.shape[0]))
    basis = vt[:n_components].T
    projection = centered @ basis @ basis.T
    residual = centered - projection
    return np.sqrt((residual**2).sum(axis=1))


def _detector_scores(frame: pd.DataFrame, method: str, feature_columns: List[str]) -> pd.Series:
    if len(frame) < 3:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)

    x_scaled = _standardize(frame, feature_columns)
    if method == "mahalanobis":
        scores = _mahalanobis_scores(x_scaled)
    elif method == "knn_distance":
        scores = _knn_distance_scores(x_scaled)
    elif method == "pca_recon":
        scores = _pca_reconstruction_scores(x_scaled)
    else:
        raise ValueError("Unknown detector method: {0}".format(method))
    return pd.Series(scores, index=frame.index, dtype=float)


def _evaluate_method(frame: pd.DataFrame, method: str, feature_columns: List[str]) -> Dict[str, float]:
    scored = frame.copy().reset_index(drop=True)

    if method == "reward_collapse":
        anomaly_score = -scored["r_bar"].astype(float)
        shift_score = anomaly_score.diff().abs().fillna(0.0)
    elif method == "reward_jump":
        shift_score = _reward_jump_score(scored)
        anomaly_score = shift_score
    else:
        anomaly_score = _detector_scores(scored, method=method, feature_columns=feature_columns)
        shift_score = anomaly_score.diff().abs().fillna(0.0)

    anomaly_auc, anomaly_ap = _safe_auc(scored["gt_intervention"], anomaly_score)
    shift_auc, shift_ap = _safe_auc(scored["gt_shift_start"], shift_score)
    return {
        "num_windows": float(len(scored)),
        "num_intervention_windows": float(scored["gt_intervention"].sum()),
        "num_shift_windows": float(scored["gt_shift_start"].sum()),
        "anomaly_auc": anomaly_auc,
        "anomaly_ap": anomaly_ap,
        "anomaly_best_f1": _best_f1(scored["gt_intervention"], anomaly_score),
        "shift_auc": shift_auc,
        "shift_ap": shift_ap,
        "shift_best_f1": _best_f1(scored["gt_shift_start"], shift_score),
        "shift_localization_error": _localization_error(
            scored.assign(baseline_shift_score=shift_score),
            score_col="baseline_shift_score",
            label_col="gt_shift_start",
        ),
    }


def _format_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    return frame.apply(
        lambda row: "{0:.3f} +/- {1:.3f}".format(float(row[mean_col]), float(row[std_col])),
        axis=1,
    )


def _aggregate_table(detail_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "anomaly_auc",
        "anomaly_ap",
        "anomaly_best_f1",
        "shift_auc",
        "shift_ap",
        "shift_best_f1",
        "shift_localization_error",
    ]
    agg_rows: List[Dict[str, float]] = []
    for (env_name, policy_name, method_name), frame in detail_df.groupby(
        ["env", "policy", "method"],
        as_index=False,
    ):
        row: Dict[str, float] = {
            "env": env_name,
            "policy": policy_name,
            "method": method_name,
            "num_seeds": float(frame["seed"].nunique()),
        }
        for metric in metric_cols:
            row["{0}_mean".format(metric)] = float(frame[metric].mean())
            row["{0}_std".format(metric)] = float(frame[metric].std(ddof=0))
        agg_rows.append(row)

    table_df = pd.DataFrame(agg_rows).sort_values(["env", "policy", "method"]).reset_index(drop=True)
    for metric in metric_cols:
        table_df["{0}_summary".format(metric)] = _format_mean_std(table_df, metric)
    return table_df


def main() -> None:
    args = parse_args()
    if args.features != "summary":
        raise ValueError("Unsupported feature preset: {0}".format(args.features))

    frames = _collect_summaries(args.summary_root)
    detail_rows: List[Dict[str, float]] = []
    for frame in frames:
        env_name = str(frame["env"].iloc[0])
        policy_name = str(frame["policy"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        for method in DETECTOR_METHODS:
            row: Dict[str, float] = {
                "env": env_name,
                "policy": policy_name,
                "seed": seed,
                "method": method,
                "feature_set": args.features,
            }
            row.update(_evaluate_method(frame, method=method, feature_columns=SUMMARY_FEATURE_COLUMNS))
            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows).sort_values(["env", "policy", "method", "seed"]).reset_index(drop=True)
    table_df = _aggregate_table(detail_df)

    args.seed_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.seed_out, index=False)
    table_df.to_csv(args.table_out, index=False)

    display_cols = [
        "env",
        "policy",
        "method",
        "num_seeds",
        "anomaly_auc_summary",
        "shift_auc_summary",
        "shift_localization_error_summary",
    ]
    print(table_df[display_cols].to_string(index=False))
    print("\nSaved baseline seed metrics to: {0}".format(args.seed_out))
    print("Saved baseline detector table to: {0}".format(args.table_out))


if __name__ == "__main__":
    main()
