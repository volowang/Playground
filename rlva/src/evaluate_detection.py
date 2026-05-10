from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR


ANOMALY_DETECTORS: Sequence[Tuple[str, str]] = (
    ("rlva_anomaly", "anomaly_score"),
    ("reward_collapse", "reward_collapse_score"),
    ("state_outlier", "state_outlier_score"),
)
SHIFT_DETECTORS: Sequence[Tuple[str, str]] = (
    ("rlva_shift", "regime_shift_score"),
    ("reward_jump", "reward_jump_score"),
    ("action_shift", "action_shift_score"),
    ("state_shift", "state_shift_score"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RLVA anomaly and shift scores against intervention ground truth")
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--out", type=Path, default=BENCHMARK_REPORT_DIR / "detection_metrics.csv")
    parser.add_argument("--seed-out", type=Path, default=BENCHMARK_REPORT_DIR / "detection_seed_metrics.csv")
    parser.add_argument("--detector-out", type=Path, default=BENCHMARK_REPORT_DIR / "detector_comparison.csv")
    parser.add_argument("--detector-table-out", type=Path, default=BENCHMARK_REPORT_DIR / "detector_comparison_table.csv")
    return parser.parse_args()


def best_f1(labels: pd.Series, scores: pd.Series) -> float:
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


def safe_auc(labels: pd.Series, scores: pd.Series) -> Tuple[float, float]:
    y = labels.astype(int)
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def reward_collapse_score(frame: pd.DataFrame) -> pd.Series:
    return -frame["r_bar"].astype(float)


def reward_jump_score(frame: pd.DataFrame) -> pd.Series:
    reward = frame["r_bar"].astype(float).reset_index(drop=True)
    return reward.diff().abs().fillna(0.0)


def state_outlier_score(frame: pd.DataFrame) -> pd.Series:
    cols = ["q_bar", "mu_bar", "sigma_bar", "rho_bar", "state_delta_bar", "state_delta_std"]
    values = frame[cols].astype(float).reset_index(drop=True)
    baseline = values.iloc[0].to_numpy(dtype=float)
    centered = values.to_numpy(dtype=float) - baseline
    return pd.Series(np.sqrt((centered ** 2).sum(axis=1)), index=frame.index)


def action_shift_score(frame: pd.DataFrame) -> pd.Series:
    values = frame[["freq0", "freq1", "freq2"]].astype(float).reset_index(drop=True)
    diffs = values.diff().abs().fillna(0.0)
    return diffs.sum(axis=1)


def state_shift_score(frame: pd.DataFrame) -> pd.Series:
    cols = ["q_bar", "mu_bar", "sigma_bar", "rho_bar", "state_delta_bar", "state_delta_std"]
    values = frame[cols].astype(float).reset_index(drop=True)
    diffs = values.diff().fillna(0.0)
    return pd.Series(np.sqrt((diffs.to_numpy() ** 2).sum(axis=1)), index=frame.index)


def topk_precision(labels: pd.Series, scores: pd.Series, k: int) -> float:
    if k <= 0:
        return float("nan")
    ranking = scores.astype(float).sort_values(ascending=False)
    if ranking.empty:
        return float("nan")
    k = min(int(k), len(ranking))
    selected = labels.loc[ranking.index[:k]].astype(bool)
    return float(selected.mean()) if len(selected) else float("nan")


def localization_error(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    labels = frame[label_col].astype(bool).reset_index(drop=True)
    if not labels.any() or score_col not in frame.columns:
        return float("nan")
    scores = frame[score_col].astype(float).reset_index(drop=True)
    pred_idx = int(scores.idxmax())
    true_indices = labels[labels].index.tolist()
    return float(min(abs(pred_idx - true_idx) for true_idx in true_indices)) if true_indices else float("nan")


def localization_hit(frame: pd.DataFrame, score_col: str, label_col: str, tolerance: int) -> float:
    error = localization_error(frame=frame, score_col=score_col, label_col=label_col)
    if np.isnan(error):
        return float("nan")
    return float(error <= float(tolerance))


def collect_summary_frames(summary_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(summary_root.glob("*/*/seed_*.csv")):
        frame = pd.read_csv(csv_path)
        frame["source_path"] = str(csv_path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No benchmark summaries found under {0}".format(summary_root))
    return pd.concat(frames, ignore_index=True)


def _score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy().reset_index(drop=True)
    scored["reward_collapse_score"] = reward_collapse_score(scored)
    scored["reward_jump_score"] = reward_jump_score(scored)
    scored["state_outlier_score"] = state_outlier_score(scored)
    scored["action_shift_score"] = action_shift_score(scored)
    scored["state_shift_score"] = state_shift_score(scored)
    return scored


def _detector_rows_for_task(
    frame: pd.DataFrame,
    *,
    task: str,
    label_col: str,
    detector_specs: Iterable[Tuple[str, str]],
) -> List[Dict[str, float]]:
    labels = frame[label_col].astype(bool)
    positive_count = int(labels.sum())
    rows: List[Dict[str, float]] = []
    for detector_name, score_col in detector_specs:
        auc, ap = safe_auc(labels, frame[score_col])
        row: Dict[str, float] = {
            "task": task,
            "detector": detector_name,
            "score_col": score_col,
            "num_windows": float(len(frame)),
            "num_positive_windows": float(positive_count),
            "auc": auc,
            "ap": ap,
            "best_f1": best_f1(labels, frame[score_col]),
            "topk_precision": topk_precision(labels, frame[score_col], positive_count),
            "localization_error": float("nan"),
            "hit_at_1": float("nan"),
            "hit_at_3": float("nan"),
        }
        if task == "shift":
            row["localization_error"] = localization_error(frame, score_col=score_col, label_col=label_col)
            row["hit_at_1"] = localization_hit(frame, score_col=score_col, label_col=label_col, tolerance=1)
            row["hit_at_3"] = localization_hit(frame, score_col=score_col, label_col=label_col, tolerance=3)
        rows.append(row)
    return rows


def detector_comparison_rows(frame: pd.DataFrame) -> pd.DataFrame:
    scored = _score_frame(frame)
    rows = _detector_rows_for_task(
        scored,
        task="anomaly",
        label_col="gt_intervention",
        detector_specs=ANOMALY_DETECTORS,
    )
    rows.extend(
        _detector_rows_for_task(
            scored,
            task="shift",
            label_col="gt_shift_start",
            detector_specs=SHIFT_DETECTORS,
        )
    )
    return pd.DataFrame(rows)


def evaluate_group(frame: pd.DataFrame) -> Dict[str, float]:
    scored = _score_frame(frame)
    detector_df = detector_comparison_rows(scored)

    def _metric(task: str, detector: str, metric: str) -> float:
        match = detector_df[(detector_df["task"] == task) & (detector_df["detector"] == detector)]
        if match.empty:
            return float("nan")
        return float(match.iloc[0][metric])

    return {
        "num_windows": float(len(scored)),
        "num_intervention_windows": float(scored["gt_intervention"].sum()),
        "num_shift_windows": float(scored["gt_shift_start"].sum()),
        "anomaly_auc": _metric("anomaly", "rlva_anomaly", "auc"),
        "anomaly_ap": _metric("anomaly", "rlva_anomaly", "ap"),
        "anomaly_best_f1": _metric("anomaly", "rlva_anomaly", "best_f1"),
        "anomaly_topk_precision": _metric("anomaly", "rlva_anomaly", "topk_precision"),
        "reward_baseline_auc": _metric("anomaly", "reward_collapse", "auc"),
        "reward_baseline_ap": _metric("anomaly", "reward_collapse", "ap"),
        "reward_baseline_best_f1": _metric("anomaly", "reward_collapse", "best_f1"),
        "reward_baseline_topk_precision": _metric("anomaly", "reward_collapse", "topk_precision"),
        "state_baseline_auc": _metric("anomaly", "state_outlier", "auc"),
        "state_baseline_ap": _metric("anomaly", "state_outlier", "ap"),
        "state_baseline_best_f1": _metric("anomaly", "state_outlier", "best_f1"),
        "state_baseline_topk_precision": _metric("anomaly", "state_outlier", "topk_precision"),
        "shift_auc": _metric("shift", "rlva_shift", "auc"),
        "shift_ap": _metric("shift", "rlva_shift", "ap"),
        "shift_best_f1": _metric("shift", "rlva_shift", "best_f1"),
        "shift_topk_precision": _metric("shift", "rlva_shift", "topk_precision"),
        "shift_localization_error": _metric("shift", "rlva_shift", "localization_error"),
        "shift_hit_at_1": _metric("shift", "rlva_shift", "hit_at_1"),
        "shift_hit_at_3": _metric("shift", "rlva_shift", "hit_at_3"),
        "reward_jump_auc": _metric("shift", "reward_jump", "auc"),
        "reward_jump_ap": _metric("shift", "reward_jump", "ap"),
        "reward_jump_best_f1": _metric("shift", "reward_jump", "best_f1"),
        "reward_jump_topk_precision": _metric("shift", "reward_jump", "topk_precision"),
        "reward_jump_localization_error": _metric("shift", "reward_jump", "localization_error"),
        "reward_jump_hit_at_1": _metric("shift", "reward_jump", "hit_at_1"),
        "reward_jump_hit_at_3": _metric("shift", "reward_jump", "hit_at_3"),
        "action_shift_auc": _metric("shift", "action_shift", "auc"),
        "action_shift_ap": _metric("shift", "action_shift", "ap"),
        "action_shift_best_f1": _metric("shift", "action_shift", "best_f1"),
        "action_shift_topk_precision": _metric("shift", "action_shift", "topk_precision"),
        "action_shift_localization_error": _metric("shift", "action_shift", "localization_error"),
        "action_shift_hit_at_1": _metric("shift", "action_shift", "hit_at_1"),
        "action_shift_hit_at_3": _metric("shift", "action_shift", "hit_at_3"),
        "state_shift_auc": _metric("shift", "state_shift", "auc"),
        "state_shift_ap": _metric("shift", "state_shift", "ap"),
        "state_shift_best_f1": _metric("shift", "state_shift", "best_f1"),
        "state_shift_topk_precision": _metric("shift", "state_shift", "topk_precision"),
        "state_shift_localization_error": _metric("shift", "state_shift", "localization_error"),
        "state_shift_hit_at_1": _metric("shift", "state_shift", "hit_at_1"),
        "state_shift_hit_at_3": _metric("shift", "state_shift", "hit_at_3"),
    }


def aggregate_metrics(data: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_key, frame in data.groupby(group_cols, as_index=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {name: value for name, value in zip(group_cols, key_tuple)}
        row.update(evaluate_group(frame.reset_index(drop=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_detector_rows(data: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for group_key, frame in data.groupby(group_cols, as_index=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        detail = detector_comparison_rows(frame.reset_index(drop=True))
        for name, value in zip(group_cols, key_tuple):
            detail[name] = value
        rows.append(detail)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _format_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    return frame.apply(
        lambda row: "{0:.3f} +/- {1:.3f}".format(float(row["{0}_mean".format(metric)]), float(row["{0}_std".format(metric)])),
        axis=1,
    )


def _summarize_detector_rows(detector_seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "ap", "best_f1", "topk_precision", "localization_error", "hit_at_1", "hit_at_3"]
    rows: List[Dict[str, float]] = []
    for group_key, frame in detector_seed_df.groupby(["env", "policy", "task", "detector"], as_index=False):
        env_name, policy_name, task, detector = group_key
        row: Dict[str, float] = {
            "env": env_name,
            "policy": policy_name,
            "task": task,
            "detector": detector,
            "num_seeds": float(frame["seed"].nunique()),
        }
        for metric in metrics:
            row["{0}_mean".format(metric)] = float(frame[metric].mean())
            row["{0}_std".format(metric)] = float(frame[metric].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["env", "policy", "task", "detector"]).reset_index(drop=True)
    for metric in metrics:
        out["{0}_summary".format(metric)] = _format_mean_std(out, metric)
    return out


def main() -> None:
    args = parse_args()
    data = collect_summary_frames(args.summary_root)

    pooled_df = aggregate_metrics(data=data, group_cols=["env", "policy"])
    overall = {"env": "all", "policy": "all"}
    overall.update(evaluate_group(data.reset_index(drop=True)))
    pooled_df = pd.concat([pooled_df, pd.DataFrame([overall])], ignore_index=True)

    seed_df = aggregate_metrics(data=data, group_cols=["env", "policy", "seed"])
    detector_seed_df = aggregate_detector_rows(data=data, group_cols=["env", "policy", "seed"])
    detector_table_df = _summarize_detector_rows(detector_seed_df)

    for path in [args.out, args.seed_out, args.detector_out, args.detector_table_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    pooled_df.to_csv(args.out, index=False)
    seed_df.to_csv(args.seed_out, index=False)
    detector_seed_df.to_csv(args.detector_out, index=False)
    detector_table_df.to_csv(args.detector_table_out, index=False)

    print(pooled_df.to_string(index=False))
    print("\nSaved pooled metrics to: {0}".format(args.out))
    print("Saved per-seed metrics to: {0}".format(args.seed_out))
    print("Saved detector comparison rows to: {0}".format(args.detector_out))
    print("Saved detector comparison table to: {0}".format(args.detector_table_out))


if __name__ == "__main__":
    main()
