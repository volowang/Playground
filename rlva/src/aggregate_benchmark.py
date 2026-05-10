from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR
from rlva.src.evaluate_detection import aggregate_metrics, collect_summary_frames


METRIC_COLUMNS = [
    "reward_mean",
    "reward_std",
    "anomaly_auc",
    "anomaly_ap",
    "anomaly_best_f1",
    "anomaly_topk_precision",
    "reward_baseline_auc",
    "reward_baseline_ap",
    "reward_baseline_best_f1",
    "reward_baseline_topk_precision",
    "state_baseline_auc",
    "state_baseline_ap",
    "state_baseline_best_f1",
    "state_baseline_topk_precision",
    "shift_auc",
    "shift_ap",
    "shift_best_f1",
    "shift_topk_precision",
    "reward_jump_auc",
    "reward_jump_ap",
    "reward_jump_best_f1",
    "reward_jump_topk_precision",
    "action_shift_auc",
    "action_shift_ap",
    "action_shift_best_f1",
    "action_shift_topk_precision",
    "state_shift_auc",
    "state_shift_ap",
    "state_shift_best_f1",
    "state_shift_topk_precision",
    "shift_localization_error",
    "shift_hit_at_1",
    "shift_hit_at_3",
    "reward_jump_localization_error",
    "reward_jump_hit_at_1",
    "reward_jump_hit_at_3",
    "action_shift_localization_error",
    "action_shift_hit_at_1",
    "action_shift_hit_at_3",
    "state_shift_localization_error",
    "state_shift_hit_at_1",
    "state_shift_hit_at_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed benchmark results into publication-style tables")
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--seed-out", type=Path, default=BENCHMARK_REPORT_DIR / "benchmark_seed_metrics.csv")
    parser.add_argument("--table-out", type=Path, default=BENCHMARK_REPORT_DIR / "benchmark_table.csv")
    return parser.parse_args()


def _add_reward_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (env_name, policy_name, seed), frame in data.groupby(["env", "policy", "seed"], as_index=False):
        rows.append(
            {
                "env": env_name,
                "policy": policy_name,
                "seed": int(seed),
                "reward_mean": float(frame["r_bar"].mean()),
                "reward_std": float(frame["r_bar"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _format_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    return frame.apply(lambda row: "{0:.3f} +/- {1:.3f}".format(float(row[mean_col]), float(row[std_col])), axis=1)


def main() -> None:
    args = parse_args()
    data = collect_summary_frames(args.summary_root)
    seed_metrics = aggregate_metrics(data=data, group_cols=["env", "policy", "seed"])
    reward_stats = _add_reward_stats(data)
    seed_metrics = seed_metrics.merge(reward_stats, on=["env", "policy", "seed"], how="left")

    agg_rows: List[Dict[str, float]] = []
    for (env_name, policy_name), frame in seed_metrics.groupby(["env", "policy"], as_index=False):
        row: Dict[str, float] = {"env": env_name, "policy": policy_name, "num_seeds": float(frame["seed"].nunique())}
        for metric in METRIC_COLUMNS:
            row["{0}_mean".format(metric)] = float(frame[metric].mean())
            row["{0}_std".format(metric)] = float(frame[metric].std(ddof=0))
        agg_rows.append(row)

    table_df = pd.DataFrame(agg_rows).sort_values(["env", "policy"]).reset_index(drop=True)
    for metric in METRIC_COLUMNS:
        table_df["{0}_summary".format(metric)] = _format_mean_std(table_df, metric)

    args.seed_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(args.seed_out, index=False)
    table_df.to_csv(args.table_out, index=False)

    display_cols = ["env", "policy", "num_seeds"] + ["{0}_summary".format(metric) for metric in METRIC_COLUMNS]
    print(table_df[display_cols].to_string(index=False))
    print("\nSaved per-seed metrics to: {0}".format(args.seed_out))
    print("Saved benchmark table to: {0}".format(args.table_out))


if __name__ == "__main__":
    main()
