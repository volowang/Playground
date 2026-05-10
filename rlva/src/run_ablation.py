from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rlva.src.behavior import annotate_regimes
from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR, SUMMARY_FEATURE_COLUMNS
from rlva.src.evaluate_detection import evaluate_group


FEATURE_GROUPS = {
    "full": SUMMARY_FEATURE_COLUMNS,
    "reward_only": ["r_bar", "H_bar", "r_std", "reward_trend"],
    "action_only": ["freq0", "freq1", "freq2", "switch_rate", "dominance_gap"],
    "state_only": ["q_bar", "mu_bar", "sigma_bar", "rho_bar", "state_delta_bar", "state_delta_std"],
    "state_action": ["q_bar", "mu_bar", "sigma_bar", "rho_bar", "state_delta_bar", "state_delta_std", "freq0", "freq1", "freq2", "switch_rate", "dominance_gap"],
    "action_reward": ["freq0", "freq1", "freq2", "switch_rate", "dominance_gap", "r_bar", "H_bar", "r_std", "reward_trend"],
    "no_entropy": [column for column in SUMMARY_FEATURE_COLUMNS if column != "H_bar"],
    "no_switch_rate": [column for column in SUMMARY_FEATURE_COLUMNS if column != "switch_rate"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-group ablations for RLVA detection metrics")
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--detail-out", type=Path, default=BENCHMARK_REPORT_DIR / "ablation_seed_metrics.csv")
    parser.add_argument("--table-out", type=Path, default=BENCHMARK_REPORT_DIR / "ablation_table.csv")
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


def _mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    return frame.apply(lambda row: "{0:.3f} +/- {1:.3f}".format(float(row[mean_col]), float(row[std_col])), axis=1)


def main() -> None:
    args = parse_args()
    frames = _collect_summaries(args.summary_root)

    detail_rows: List[Dict[str, float]] = []
    for frame in frames:
        env_name = str(frame["env"].iloc[0])
        policy_name = str(frame["policy"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        reward_mean = float(frame["r_bar"].mean())
        for group_name, columns in FEATURE_GROUPS.items():
            rescored = annotate_regimes(frame, feature_columns=columns)
            row = {"env": env_name, "policy": policy_name, "seed": seed, "feature_group": group_name, "reward_mean": reward_mean}
            row.update(evaluate_group(rescored))
            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows).sort_values(["env", "policy", "feature_group", "seed"]).reset_index(drop=True)

    metric_cols = [
        "reward_mean",
        "anomaly_auc",
        "anomaly_ap",
        "anomaly_best_f1",
        "anomaly_topk_precision",
        "shift_auc",
        "shift_ap",
        "shift_best_f1",
        "shift_topk_precision",
        "shift_localization_error",
        "shift_hit_at_1",
        "shift_hit_at_3",
    ]
    agg_rows: List[Dict[str, float]] = []
    for (env_name, policy_name, feature_group), frame in detail_df.groupby(["env", "policy", "feature_group"], as_index=False):
        row: Dict[str, float] = {
            "env": env_name,
            "policy": policy_name,
            "feature_group": feature_group,
            "num_seeds": float(frame["seed"].nunique()),
        }
        for metric in metric_cols:
            row["{0}_mean".format(metric)] = float(frame[metric].mean())
            row["{0}_std".format(metric)] = float(frame[metric].std(ddof=0))
        agg_rows.append(row)

    table_df = pd.DataFrame(agg_rows).sort_values(["env", "policy", "feature_group"]).reset_index(drop=True)
    for metric in metric_cols:
        table_df["{0}_summary".format(metric)] = _mean_std(table_df, metric)

    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.detail_out, index=False)
    table_df.to_csv(args.table_out, index=False)

    display_cols = ["env", "policy", "feature_group", "num_seeds"] + ["{0}_summary".format(metric) for metric in metric_cols]
    print(table_df[display_cols].to_string(index=False))
    print("\nSaved ablation seed metrics to: {0}".format(args.detail_out))
    print("Saved ablation table to: {0}".format(args.table_out))


if __name__ == "__main__":
    main()
