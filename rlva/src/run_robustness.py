from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from rlva.src.behavior import summarize_trace
from rlva.src.compare import compare_summary_clusters
from rlva.src.config import BENCHMARK_POLICIES, BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR, BENCHMARK_TRACE_DIR
from rlva.src.evaluate_detection import evaluate_group


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _format_mean_std(frame: pd.DataFrame, metric: str) -> pd.Series:
    return frame.apply(
        lambda row: "{0:.3f} +/- {1:.3f}".format(float(row["{0}_mean".format(metric)]), float(row["{0}_std".format(metric)])),
        axis=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness sweeps for RLVA benchmark summaries and comparisons")
    parser.add_argument("--trace-root", type=Path, default=BENCHMARK_TRACE_DIR)
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--window-lengths", type=str, default="25,50,75,100")
    parser.add_argument("--cluster-counts", type=str, default="4,8,12,16")
    parser.add_argument("--window-detail-out", type=Path, default=BENCHMARK_REPORT_DIR / "robustness_window_seed_metrics.csv")
    parser.add_argument("--window-table-out", type=Path, default=BENCHMARK_REPORT_DIR / "robustness_window_table.csv")
    parser.add_argument("--cluster-detail-out", type=Path, default=BENCHMARK_REPORT_DIR / "robustness_cluster_seed_metrics.csv")
    parser.add_argument("--cluster-table-out", type=Path, default=BENCHMARK_REPORT_DIR / "robustness_cluster_table.csv")
    return parser.parse_args()


def _collect_trace_paths(trace_root: Path) -> List[Path]:
    paths = sorted(trace_root.glob("*/*/seed_*.csv"))
    if not paths:
        raise FileNotFoundError("No benchmark traces found under {0}".format(trace_root))
    return paths


def _collect_summary_paths(summary_root: Path) -> List[Path]:
    paths = sorted(summary_root.glob("*/*/seed_*.csv"))
    if not paths:
        raise FileNotFoundError("No benchmark summaries found under {0}".format(summary_root))
    return paths


def _window_robustness_rows(trace_paths: Sequence[Path], window_lengths: Iterable[int]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for trace_path in trace_paths:
        trace_df = pd.read_csv(trace_path)
        env_name = str(trace_df["env"].iloc[0])
        policy_name = str(trace_df["policy"].iloc[0])
        seed = int(trace_df["seed"].iloc[0])
        for window_length in window_lengths:
            summary_df = summarize_trace(trace=trace_df, L=window_length)
            metrics = evaluate_group(summary_df)
            row: Dict[str, float] = {
                "env": env_name,
                "policy": policy_name,
                "seed": seed,
                "window_length": float(window_length),
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _summary_map(summary_paths: Sequence[Path]) -> Dict[Tuple[str, str, int], Path]:
    mapping: Dict[Tuple[str, str, int], Path] = {}
    for path in summary_paths:
        frame = pd.read_csv(path, nrows=1)
        mapping[(str(frame["env"].iloc[0]), str(frame["policy"].iloc[0]), int(frame["seed"].iloc[0]))] = path
    return mapping


def _cluster_robustness_rows(summary_paths: Sequence[Path], cluster_counts: Iterable[int]) -> pd.DataFrame:
    summary_lookup = _summary_map(summary_paths)
    keys = sorted(summary_lookup.keys())
    env_seed_pairs = sorted({(env_name, seed) for env_name, _policy, seed in keys})
    rows: List[Dict[str, float]] = []

    for env_name, seed in env_seed_pairs:
        available_policies = sorted(policy_name for lookup_env, policy_name, lookup_seed in keys if lookup_env == env_name and lookup_seed == seed)
        for p1, p2 in combinations(available_policies, 2):
            df1 = pd.read_csv(summary_lookup[(env_name, p1, seed)])
            df2 = pd.read_csv(summary_lookup[(env_name, p2, seed)])
            for cluster_count in cluster_counts:
                compare_df = compare_summary_clusters(df1, df2, n_clusters=cluster_count)
                comparable = compare_df[compare_df["js_div"].notna()].copy()
                row = {
                    "env": env_name,
                    "policy_1": p1,
                    "policy_2": p2,
                    "seed": int(seed),
                    "cluster_count": float(cluster_count),
                    "num_clusters": float(len(compare_df)),
                    "num_comparable_clusters": float(len(comparable)),
                    "top1_js_div": float(comparable["js_div"].iloc[0]) if not comparable.empty else float("nan"),
                    "top3_js_div_mean": float(comparable["js_div"].head(3).mean()) if not comparable.empty else float("nan"),
                    "top1_reward_gap": float(comparable["reward_gap"].iloc[0]) if not comparable.empty else float("nan"),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_key, group in frame.groupby(list(group_cols), as_index=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        row: Dict[str, float] = {name: value for name, value in zip(group_cols, key_tuple)}
        row["num_seeds"] = float(group["seed"].nunique())
        for metric in metric_cols:
            row["{0}_mean".format(metric)] = float(group[metric].mean())
            row["{0}_std".format(metric)] = float(group[metric].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)
    for metric in metric_cols:
        out["{0}_summary".format(metric)] = _format_mean_std(out, metric)
    return out


def main() -> None:
    args = parse_args()
    window_lengths = _parse_int_list(args.window_lengths)
    cluster_counts = _parse_int_list(args.cluster_counts)

    trace_paths = _collect_trace_paths(args.trace_root)
    summary_paths = _collect_summary_paths(args.summary_root)

    window_detail_df = _window_robustness_rows(trace_paths, window_lengths)
    window_metrics = [
        "anomaly_auc",
        "anomaly_ap",
        "anomaly_best_f1",
        "shift_auc",
        "shift_ap",
        "shift_best_f1",
        "shift_localization_error",
        "shift_hit_at_1",
        "shift_hit_at_3",
    ]
    window_table_df = _aggregate(
        frame=window_detail_df,
        group_cols=["env", "policy", "window_length"],
        metric_cols=window_metrics,
    )

    cluster_detail_df = _cluster_robustness_rows(summary_paths, cluster_counts)
    cluster_table_df = _aggregate(
        frame=cluster_detail_df,
        group_cols=["env", "policy_1", "policy_2", "cluster_count"],
        metric_cols=["num_clusters", "num_comparable_clusters", "top1_js_div", "top3_js_div_mean", "top1_reward_gap"],
    )

    for path in [args.window_detail_out, args.window_table_out, args.cluster_detail_out, args.cluster_table_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    window_detail_df.to_csv(args.window_detail_out, index=False)
    window_table_df.to_csv(args.window_table_out, index=False)
    cluster_detail_df.to_csv(args.cluster_detail_out, index=False)
    cluster_table_df.to_csv(args.cluster_table_out, index=False)

    print(window_table_df.to_string(index=False))
    print("\nSaved window robustness seed metrics to: {0}".format(args.window_detail_out))
    print("Saved window robustness table to: {0}".format(args.window_table_out))
    print("Saved cluster robustness seed metrics to: {0}".format(args.cluster_detail_out))
    print("Saved cluster robustness table to: {0}".format(args.cluster_table_out))


if __name__ == "__main__":
    main()
