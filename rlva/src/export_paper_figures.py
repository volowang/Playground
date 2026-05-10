from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR, ENV_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready static figures from benchmark report tables")
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=BENCHMARK_REPORT_DIR.parent / "figs")
    return parser.parse_args()


def _save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_stem) + ".png", bbox_inches="tight", dpi=200)


def _best_external_method(frame: pd.DataFrame) -> pd.Series:
    external = frame[~frame["method"].isin(["reward_collapse", "reward_jump"])].copy()
    if external.empty:
        return frame.sort_values(["anomaly_auc_mean", "shift_auc_mean"], ascending=[False, False]).iloc[0]
    return external.sort_values(["anomaly_auc_mean", "shift_auc_mean"], ascending=[False, False]).iloc[0]


def _export_main_benchmark(benchmark_df: pd.DataFrame, out_dir: Path) -> None:
    focus = benchmark_df[
        benchmark_df["env"].isin(["inventory", "traffic"]) | (
            benchmark_df["env"].eq("lunarlander") & benchmark_df["policy"].isin(["dqn", "pg"])
        )
    ].copy()
    focus["label"] = focus.apply(lambda row: "{0}\n{1}".format(row["env"], row["policy"]), axis=1)
    x = np.arange(len(focus))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, focus["anomaly_auc_mean"], width, yerr=focus["anomaly_auc_std"], label="Anomaly AUC", color="#b84a39")
    ax.bar(x + width / 2, focus["shift_auc_mean"], width, yerr=focus["shift_auc_std"], label="Shift AUC", color="#355c7d")
    ax.set_xticks(x)
    ax.set_xticklabels(focus["label"], fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Main Benchmark")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, out_dir / "main_benchmark")
    plt.close(fig)


def _export_baseline_comparison(benchmark_df: pd.DataFrame, baseline_df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [("inventory", "dqn"), ("traffic", "dqn"), ("lunarlander", "dqn")]
    rows: List[Dict[str, float]] = []
    for env_name, policy_name in pairs:
        rlva = benchmark_df[(benchmark_df["env"] == env_name) & (benchmark_df["policy"] == policy_name)]
        base = baseline_df[(baseline_df["env"] == env_name) & (baseline_df["policy"] == policy_name)]
        if rlva.empty or base.empty:
            continue
        rlva_row = rlva.iloc[0]
        reward_jump = base[base["method"] == "reward_jump"].iloc[0]
        external = _best_external_method(base)
        rows.extend(
            [
                {"label": "{0}\nRLVA".format(env_name), "anomaly": float(rlva_row["anomaly_auc_mean"]), "shift": float(rlva_row["shift_auc_mean"])},
                {"label": "{0}\nreward_jump".format(env_name), "anomaly": float(reward_jump["anomaly_auc_mean"]), "shift": float(reward_jump["shift_auc_mean"])},
                {"label": "{0}\n{1}".format(env_name, str(external["method"])), "anomaly": float(external["anomaly_auc_mean"]), "shift": float(external["shift_auc_mean"])},
            ]
        )

    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(x - width / 2, frame["anomaly"], width, label="Anomaly AUC", color="#d95f02")
    ax.bar(x + width / 2, frame["shift"], width, label="Shift AUC", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("RLVA vs Simple Baselines")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, out_dir / "baseline_comparison")
    plt.close(fig)


def _export_seed_stability(seed_df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [("inventory", "dqn"), ("traffic", "dqn"), ("lunarlander", "dqn")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    metrics = ["anomaly_auc", "shift_auc"]
    colors = {"inventory": "#355c7d", "traffic": "#6c8ebf", "lunarlander": "#b84a39"}
    for axis, metric in zip(axes, metrics):
        labels = []
        for idx, (env_name, policy_name) in enumerate(pairs):
            view = seed_df[(seed_df["env"] == env_name) & (seed_df["policy"] == policy_name)].copy()
            labels.append(env_name)
            jitter = np.linspace(-0.12, 0.12, num=len(view))
            axis.scatter(
                np.full(len(view), idx, dtype=float) + jitter,
                view[metric].astype(float),
                color=colors[env_name],
                s=32,
                alpha=0.85,
            )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels)
        axis.set_title(metric.replace("_", " ").title())
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("seed score")
    fig.suptitle("Seed Stability")
    fig.tight_layout()
    _save_figure(fig, out_dir / "seed_stability")
    plt.close(fig)


def _export_budget_sensitivity(sensitivity_df: pd.DataFrame, out_dir: Path) -> None:
    focus = sensitivity_df[(sensitivity_df["L"] == 50) & (sensitivity_df["seed_count"] == 5)].copy()
    pairs = [("inventory", "dqn"), ("traffic", "dqn"), ("lunarlander", "dqn")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    metric_specs = [("anomaly_auc_mean", "Anomaly AUC"), ("shift_auc_mean", "Shift AUC")]
    colors = {"inventory": "#355c7d", "traffic": "#6c8ebf", "lunarlander": "#b84a39"}
    for axis, (metric, title) in zip(axes, metric_specs):
        for env_name, policy_name in pairs:
            view = focus[(focus["env"] == env_name) & (focus["policy"] == policy_name)].sort_values("T")
            axis.plot(view["T"], view[metric], marker="o", linewidth=1.8, color=colors[env_name], label=env_name)
        axis.set_title(title)
        axis.set_xlabel("Trace horizon T")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("mean score")
    axes[1].legend(frameon=False)
    fig.suptitle("Budget Sensitivity (L=50, seeds=5)")
    fig.tight_layout()
    _save_figure(fig, out_dir / "budget_sensitivity")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = pd.read_csv(args.report_dir / "benchmark_table.csv")
    baseline_df = pd.read_csv(args.report_dir / "baseline_detector_table.csv")
    seed_df = pd.read_csv(args.report_dir / "benchmark_seed_metrics.csv")
    sensitivity_df = pd.read_csv(args.report_dir / "budget_sensitivity_table.csv")

    _export_main_benchmark(benchmark_df, args.out_dir)
    _export_baseline_comparison(benchmark_df, baseline_df, args.out_dir)
    _export_seed_stability(seed_df, args.out_dir)
    _export_budget_sensitivity(sensitivity_df, args.out_dir)
    print("Saved paper figures to: {0}".format(args.out_dir))


if __name__ == "__main__":
    main()
