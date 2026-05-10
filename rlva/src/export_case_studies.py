from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR, BENCHMARK_SUMMARY_DIR, BENCHMARK_TRACE_DIR, ENV_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready qualitative case-study figures from benchmark traces")
    parser.add_argument("--trace-root", type=Path, default=BENCHMARK_TRACE_DIR)
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--pairs", type=str, default="inventory:dqn,traffic:dqn,lunarlander:dqn")
    parser.add_argument("--out-dir", type=Path, default=BENCHMARK_REPORT_DIR.parent / "figs")
    return parser.parse_args()


def _parse_pairs(value: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        env_name, policy_name = item.split(":", 1)
        pairs.append((env_name.strip(), policy_name.strip()))
    return pairs


def _pick_seed(report_dir: Path, env_name: str, policy_name: str) -> int:
    seed_metrics = pd.read_csv(report_dir / "benchmark_seed_metrics.csv")
    view = seed_metrics[(seed_metrics["env"] == env_name) & (seed_metrics["policy"] == policy_name)].copy()
    if view.empty:
        raise ValueError("No benchmark seed metrics for env={0} policy={1}".format(env_name, policy_name))
    best = view.sort_values(["anomaly_auc", "shift_auc"], ascending=[False, False]).iloc[0]
    return int(best["seed"])


def _shade_interventions(ax: plt.Axes, trace_df: pd.DataFrame) -> None:
    active = trace_df["intervention_active"].astype(bool).tolist()
    start_idx = None
    for idx, flag in enumerate(active):
        if flag and start_idx is None:
            start_idx = idx
        if not flag and start_idx is not None:
            ax.axvspan(start_idx, idx - 1, color="#f3d3b4", alpha=0.35, linewidth=0)
            start_idx = None
    if start_idx is not None:
        ax.axvspan(start_idx, len(active) - 1, color="#f3d3b4", alpha=0.35, linewidth=0)


def _save_figure(fig: plt.Figure, out_stem: Path) -> None:
    fig.savefig(str(out_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_stem) + ".png", bbox_inches="tight", dpi=200)


def _plot_case(trace_df: pd.DataFrame, summary_df: pd.DataFrame, env_name: str, policy_name: str, seed: int) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=False)

    axes[0].plot(trace_df["t"], trace_df["r"], color="#1f4e79", linewidth=1.8)
    _shade_interventions(axes[0], trace_df)
    axes[0].set_ylabel("reward")
    axes[0].set_title("{0} / {1} / seed {2}".format(ENV_LABELS.get(env_name, env_name), policy_name.upper(), seed))

    summary_x = summary_df["k"].astype(int)
    axes[1].plot(summary_x, summary_df["anomaly_score"], color="#b84a39", linewidth=1.8, label="anomaly")
    axes[1].plot(summary_x, summary_df["regime_shift_score"], color="#355c7d", linewidth=1.8, label="shift")
    anomaly_points = summary_df[summary_df["gt_intervention"].astype(bool)]
    shift_points = summary_df[summary_df["gt_shift_start"].astype(bool)]
    axes[1].scatter(anomaly_points["k"], anomaly_points["anomaly_score"], color="#d95f02", s=28, zorder=3, label="intervention")
    axes[1].scatter(shift_points["k"], shift_points["regime_shift_score"], color="#1b9e77", s=28, zorder=3, label="shift start")
    axes[1].set_ylabel("summary score")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)

    axes[2].plot(trace_df["t"], trace_df["p0"], color="#6c8ebf", linewidth=1.5, label="p0")
    axes[2].plot(trace_df["t"], trace_df["p1"], color="#93c47d", linewidth=1.5, label="p1")
    axes[2].plot(trace_df["t"], trace_df["p2"], color="#e69138", linewidth=1.5, label="p2")
    _shade_interventions(axes[2], trace_df)
    axes[2].set_ylabel("action prob")
    axes[2].set_xlabel("time step")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for env_name, policy_name in _parse_pairs(args.pairs):
        seed = _pick_seed(args.report_dir, env_name=env_name, policy_name=policy_name)
        trace_path = args.trace_root / env_name / policy_name / "seed_{0}.csv".format(seed)
        summary_path = args.summary_root / env_name / policy_name / "seed_{0}.csv".format(seed)
        trace_df = pd.read_csv(trace_path)
        summary_df = pd.read_csv(summary_path)
        fig = _plot_case(trace_df, summary_df, env_name=env_name, policy_name=policy_name, seed=seed)
        out_stem = args.out_dir / "{0}_{1}_case_study".format(env_name, policy_name)
        _save_figure(fig, out_stem)
        plt.close(fig)
        rows.append({"env": env_name, "policy": policy_name, "seed": seed, "trace_path": str(trace_path), "summary_path": str(summary_path)})

    pd.DataFrame(rows).to_csv(args.out_dir / "case_study_index.csv", index=False)
    print("Saved case-study figures to: {0}".format(args.out_dir))


if __name__ == "__main__":
    main()
