from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from rlva.src.config import BENCHMARK_REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark and ablation summaries into paper-friendly markdown tables")
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=BENCHMARK_REPORT_DIR / "submission_tables.md")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_markdown(frame: pd.DataFrame, columns: List[str]) -> str:
    if frame.empty:
        return "_No data available._"
    view = frame[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in view.to_numpy().tolist()]
    return "\n".join([header, sep] + rows)


def main() -> None:
    args = parse_args()
    benchmark_df = _read_csv(args.report_dir / "benchmark_table.csv")
    ablation_df = _read_csv(args.report_dir / "ablation_table.csv")

    sections: List[str] = ["# RLVA Submission Tables", ""]
    sections.append("## Benchmark Table")
    sections.append(
        _to_markdown(
            benchmark_df,
            [
                "env",
                "policy",
                "num_seeds",
                "reward_mean_summary",
                "anomaly_auc_summary",
                "reward_baseline_auc_summary",
                "shift_auc_summary",
                "reward_jump_auc_summary",
                "shift_localization_error_summary",
                "reward_jump_localization_error_summary",
            ],
        )
    )
    sections.append("")
    sections.append("## Ablation Table")
    sections.append(
        _to_markdown(
            ablation_df,
            ["env", "policy", "feature_group", "reward_mean_summary", "anomaly_auc_summary", "shift_auc_summary"],
        )
    )
    sections.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections))
    print("Saved paper-friendly tables to: {0}".format(args.out))


if __name__ == "__main__":
    main()
