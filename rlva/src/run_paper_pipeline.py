from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from rlva.src.config import PAPER_ENVIRONMENTS, PAPER_POLICY_FAMILIES


def _csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full RLVA paper pipeline from training through reporting")
    parser.add_argument(
        "--steps",
        type=str,
        default="train,benchmark,evaluate,aggregate,ablation,robustness,exports",
        help="Comma-separated subset of: train,benchmark,evaluate,aggregate,ablation,robustness,exports",
    )
    parser.add_argument("--envs", type=str, default=",".join(PAPER_ENVIRONMENTS))
    parser.add_argument("--train-policies", type=str, default="pg,ppo,dqn")
    parser.add_argument("--benchmark-policies", type=str, default=",".join(PAPER_POLICY_FAMILIES))
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--steps-per-episode", type=int, default=180)
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--window-lengths", type=str, default="25,50,75,100")
    parser.add_argument("--cluster-counts", type=str, default="4,8,12,16")
    parser.add_argument("--intervention-profile", type=str, default="standard", choices=["mild", "standard", "severe"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def _run(module: str, args: List[str], project_root: Path) -> None:
    cmd = [sys.executable, "-m", module] + args
    print("[run] {0}".format(" ".join(cmd)))
    subprocess.check_call(cmd, cwd=str(project_root))


def main() -> None:
    args = parse_args()
    steps = set(_csv_list(args.steps))

    if "train" in steps:
        _run(
            "rlva.src.train_baselines",
            [
                "--envs",
                args.envs,
                "--policies",
                args.train_policies,
                "--seeds",
                args.seeds,
                "--episodes",
                str(args.episodes),
                "--steps-per-episode",
                str(args.steps_per_episode),
            ],
            args.project_root,
        )

    if "benchmark" in steps:
        _run(
            "rlva.src.run_benchmark",
            [
                "--envs",
                args.envs,
                "--policies",
                args.benchmark_policies,
                "--seeds",
                args.seeds,
                "--T",
                str(args.T),
                "--L",
                str(args.L),
                "--intervention-profile",
                args.intervention_profile,
            ],
            args.project_root,
        )

    if "evaluate" in steps:
        _run("rlva.src.evaluate_detection", [], args.project_root)

    if "aggregate" in steps:
        _run("rlva.src.aggregate_benchmark", [], args.project_root)

    if "ablation" in steps:
        _run("rlva.src.run_ablation", [], args.project_root)

    if "robustness" in steps:
        _run(
            "rlva.src.run_robustness",
            [
                "--window-lengths",
                args.window_lengths,
                "--cluster-counts",
                args.cluster_counts,
            ],
            args.project_root,
        )

    if "exports" in steps:
        _run("rlva.src.export_paper_figures", [], args.project_root)
        _run("rlva.src.export_paper_tables", [], args.project_root)
        _run("rlva.src.export_case_studies", [], args.project_root)
        _run("rlva.src.measure_course_interactivity", [], args.project_root)
        _run("rlva.src.export_course_deliverables", [], args.project_root)


if __name__ == "__main__":
    main()
