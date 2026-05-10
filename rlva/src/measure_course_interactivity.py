from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rlva.src.compare import compare_summary_clusters
from rlva.src.config import (
    BENCHMARK_REPORT_DIR,
    BENCHMARK_SUMMARY_DIR,
    COURSE_INTERACTIVITY_METRICS_PATH,
    COURSE_INTERACTIVITY_REPORT_PATH,
    resolve_course_dataset_path,
)
from rlva.src.viz import (
    build_behavior_space_figure,
    build_benchmark_overview_figure,
    build_comparison_figure,
    build_temporal_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure course-facing backend throughput and front-end latency")
    parser.add_argument("--dataset-path", type=Path, default=resolve_course_dataset_path())
    parser.add_argument("--summary-dir", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--out", type=Path, default=COURSE_INTERACTIVITY_METRICS_PATH)
    parser.add_argument("--markdown-out", type=Path, default=COURSE_INTERACTIVITY_REPORT_PATH)
    parser.add_argument(
        "--dataset-nrows",
        type=int,
        default=0,
        help="Optional row limit for the raw dataset scan. Use 0 to scan the full file.",
    )
    return parser.parse_args()


def _read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


def _parse_seed(path: Path) -> Optional[int]:
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def _choose_probe_case(summary_dir: Path) -> Optional[Dict[str, Any]]:
    preferred_pairs: List[Tuple[str, str]] = [
        ("inventory", "dqn"),
        ("traffic", "dqn"),
        ("lunarlander", "dqn"),
        ("inventory", "ppo"),
    ]

    available = sorted(summary_dir.glob("*/*/seed_*.csv"))
    if not available:
        return None

    selected_path: Optional[Path] = None
    for env_name, policy_name in preferred_pairs:
        for path in available:
            if path.parent.name == policy_name and path.parent.parent.name == env_name:
                selected_path = path
                break
        if selected_path is not None:
            break
    if selected_path is None:
        selected_path = available[0]

    env_name = selected_path.parent.parent.name
    policy_name = selected_path.parent.name
    seed = _parse_seed(selected_path)

    comparison_candidates = sorted((summary_dir / env_name).glob("*/seed_*.csv"))
    compare_policy = None
    compare_path = None
    preferred_comparison = ["ppo", "pg", "heuristic", "random", "dqn"]
    for policy in preferred_comparison:
        if policy == policy_name:
            continue
        for path in comparison_candidates:
            if path.parent.name == policy:
                compare_policy = policy
                compare_path = path
                break
        if compare_path is not None:
            break

    return {
        "env": env_name,
        "policy": policy_name,
        "seed": seed,
        "summary_path": selected_path,
        "compare_policy": compare_policy,
        "compare_path": compare_path,
    }


def _measure(
    label: str,
    layer: str,
    records_processed: int,
    note: str,
    fn: Any,
) -> Dict[str, Any]:
    started = time.perf_counter()
    fn()
    seconds = max(time.perf_counter() - started, 1e-9)
    records_per_second = float(records_processed) / seconds if records_processed > 0 else 0.0
    return {
        "interaction": label,
        "layer": layer,
        "records_processed": int(records_processed),
        "seconds": float(seconds),
        "latency_ms": float(seconds * 1000.0),
        "records_per_second": float(records_per_second),
        "note": note,
    }


def summarize_measurements(measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
    backend_rows = [row for row in measurements if row.get("layer") == "backend"]
    frontend_rows = [row for row in measurements if row.get("layer") == "frontend"]
    backend_rates = [float(row["records_per_second"]) for row in backend_rows if float(row["records_per_second"]) > 0.0]
    frontend_latencies = [float(row["latency_ms"]) for row in frontend_rows]
    backend_total_records = sum(int(row["records_processed"]) for row in backend_rows)
    backend_total_seconds = sum(float(row["seconds"]) for row in backend_rows)
    under_one_second = [value for value in frontend_latencies if value <= 1000.0]

    return {
        "backend_measurements": len(backend_rows),
        "frontend_measurements": len(frontend_rows),
        "backend_peak_records_per_second": max(backend_rates) if backend_rates else 0.0,
        "backend_median_records_per_second": float(pd.Series(backend_rates).median()) if backend_rates else 0.0,
        "backend_weighted_records_per_second": (
            float(backend_total_records) / float(backend_total_seconds) if backend_total_records > 0 and backend_total_seconds > 0 else 0.0
        ),
        "frontend_response_median_ms": float(pd.Series(frontend_latencies).median()) if frontend_latencies else 0.0,
        "frontend_response_max_ms": max(frontend_latencies) if frontend_latencies else 0.0,
        "frontend_actions_under_one_second": len(under_one_second),
        "frontend_meets_one_second_reference": bool(frontend_latencies) and max(frontend_latencies) <= 1000.0,
    }


def measure_course_interactivity(
    dataset_path: Path,
    summary_dir: Path,
    report_dir: Path,
    dataset_nrows: int = 0,
) -> Dict[str, Any]:
    benchmark_path = report_dir / "benchmark_table.csv"
    benchmark_df = _read_csv(benchmark_path)
    measurements: List[Dict[str, Any]] = []

    if dataset_path.exists():
        dataset_rows = _read_csv(dataset_path, nrows=dataset_nrows if dataset_nrows > 0 else None)
        measurements.append(
            _measure(
                "Scan raw course dataset",
                layer="backend",
                records_processed=len(dataset_rows),
                note="Sequential CSV scan of the stored course dataset.",
                fn=lambda: _read_csv(dataset_path, nrows=dataset_nrows if dataset_nrows > 0 else None),
            )
        )

    if benchmark_path.exists():
        measurements.append(
            _measure(
                "Load benchmark report table",
                layer="backend",
                records_processed=len(benchmark_df),
                note="Reads the aggregated evaluation table used by the dashboard.",
                fn=lambda: _read_csv(benchmark_path),
            )
        )

    probe_case = _choose_probe_case(summary_dir)
    probe_summary = pd.DataFrame()
    compare_summary = pd.DataFrame()
    if probe_case is not None:
        probe_summary = _read_csv(Path(probe_case["summary_path"]))
        measurements.append(
            _measure(
                "Load seeded summary",
                layer="backend",
                records_processed=len(probe_summary),
                note="Loads one precomputed window-summary artifact for live inspection.",
                fn=lambda: _read_csv(Path(probe_case["summary_path"])),
            )
        )

        if probe_case.get("compare_path") is not None:
            compare_summary = _read_csv(Path(probe_case["compare_path"]))
            comparison_records = len(probe_summary) + len(compare_summary)
            measurements.append(
                _measure(
                    "Compute policy comparison clusters",
                    layer="backend",
                    records_processed=comparison_records,
                    note="Clusters two controller summaries to prepare coordinated comparison views.",
                    fn=lambda: compare_summary_clusters(probe_summary, compare_summary, n_clusters=8),
                )
            )

    if not probe_summary.empty:
        measurements.append(
            _measure(
                "Build behavior-space view",
                layer="frontend",
                records_processed=len(probe_summary),
                note="Constructs the linked scatter plot used for focused anomaly inspection.",
                fn=lambda: build_behavior_space_figure(probe_summary, env_name=str(probe_case["env"]), color_by="anomaly_score"),
            )
        )
        measurements.append(
            _measure(
                "Build temporal-trend view",
                layer="frontend",
                records_processed=len(probe_summary),
                note="Constructs the linked timeline view for the focused summary windows.",
                fn=lambda: build_temporal_figure(probe_summary, selected_k=int(probe_summary["k"].iloc[0])),
            )
        )

    if not benchmark_df.empty:
        measurements.append(
            _measure(
                "Build benchmark-overview view",
                layer="frontend",
                records_processed=len(benchmark_df),
                note="Constructs the multi-environment overview used in the validation narrative.",
                fn=lambda: build_benchmark_overview_figure(benchmark_df, metric="anomaly_auc"),
            )
        )

    if not probe_summary.empty and not compare_summary.empty and probe_case is not None:
        compare_df = compare_summary_clusters(probe_summary, compare_summary, n_clusters=8)
        measurements.append(
            _measure(
                "Build policy-comparison view",
                layer="frontend",
                records_processed=len(compare_df),
                note="Constructs the cluster-level comparison chart for the active probe case.",
                fn=lambda: build_comparison_figure(
                    compare_df,
                    env_name=str(probe_case["env"]),
                    p1=str(probe_case["policy"]),
                    p2=str(probe_case["compare_policy"]),
                ),
            )
        )

    return {
        "dataset_path": str(dataset_path),
        "dataset_scan_nrows": int(dataset_nrows),
        "benchmark_report_dir": str(report_dir),
        "summary_dir": str(summary_dir),
        "probe_case": {
            "env": probe_case.get("env"),
            "policy": probe_case.get("policy"),
            "seed": probe_case.get("seed"),
            "compare_policy": probe_case.get("compare_policy"),
        }
        if probe_case is not None
        else {},
        "measurements": measurements,
        "summary": summarize_measurements(measurements),
    }


def build_course_interactivity_report(metrics: Dict[str, Any]) -> str:
    summary = metrics.get("summary", {})
    probe_case = metrics.get("probe_case", {})
    measurements = metrics.get("measurements", [])

    lines = [
        "# CS526 Interactivity Report",
        "",
        "## Summary",
        "",
        "- Dataset path: `{0}`.".format(metrics.get("dataset_path", "")),
        "- Probe case: `{0}` / `{1}` seed `{2}` compared against `{3}`.".format(
            probe_case.get("env", "n/a"),
            probe_case.get("policy", "n/a"),
            probe_case.get("seed", "n/a"),
            probe_case.get("compare_policy", "n/a"),
        ),
        "- Backend weighted throughput: {0:,.0f} records per second.".format(float(summary.get("backend_weighted_records_per_second", 0.0))),
        "- Backend peak throughput: {0:,.0f} records per second.".format(float(summary.get("backend_peak_records_per_second", 0.0))),
        "- Front-end median response: {0:.1f} ms.".format(float(summary.get("frontend_response_median_ms", 0.0))),
        "- Front-end slowest response: {0:.1f} ms.".format(float(summary.get("frontend_response_max_ms", 0.0))),
        "- Sub-second front-end actions: {0}/{1}.".format(
            int(summary.get("frontend_actions_under_one_second", 0)),
            int(summary.get("frontend_measurements", 0)),
        ),
        "- One-second interaction target satisfied: `{0}`.".format(
            "yes" if bool(summary.get("frontend_meets_one_second_reference", False)) else "no"
        ),
        "",
        "## Measurement Breakdown",
        "",
    ]
    for row in measurements:
        lines.append(
            "- `{0}` [{1}]: {2} records in {3:.4f} s ({4:,.0f} records/s). {5}".format(
                str(row.get("interaction", "")),
                str(row.get("layer", "")),
                int(row.get("records_processed", 0)),
                float(row.get("seconds", 0.0)),
                float(row.get("records_per_second", 0.0)),
                str(row.get("note", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_course_interactivity_outputs(metrics: Dict[str, Any], out_path: Path, markdown_out: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_out.write_text(build_course_interactivity_report(metrics) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = measure_course_interactivity(
        dataset_path=args.dataset_path,
        summary_dir=args.summary_dir,
        report_dir=args.report_dir,
        dataset_nrows=args.dataset_nrows,
    )
    write_course_interactivity_outputs(metrics=metrics, out_path=args.out, markdown_out=args.markdown_out)
    print("Saved course interactivity metrics to: {0}".format(args.out))
    print("Saved course interactivity report to: {0}".format(args.markdown_out))


if __name__ == "__main__":
    main()
