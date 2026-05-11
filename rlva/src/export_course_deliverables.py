from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rlva.src.config import (
    BENCHMARK_REPORT_DIR,
    BENCHMARK_SUMMARY_DIR,
    BENCHMARK_TRACE_DIR,
    COURSE_DATASET_REPORT_DIR,
    COURSE_INTERACTIVITY_METRICS_PATH,
    COURSE_TASK_PERFORMANCE_METRICS_PATH,
    COURSE_SUBMISSION_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a CS526 final-project brief from benchmark artifacts")
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--trace-dir", type=Path, default=BENCHMARK_TRACE_DIR)
    parser.add_argument("--summary-dir", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--out", type=Path, default=BENCHMARK_REPORT_DIR / "cs526_final_project_brief.md")
    parser.add_argument("--evidence-out", type=Path, default=BENCHMARK_REPORT_DIR / "course_evidence_report.md")
    parser.add_argument("--submission-dir", type=Path, default=COURSE_SUBMISSION_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_large_dataset_metadata() -> Dict[str, object]:
    path = COURSE_DATASET_REPORT_DIR / "large_dataset_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_interactivity_metrics() -> Dict[str, object]:
    if not COURSE_INTERACTIVITY_METRICS_PATH.exists():
        return {}
    return json.loads(COURSE_INTERACTIVITY_METRICS_PATH.read_text(encoding="utf-8"))


def _read_task_performance_metrics() -> Dict[str, object]:
    if not COURSE_TASK_PERFORMANCE_METRICS_PATH.exists():
        return {}
    return json.loads(COURSE_TASK_PERFORMANCE_METRICS_PATH.read_text(encoding="utf-8"))


def _format_number(value: float, digits: int = 3) -> str:
    return "{0:.{1}f}".format(float(value), digits)


def _format_summary(mean_value: float, std_value: float, digits: int = 3) -> str:
    return "{0} +/- {1}".format(_format_number(mean_value, digits), _format_number(std_value, digits))


def _format_bytes(num_bytes: Any) -> str:
    value = _safe_float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return "{0:.2f} {1}".format(value, unit)


def _escape_markdown_cell(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def _markdown_table_from_records(records: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> List[str]:
    if not records:
        return ["No rows available.", ""]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for record in records:
        lines.append("| " + " | ".join(_escape_markdown_cell(record.get(key, "")) for key, _ in columns) + " |")
    lines.append("")
    return lines


def _relative_markdown_path(from_path: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, start=from_path.parent)


def _safe_metric_summary(row: pd.Series, metric: str) -> str:
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    return _format_summary(row.get(mean_col, 0.0), row.get(std_col, 0.0))


def _safe_row(frame: pd.DataFrame, sort_col: str, ascending: bool = False) -> pd.Series:
    if frame.empty or sort_col not in frame.columns:
        return pd.Series(dtype=object)
    return frame.sort_values(sort_col, ascending=ascending).iloc[0]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _course_highlights(
    benchmark_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    robustness_cluster_df: pd.DataFrame,
    interactivity_metrics: Optional[Dict[str, object]] = None,
    task_performance_metrics: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    interactivity_metrics = interactivity_metrics or {}
    task_performance_metrics = task_performance_metrics or {}
    best_anomaly = _safe_row(benchmark_df, "anomaly_auc_mean", ascending=False)
    best_shift = _safe_row(benchmark_df, "shift_auc_mean", ascending=False)
    best_cluster = _safe_row(robustness_cluster_df, "top1_js_div_mean", ascending=False)

    best_baseline = pd.Series(dtype=object)
    if not baseline_df.empty:
        best_baseline = _safe_row(baseline_df, "anomaly_auc_mean", ascending=False)

    return {
        "best_anomaly": {
            "env": str(best_anomaly.get("env", "n/a")),
            "policy": str(best_anomaly.get("policy", "n/a")),
            "score": _format_summary(best_anomaly.get("anomaly_auc_mean", 0.0), best_anomaly.get("anomaly_auc_std", 0.0)),
        }
        if not best_anomaly.empty
        else {},
        "best_shift": {
            "env": str(best_shift.get("env", "n/a")),
            "policy": str(best_shift.get("policy", "n/a")),
            "score": _format_summary(best_shift.get("shift_auc_mean", 0.0), best_shift.get("shift_auc_std", 0.0)),
        }
        if not best_shift.empty
        else {},
        "best_baseline": {
            "method": str(best_baseline.get("method", "n/a")),
            "env": str(best_baseline.get("env", "n/a")),
            "policy": str(best_baseline.get("policy", "n/a")),
            "score": _format_summary(best_baseline.get("anomaly_auc_mean", 0.0), best_baseline.get("anomaly_auc_std", 0.0)),
        }
        if not best_baseline.empty
        else {},
        "best_cluster": {
            "env": str(best_cluster.get("env", "n/a")),
            "pair": "{0} vs {1}".format(str(best_cluster.get("policy_1", "n/a")), str(best_cluster.get("policy_2", "n/a"))),
            "score": _format_summary(best_cluster.get("top1_js_div_mean", 0.0), best_cluster.get("top1_js_div_std", 0.0)),
        }
        if not best_cluster.empty
        else {},
        "interactivity": interactivity_metrics.get("summary", {}) if interactivity_metrics else {},
        "task_performance": task_performance_metrics.get("summary", {}) if task_performance_metrics else {},
    }


def summarize_dataset(trace_dir: Path, summary_dir: Path) -> Dict[str, object]:
    trace_paths = sorted(trace_dir.glob("*/*/seed_*.csv"))
    summary_paths = sorted(summary_dir.glob("*/*/seed_*.csv"))

    total_trace_rows = 0
    total_summary_rows = 0
    total_trace_bytes = 0
    env_counts: Dict[str, int] = {}
    policy_counts: Dict[str, int] = {}

    for path in trace_paths:
        frame = pd.read_csv(path)
        total_trace_rows += int(len(frame))
        total_trace_bytes += int(path.stat().st_size)
        env_name = path.parent.parent.name
        policy = path.parent.name
        env_counts[env_name] = env_counts.get(env_name, 0) + 1
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    for path in summary_paths:
        frame = pd.read_csv(path)
        total_summary_rows += int(len(frame))

    return {
        "trace_file_count": len(trace_paths),
        "summary_file_count": len(summary_paths),
        "total_trace_rows": total_trace_rows,
        "total_summary_rows": total_summary_rows,
        "total_trace_bytes": total_trace_bytes,
        "env_counts": env_counts,
        "policy_counts": policy_counts,
    }


def build_course_brief(
    benchmark_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    robustness_cluster_df: pd.DataFrame,
    robustness_window_df: pd.DataFrame,
    dataset_summary: Dict[str, object],
    interactivity_metrics: Optional[Dict[str, object]] = None,
    task_performance_metrics: Optional[Dict[str, object]] = None,
) -> str:
    interactivity_metrics = interactivity_metrics or _read_interactivity_metrics()
    task_performance_metrics = task_performance_metrics or _read_task_performance_metrics()
    large_dataset_meta = _read_large_dataset_metadata()
    best_anomaly = _safe_row(benchmark_df, "anomaly_auc_mean", ascending=False)
    best_shift = _safe_row(benchmark_df, "shift_auc_mean", ascending=False)
    best_localization = _safe_row(benchmark_df, "shift_localization_error_mean", ascending=True)
    best_cluster = _safe_row(robustness_cluster_df, "top1_js_div_mean", ascending=False)

    best_over_baseline = pd.Series(dtype=object)
    if not benchmark_df.empty and "reward_baseline_auc_mean" in benchmark_df.columns:
        delta_df = benchmark_df.copy()
        delta_df["anomaly_gain_vs_reward"] = delta_df["anomaly_auc_mean"] - delta_df["reward_baseline_auc_mean"]
        best_over_baseline = _safe_row(delta_df, "anomaly_gain_vs_reward", ascending=False)

    best_feature_rows: List[Tuple[str, str, str]] = []
    if not ablation_df.empty:
        grouped = ablation_df.groupby(["env", "policy"], as_index=False)
        for _, frame in grouped:
            row = _safe_row(frame, "anomaly_auc_mean", ascending=False)
            if row.empty:
                continue
            best_feature_rows.append(
                (
                    str(row["env"]),
                    str(row["policy"]),
                    "{0} (anomaly AUC {1})".format(
                        str(row["feature_group"]),
                        _format_summary(row["anomaly_auc_mean"], row["anomaly_auc_std"]),
                    ),
                )
            )

    baseline_note = ""
    if not baseline_df.empty:
        baseline_best = _safe_row(baseline_df, "anomaly_auc_mean", ascending=False)
        if not baseline_best.empty:
            baseline_note = (
                "- Strongest external anomaly baseline: `{method}` on `{env}` / `{policy}` with anomaly AUC {score}.".format(
                    method=str(baseline_best["method"]),
                    env=str(baseline_best["env"]),
                    policy=str(baseline_best["policy"]),
                    score=_format_summary(baseline_best["anomaly_auc_mean"], baseline_best["anomaly_auc_std"]),
                )
            )

    env_counts = dataset_summary.get("env_counts", {})
    policy_counts = dataset_summary.get("policy_counts", {})
    env_inventory = ", ".join(
        "{0} ({1} seeds)".format(env_name, count) for env_name, count in sorted(env_counts.items())
    ) or "No trace artifacts found"
    policy_inventory = ", ".join(
        "{0} ({1} runs)".format(policy, count) for policy, count in sorted(policy_counts.items())
    ) or "No policy artifacts found"
    large_dataset_lines: List[str] = []
    if large_dataset_meta:
        large_dataset_lines = [
            "- Course-scale raw dataset path: `{0}`.".format(str(large_dataset_meta.get("dataset_path", ""))),
            "- Course-scale raw dataset records: {0}.".format(_safe_int(large_dataset_meta.get("total_records", 0))),
            "- Course-scale raw dataset size: {0} bytes.".format(_safe_int(large_dataset_meta.get("actual_bytes", 0))),
            "- Course-scale raw dataset bytes per record: {0:.3f}.".format(_safe_float(large_dataset_meta.get("bytes_per_record", 0.0))),
            "- Course-scale data source: {0}".format(str(large_dataset_meta.get("data_source", ""))),
        ]

    interactivity_lines: List[str] = []
    interactivity_summary = interactivity_metrics.get("summary", {}) if interactivity_metrics else {}
    if interactivity_summary:
        interactivity_lines = [
            "## Measured Interactivity",
            "",
            "- Backend weighted throughput: {0:,.0f} records per second.".format(_safe_float(interactivity_summary.get("backend_weighted_records_per_second", 0.0))),
            "- Backend peak throughput: {0:,.0f} records per second.".format(_safe_float(interactivity_summary.get("backend_peak_records_per_second", 0.0))),
            "- Front-end median response: {0:.1f} ms.".format(_safe_float(interactivity_summary.get("frontend_response_median_ms", 0.0))),
            "- Front-end slowest measured response: {0:.1f} ms.".format(_safe_float(interactivity_summary.get("frontend_response_max_ms", 0.0))),
            "- Sub-second actions: {0}/{1}.".format(
                _safe_int(interactivity_summary.get("frontend_actions_under_one_second", 0)),
                _safe_int(interactivity_summary.get("frontend_measurements", 0)),
            ),
            "- One-second classroom interaction target satisfied: `{0}`.".format(
                "yes" if bool(interactivity_summary.get("frontend_meets_one_second_reference", False)) else "no"
            ),
            "",
        ]

    task_performance_lines: List[str] = []
    task_performance_summary = task_performance_metrics.get("summary", {}) if task_performance_metrics else {}
    if task_performance_summary:
        task_performance_lines = [
            "## User Task Performance",
            "",
            "- Completed user tasks logged: {0}.".format(_safe_int(task_performance_summary.get("completed_tasks", 0))),
            "- Median task completion time: {0:.1f} seconds.".format(_safe_float(task_performance_summary.get("median_duration_seconds", 0.0))),
            "- Median focus changes per task: {0:.1f}.".format(_safe_float(task_performance_summary.get("median_focus_changes", 0.0))),
            "- Median filter changes per task: {0:.1f}.".format(_safe_float(task_performance_summary.get("median_filter_changes", 0.0))),
            "- Average operator confidence: {0:.2f}/5.".format(_safe_float(task_performance_summary.get("average_user_confidence", 0.0))),
            "- Support-signal rate: {0:.1%}.".format(_safe_float(task_performance_summary.get("support_signal_rate", 0.0))),
            "- Export usage rate: {0:.1%}.".format(_safe_float(task_performance_summary.get("export_completion_rate", 0.0))),
            "",
        ]

    lines = [
        "# CS526 Final Project Brief",
        "",
        "## Project Identity",
        "",
        "- Product name: `Decision System Review Studio`.",
        "- Analytic engine: `RLVA: Behavior-Level Diagnostics for Sequential Decision Policies`.",
        "- Project type: user-facing visual analytics system plus reproducible experiment pipeline.",
        "- Primary target users: operations analysts, inventory planners, service managers, control QA engineers, robotics safety reviewers, and supervisors who review controller behavior.",
        "- Core execution scenario: a user selects one of five decision systems, clicks visual evidence to focus a window, controller, signal, or behavior group, compares the active controller against an alternate controller, and exports a focused decision memo.",
        "",
        "## Fundamental Data Questions",
        "",
        "- Which time windows are risky enough to require user review?",
        "- When does the selected system appear to change operating regime?",
        "- Which controller is safer or more stable for the selected system?",
        "",
        "## Dataset and Processing Scale",
        "",
        "- Environments represented in benchmark outputs: {0}.".format(env_inventory),
        "- Policy families represented in benchmark outputs: {0}.".format(policy_inventory),
        "- Benchmark trace files: {0}.".format(dataset_summary.get("trace_file_count", 0)),
        "- Benchmark summary files: {0}.".format(dataset_summary.get("summary_file_count", 0)),
        "- Total trace records processed: {0}.".format(dataset_summary.get("total_trace_rows", 0)),
        "- Total summary windows processed: {0}.".format(dataset_summary.get("total_summary_rows", 0)),
        "- Trace storage footprint: {0} bytes.".format(dataset_summary.get("total_trace_bytes", 0)),
        "",
    ]
    lines.extend(large_dataset_lines)
    lines.extend(
        [
        "## Interface and Interaction Design",
        "",
        "- Front-end: Streamlit dashboard focused on five user-facing scenarios; course evidence is exported as a static report instead of being exposed inside the product UI.",
        "- Coordinated views: system selector, risk-change matrix, priority queue, action mix, timeline, behavior-space plot, focused-signal chart, benchmark heatmap, controller scorecard, controller-comparison charts, case notes, and review exports.",
        "- Interaction components: every visible Plotly chart supports click selection; clicks update the focused window, action filter, selected signal, selected system/controller, score metric, or comparison group.",
        "- Interactivity target: precomputed benchmark artifacts are loaded from CSV so classroom interactions stay in the sub-second to few-second range instead of re-running training online.",
        "",
        "## Development Stack",
        "",
        "- Back-end tools: Python, pandas, scikit-learn, PyTorch, Gymnasium, Plotly export utilities, and reproducible benchmark scripts under `rlva/src/`.",
        "- Front-end tools: Streamlit plus Plotly for interactive charts.",
        "- Reproducibility assets: seeded benchmark traces, summary CSVs, benchmark tables, ablation tables, robustness tables, and exported PNG/PDF figures.",
        "",
    ])
    lines.extend(interactivity_lines)
    lines.extend(task_performance_lines)
    lines.extend(
        [
            "## Quantitative Evaluation Highlights",
            "",
        ]
    )

    if not best_anomaly.empty:
        lines.append(
            "- Best anomaly-detection result: `{env}` / `{policy}` with anomaly AUC {score}.".format(
                env=str(best_anomaly["env"]),
                policy=str(best_anomaly["policy"]),
                score=_format_summary(best_anomaly["anomaly_auc_mean"], best_anomaly["anomaly_auc_std"]),
            )
        )
    if not best_shift.empty:
        lines.append(
            "- Best shift-detection result: `{env}` / `{policy}` with shift AUC {score}.".format(
                env=str(best_shift["env"]),
                policy=str(best_shift["policy"]),
                score=_format_summary(best_shift["shift_auc_mean"], best_shift["shift_auc_std"]),
            )
        )
    if not best_localization.empty:
        lines.append(
            "- Best shift-localization result: `{env}` / `{policy}` with localization error {score}.".format(
                env=str(best_localization["env"]),
                policy=str(best_localization["policy"]),
                score=_format_summary(best_localization["shift_localization_error_mean"], best_localization["shift_localization_error_std"]),
            )
        )
    if not best_over_baseline.empty:
        lines.append(
            "- Largest gain over the reward-collapse baseline: `{env}` / `{policy}` with anomaly-AUC improvement {gain}.".format(
                env=str(best_over_baseline["env"]),
                policy=str(best_over_baseline["policy"]),
                gain=_format_number(best_over_baseline["anomaly_gain_vs_reward"]),
            )
        )
    if baseline_note:
        lines.append(baseline_note)
    if not best_cluster.empty:
        lines.append(
            "- Strongest policy-separation case: `{env}` `{policy_1}` vs `{policy_2}` with top-cluster JS divergence {score} at K={k}.".format(
                env=str(best_cluster["env"]),
                policy_1=str(best_cluster["policy_1"]),
                policy_2=str(best_cluster["policy_2"]),
                score=_format_summary(best_cluster["top1_js_div_mean"], best_cluster["top1_js_div_std"]),
                k=int(best_cluster["cluster_count"]),
            )
        )
    if not robustness_window_df.empty:
        robust_row = _safe_row(robustness_window_df, "shift_hit_at_3_mean", ascending=False)
        if not robust_row.empty:
            lines.append(
                "- Robustness example: `{env}` / `{policy}` at window length {L} reaches shift hit@3 {score}.".format(
                    env=str(robust_row["env"]),
                    policy=str(robust_row["policy"]),
                    L=int(robust_row["window_length"]),
                    score=_format_summary(robust_row["shift_hit_at_3_mean"], robust_row["shift_hit_at_3_std"]),
                )
            )

    lines.extend(
        [
            "",
            "## Ablation Takeaways",
            "",
        ]
    )
    if best_feature_rows:
        for env_name, policy, description in best_feature_rows[:12]:
            lines.append("- `{0}` / `{1}`: best feature configuration is {2}.".format(env_name, policy, description))
    else:
        lines.append("- No ablation artifacts were found.")

    lines.extend(
        [
            "",
            "## Final Demo Checklist",
            "",
            "- Activate the virtual environment in this repository: `source .venv/bin/activate`.",
            "- Run the dashboard: `PYTHONPATH=. streamlit run rlva/src/app.py`.",
            "- Regenerate the paper and course artifacts when needed: `PYTHONPATH=. python -m rlva.src.run_paper_pipeline --steps evaluate,aggregate,ablation,robustness,exports`.",
            "- Export this course brief again after new experiments: `PYTHONPATH=. python -m rlva.src.export_course_deliverables`.",
            "",
            "## Why This Scores Well In CS526 Terms",
            "",
            "- Value of extracted information: the system helps concrete users across five decision systems isolate risky windows, confirm behavior shifts, compare controllers, and leave with a documented action instead of only a chart.",
            "- Methods and models: the project combines RL policies, intervention-aware trace collection, window-level summarization, clustering, anomaly scoring, shift localization, ablation, baseline comparison, and explicit task-performance logging.",
            "- Interactivity: every main chart is clickable and linked, so the dashboard behaves like a coordinated visual analysis tool rather than a static report.",
            "",
        ]
    )
    return "\n".join(lines)


def build_course_evidence_report(
    benchmark_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    robustness_cluster_df: pd.DataFrame,
    robustness_window_df: pd.DataFrame,
    dataset_summary: Dict[str, object],
    interactivity_metrics: Optional[Dict[str, object]],
    task_performance_metrics: Optional[Dict[str, object]],
    submission_dir: Path,
    out_path: Path,
) -> str:
    fig_dir = BENCHMARK_REPORT_DIR.parent / "figs"
    large_dataset_meta = _read_large_dataset_metadata()
    interactivity_metrics = interactivity_metrics or {}
    task_performance_metrics = task_performance_metrics or {}
    interactivity_summary = interactivity_metrics.get("summary", {}) if interactivity_metrics else {}
    task_summary = task_performance_metrics.get("summary", {}) if task_performance_metrics else {}
    task_rows = task_performance_metrics.get("rows", []) if task_performance_metrics else []

    image_descriptions = {
        "main_benchmark": "Main benchmark view used in the course evidence workspace to explain anomaly, shift, and reward-jump performance together.",
        "baseline_comparison": "Visual comparison between RLVA and simpler detector families.",
        "seed_stability": "Repeated-run stability evidence across random seeds.",
        "budget_sensitivity": "Sensitivity to trace horizon, window length, and analysis budget.",
        "traffic_dqn_case_study": "Traffic-control qualitative case used for the user-facing incident-review scenario.",
        "inventory_dqn_case_study": "Inventory-control case study used as operational transfer evidence.",
        "lunarlander_dqn_case_study": "Standard-control case study showing the method is not limited to custom operational systems.",
    }
    image_paths = sorted(fig_dir.glob("*.png"))

    best_anomaly = _safe_row(benchmark_df, "anomaly_auc_mean", ascending=False)
    best_shift = _safe_row(benchmark_df, "shift_auc_mean", ascending=False)
    best_localization = _safe_row(benchmark_df, "shift_localization_error_mean", ascending=True)

    benchmark_records: List[Dict[str, Any]] = []
    if not benchmark_df.empty:
        for _, row in benchmark_df.sort_values(["anomaly_auc_mean", "shift_auc_mean"], ascending=[False, False]).head(10).iterrows():
            benchmark_records.append(
                {
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "anomaly_auc": _safe_metric_summary(row, "anomaly_auc"),
                    "shift_auc": _safe_metric_summary(row, "shift_auc"),
                    "reward_jump_auc": _safe_metric_summary(row, "reward_jump_auc"),
                    "localization_error": _safe_metric_summary(row, "shift_localization_error"),
                }
            )

    baseline_records: List[Dict[str, Any]] = []
    if not baseline_df.empty:
        for _, row in baseline_df.sort_values("anomaly_auc_mean", ascending=False).head(12).iterrows():
            baseline_records.append(
                {
                    "method": row.get("method", ""),
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "anomaly_auc": _safe_metric_summary(row, "anomaly_auc"),
                    "shift_auc": _safe_metric_summary(row, "shift_auc"),
                    "localization_error": _safe_metric_summary(row, "shift_localization_error"),
                }
            )

    detection_records: List[Dict[str, Any]] = []
    if not detection_df.empty:
        for _, row in detection_df[detection_df["env"] != "all"].head(12).iterrows():
            detection_records.append(
                {
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "anomaly_auc": _format_number(row.get("anomaly_auc", 0.0)),
                    "shift_auc": _format_number(row.get("shift_auc", 0.0)),
                    "reward_jump_auc": _format_number(row.get("reward_jump_auc", 0.0)),
                }
            )

    ablation_records: List[Dict[str, Any]] = []
    if not ablation_df.empty:
        for (env_name, policy), frame in ablation_df.groupby(["env", "policy"]):
            row = _safe_row(frame, "anomaly_auc_mean", ascending=False)
            if row.empty:
                continue
            ablation_records.append(
                {
                    "env": env_name,
                    "policy": policy,
                    "best_feature_group": row.get("feature_group", ""),
                    "anomaly_auc": _safe_metric_summary(row, "anomaly_auc"),
                    "shift_auc": _safe_metric_summary(row, "shift_auc"),
                }
            )
        ablation_records = ablation_records[:15]

    sensitivity_records: List[Dict[str, Any]] = []
    if not sensitivity_df.empty:
        for _, row in sensitivity_df.sort_values("shift_auc_mean", ascending=False).head(12).iterrows():
            sensitivity_records.append(
                {
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "T": _safe_int(row.get("T", 0)),
                    "L": _safe_int(row.get("L", 0)),
                    "seed_count": _safe_int(row.get("seed_count", 0)),
                    "shift_auc": _safe_metric_summary(row, "shift_auc"),
                    "localization_error": _safe_metric_summary(row, "shift_localization_error"),
                }
            )

    seed_records: List[Dict[str, Any]] = []
    if not seed_df.empty:
        for _, row in seed_df.sort_values("anomaly_auc", ascending=False).head(12).iterrows():
            seed_records.append(
                {
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "seed": _safe_int(row.get("seed", 0)),
                    "anomaly_auc": _format_number(row.get("anomaly_auc", 0.0)),
                    "shift_auc": _format_number(row.get("shift_auc", 0.0)),
                    "reward_jump_auc": _format_number(row.get("reward_jump_auc", 0.0)),
                }
            )

    robust_window_records: List[Dict[str, Any]] = []
    if not robustness_window_df.empty:
        for _, row in robustness_window_df.sort_values("shift_hit_at_3_mean", ascending=False).head(12).iterrows():
            robust_window_records.append(
                {
                    "env": row.get("env", ""),
                    "policy": row.get("policy", ""),
                    "window_length": _safe_int(row.get("window_length", 0)),
                    "shift_auc": _safe_metric_summary(row, "shift_auc"),
                    "hit_at_3": _safe_metric_summary(row, "shift_hit_at_3"),
                }
            )

    robust_cluster_records: List[Dict[str, Any]] = []
    if not robustness_cluster_df.empty:
        for _, row in robustness_cluster_df.sort_values("top1_js_div_mean", ascending=False).head(12).iterrows():
            robust_cluster_records.append(
                {
                    "env": row.get("env", ""),
                    "policy_pair": "{0} vs {1}".format(row.get("policy_1", ""), row.get("policy_2", "")),
                    "cluster_count": _safe_int(row.get("cluster_count", 0)),
                    "top_js_div": _safe_metric_summary(row, "top1_js_div"),
                }
            )

    submission_records: List[Dict[str, Any]] = []
    if submission_dir.exists():
        for path in sorted(submission_dir.glob("*.md")):
            submission_records.append({"artifact": path.name, "size": _format_bytes(path.stat().st_size), "path": str(path)})

    artifact_records: List[Dict[str, Any]] = []
    for path in sorted(BENCHMARK_REPORT_DIR.glob("*")):
        if path.suffix.lower() in {".csv", ".json", ".md", ".jsonl"}:
            artifact_records.append({"artifact": path.name, "size": _format_bytes(path.stat().st_size), "path": str(path)})

    lines: List[str] = [
        "# Course Evidence Report",
        "",
        "This static report replaces the former in-app Course Evidence Workspace. The Streamlit application now stays focused on the final user-facing product, while this document preserves the grading evidence as text, tables, and embedded images.",
        "",
        "## Course Rubric Mapping",
        "",
        "- Value of extracted information: the dashboard helps users across traffic, inventory, queueing, control QA, and robotics safety isolate risky windows, confirm operating changes, compare controllers, and export action-ready memos.",
        "- Methods and models: RLVA combines seeded trace generation, window-level summarization, anomaly scoring, shift localization, clustering, controller comparison, baseline comparison, ablation, and robustness checks.",
        "- Final application interactivity: every main chart is clickable and links filters, ranked windows, behavior maps, timelines, comparison views, case notes, task logs, and exports.",
        "- This report holds the course-facing evidence so the product interface is not cluttered with developer or instructor-only screens.",
        "",
        "## Dataset Layer",
        "",
        "- Benchmark trace files: {0}.".format(dataset_summary.get("trace_file_count", 0)),
        "- Benchmark summary files: {0}.".format(dataset_summary.get("summary_file_count", 0)),
        "- Processed trace records: {0}.".format(dataset_summary.get("total_trace_rows", 0)),
        "- Processed summary windows: {0}.".format(dataset_summary.get("total_summary_rows", 0)),
        "- Benchmark trace storage footprint: {0}.".format(_format_bytes(dataset_summary.get("total_trace_bytes", 0))),
    ]
    if large_dataset_meta:
        lines.extend(
            [
                "- Course-scale raw dataset: `{0}`.".format(large_dataset_meta.get("dataset_path", "")),
                "- Course-scale raw records: {0}.".format(_safe_int(large_dataset_meta.get("total_records", 0))),
                "- Course-scale raw size: {0}.".format(_format_bytes(large_dataset_meta.get("actual_bytes", 0))),
                "- Delivery format: {0}.".format(large_dataset_meta.get("delivery_format", "")),
                "- Provenance: {0}.".format(large_dataset_meta.get("data_source", "")),
            ]
        )
    lines.extend(
        [
            "",
            "## Trust Question 1: Does It Catch Unusual Behavior?",
            "",
            "The course workspace answered this with benchmark rankings, unusual-behavior scores, baseline comparisons, and example cases. The user-facing dashboard uses the same evidence in the risk matrix and priority queue.",
            "",
        ]
    )
    if not best_anomaly.empty:
        lines.append(
            "- Best anomaly result: `{0}` / `{1}` with anomaly AUC {2}.".format(
                best_anomaly.get("env", ""),
                best_anomaly.get("policy", ""),
                _safe_metric_summary(best_anomaly, "anomaly_auc"),
            )
        )
    lines.extend(["", "### Top Benchmark Rows", ""])
    lines.extend(
        _markdown_table_from_records(
            benchmark_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("anomaly_auc", "Anomaly AUC"),
                ("shift_auc", "Shift AUC"),
                ("reward_jump_auc", "Reward-Jump AUC"),
                ("localization_error", "Shift Localization Error"),
            ],
        )
    )

    lines.extend(["### Baseline Comparison Evidence", ""])
    lines.extend(
        _markdown_table_from_records(
            baseline_records,
            [
                ("method", "Method"),
                ("env", "Environment"),
                ("policy", "Policy"),
                ("anomaly_auc", "Anomaly AUC"),
                ("shift_auc", "Shift AUC"),
                ("localization_error", "Localization Error"),
            ],
        )
    )

    lines.extend(
        [
            "## Trust Question 2: Does It Catch Operating Changes?",
            "",
            "The former workspace emphasized shift AUC, localization error, sensitivity to time-window settings, and qualitative case studies. This evidence supports the dashboard's change-confidence timeline.",
            "",
        ]
    )
    if not best_shift.empty:
        lines.append(
            "- Best shift result: `{0}` / `{1}` with shift AUC {2}.".format(
                best_shift.get("env", ""),
                best_shift.get("policy", ""),
                _safe_metric_summary(best_shift, "shift_auc"),
            )
        )
    if not best_localization.empty:
        lines.append(
            "- Best localization result: `{0}` / `{1}` with localization error {2}.".format(
                best_localization.get("env", ""),
                best_localization.get("policy", ""),
                _safe_metric_summary(best_localization, "shift_localization_error"),
            )
        )
    lines.extend(["", "### Detection Snapshot", ""])
    lines.extend(
        _markdown_table_from_records(
            detection_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("anomaly_auc", "Anomaly AUC"),
                ("shift_auc", "Shift AUC"),
                ("reward_jump_auc", "Reward-Jump AUC"),
            ],
        )
    )
    lines.extend(["### Budget Sensitivity", ""])
    lines.extend(
        _markdown_table_from_records(
            sensitivity_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("T", "Trace Horizon"),
                ("L", "Window Length"),
                ("seed_count", "Seeds"),
                ("shift_auc", "Shift AUC"),
                ("localization_error", "Localization Error"),
            ],
        )
    )

    lines.extend(
        [
            "## Trust Question 3: Does It Stay Reliable And Responsive?",
            "",
            "The course workspace grouped seed stability, robustness checks, interaction latency, and user-task metrics here. Those numbers are preserved below.",
            "",
            "### Repeated-Run Stability",
            "",
        ]
    )
    lines.extend(
        _markdown_table_from_records(
            seed_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("seed", "Seed"),
                ("anomaly_auc", "Anomaly AUC"),
                ("shift_auc", "Shift AUC"),
                ("reward_jump_auc", "Reward-Jump AUC"),
            ],
        )
    )
    lines.extend(["### Feature-Ablation Evidence", ""])
    lines.extend(
        _markdown_table_from_records(
            ablation_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("best_feature_group", "Best Feature Group"),
                ("anomaly_auc", "Anomaly AUC"),
                ("shift_auc", "Shift AUC"),
            ],
        )
    )
    lines.extend(["### Window-Length Robustness", ""])
    lines.extend(
        _markdown_table_from_records(
            robust_window_records,
            [
                ("env", "Environment"),
                ("policy", "Policy"),
                ("window_length", "Window Length"),
                ("shift_auc", "Shift AUC"),
                ("hit_at_3", "Shift Hit@3"),
            ],
        )
    )
    lines.extend(["### Behavior-Group Robustness", ""])
    lines.extend(
        _markdown_table_from_records(
            robust_cluster_records,
            [
                ("env", "Environment"),
                ("policy_pair", "Policy Pair"),
                ("cluster_count", "Clusters"),
                ("top_js_div", "Top JS Divergence"),
            ],
        )
    )

    lines.extend(["### Interaction Performance", ""])
    if interactivity_summary:
        lines.extend(
            [
                "- Backend weighted throughput: {0:,.0f} records per second.".format(_safe_float(interactivity_summary.get("backend_weighted_records_per_second", 0.0))),
                "- Backend peak throughput: {0:,.0f} records per second.".format(_safe_float(interactivity_summary.get("backend_peak_records_per_second", 0.0))),
                "- Front-end median response: {0:.1f} ms.".format(_safe_float(interactivity_summary.get("frontend_response_median_ms", 0.0))),
                "- Front-end slowest measured response: {0:.1f} ms.".format(_safe_float(interactivity_summary.get("frontend_response_max_ms", 0.0))),
                "- Sub-second front-end actions: {0}/{1}.".format(
                    _safe_int(interactivity_summary.get("frontend_actions_under_one_second", 0)),
                    _safe_int(interactivity_summary.get("frontend_measurements", 0)),
                ),
                "",
            ]
        )
    measurement_records = interactivity_metrics.get("measurements", []) if interactivity_metrics else []
    lines.extend(
        _markdown_table_from_records(
            measurement_records,
            [
                ("interaction", "Interaction"),
                ("layer", "Layer"),
                ("records_processed", "Records"),
                ("latency_ms", "Latency ms"),
                ("records_per_second", "Rows/s"),
                ("note", "Note"),
            ],
        )
    )

    lines.extend(["### User Task Performance", ""])
    if task_summary:
        lines.extend(
            [
                "- Completed user tasks: {0}.".format(_safe_int(task_summary.get("completed_tasks", 0))),
                "- Median task completion time: {0:.1f} seconds.".format(_safe_float(task_summary.get("median_duration_seconds", 0.0))),
                "- Median focus changes: {0:.1f}.".format(_safe_float(task_summary.get("median_focus_changes", 0.0))),
                "- Median filter changes: {0:.1f}.".format(_safe_float(task_summary.get("median_filter_changes", 0.0))),
                "- Average confidence: {0:.2f}/5.".format(_safe_float(task_summary.get("average_user_confidence", 0.0))),
                "- Support-signal rate: {0:.1%}.".format(_safe_float(task_summary.get("support_signal_rate", 0.0))),
                "- Export usage rate: {0:.1%}.".format(_safe_float(task_summary.get("export_completion_rate", 0.0))),
                "- Notebook usage rate: {0:.1%}.".format(_safe_float(task_summary.get("notebook_usage_rate", 0.0))),
                "",
            ]
        )
    lines.extend(
        _markdown_table_from_records(
            task_rows,
            [
                ("workflow", "Workflow"),
                ("role", "Role"),
                ("task", "Task"),
                ("focused_window", "Focused Window"),
                ("decision_level", "Decision Level"),
                ("duration_seconds", "Seconds"),
                ("user_confidence", "Confidence"),
                ("export_used", "Export Used"),
            ],
        )
    )

    lines.extend(
        [
            "## Figure And Image Evidence",
            "",
            "These are the image assets that previously appeared in the course evidence workspace. They are embedded here so the report can be read without opening the application.",
            "",
        ]
    )
    for image_path in image_paths:
        rel_path = _relative_markdown_path(out_path, image_path)
        description = image_descriptions.get(image_path.stem, "Exported visual evidence from the benchmark pipeline.")
        lines.extend(
            [
                "### {0}".format(image_path.stem.replace("_", " ").title()),
                "",
                description,
                "",
                "![{0}]({1})".format(image_path.stem, rel_path),
                "",
            ]
        )

    lines.extend(["## Course Deliverables", ""])
    lines.extend(
        _markdown_table_from_records(
            submission_records,
            [("artifact", "Artifact"), ("size", "Size"), ("path", "Path")],
        )
    )

    lines.extend(
        [
            "## Source Artifact Inventory",
            "",
            "The full CSV and JSON artifacts remain on disk for reproducibility. The tables above intentionally summarize the most important rows; the files below contain the complete evidence.",
            "",
        ]
    )
    lines.extend(
        _markdown_table_from_records(
            artifact_records,
            [("artifact", "Artifact"), ("size", "Size"), ("path", "Path")],
        )
    )
    return "\n".join(lines)


def build_submission_readme(
    dataset_summary: Dict[str, object],
    highlights: Dict[str, Any],
) -> str:
    interactivity = highlights.get("interactivity", {})
    task_performance = highlights.get("task_performance", {})
    return "\n".join(
        [
            "# CS526 Submission Package",
            "",
            "This directory contains the course-facing materials that make the Decision System Review Studio final project easy to grade and easy to present.",
            "",
            "## What Is Included",
            "",
            "- `cs526_final_report_outline.md`: report structure aligned to the course rubric.",
            "- `cs526_presentation_storyboard.md`: suggested slide-by-slide flow for the live demo.",
            "- `cs526_video_script.md`: 3+ minute narration script for the required video.",
            "- `cs526_submission_checklist.md`: final handoff checklist for code, data, report, and media.",
            "- `course_task_performance_report.md`: exported evidence that operators can complete the intended review tasks.",
            "- `../course_evidence_report.md`: static text-and-image replacement for the removed in-app course evidence workspace.",
            "",
            "## Project Snapshot",
            "",
            "- Benchmark traces: {0}".format(dataset_summary.get("trace_file_count", 0)),
            "- Benchmark summaries: {0}".format(dataset_summary.get("summary_file_count", 0)),
            "- Processed trace records: {0}".format(dataset_summary.get("total_trace_rows", 0)),
            "- Best anomaly story: {0}".format(
                "{env} / {policy} ({score})".format(**highlights["best_anomaly"]) if highlights.get("best_anomaly") else "n/a"
            ),
            "- Best shift story: {0}".format(
                "{env} / {policy} ({score})".format(**highlights["best_shift"]) if highlights.get("best_shift") else "n/a"
            ),
            "- Completed user tasks logged: {0}".format(_safe_int(task_performance.get("completed_tasks", 0))),
            "- Median task time: {0:.1f} seconds".format(_safe_float(task_performance.get("median_duration_seconds", 0.0))),
            "- Front-end median response: {0:.1f} ms".format(_safe_float(interactivity.get("frontend_response_median_ms", 0.0))),
            "- Backend weighted throughput: {0:,.0f} records per second".format(_safe_float(interactivity.get("backend_weighted_records_per_second", 0.0))),
            "",
            "## Recommended Presentation Command",
            "",
            "`source .venv/bin/activate && export PYTHONPATH=. && streamlit run rlva/src/app.py`",
            "",
        ]
    )


def build_final_report_outline(
    dataset_summary: Dict[str, object],
    highlights: Dict[str, Any],
) -> str:
    best_anomaly = highlights.get("best_anomaly", {})
    best_shift = highlights.get("best_shift", {})
    best_cluster = highlights.get("best_cluster", {})
    interactivity = highlights.get("interactivity", {})
    task_performance = highlights.get("task_performance", {})
    return "\n".join(
        [
            "# CS526 Final Report Outline",
            "",
            "## 1. Project Identity",
            "",
            "- Product name: `Decision System Review Studio`.",
            "- Analytic engine: `RLVA: Behavior-Level Diagnostics for Sequential Decision Policies`.",
            "- Course framing: interactive visual analytics system plus reproducible benchmark pipeline.",
            "- Target users: traffic analysts, inventory planners, service managers, control QA engineers, robotics safety reviewers, supervisors, and technical reviewers.",
            "",
            "## 2. Data And Scale",
            "",
            "- Raw course dataset target: close to 1 GB.",
            "- Processed benchmark trace records: {0}.".format(dataset_summary.get("total_trace_rows", 0)),
            "- Processed benchmark summary windows: {0}.".format(dataset_summary.get("total_summary_rows", 0)),
            "- Describe why long sequential-decision traces require summarization before interactive exploration.",
            "",
            "## 3. Fundamental Data Questions",
            "",
            "- Which system windows look behaviorally risky?",
            "- When does the selected system change operating regime?",
            "- Which controllers behave differently even when reward differences are small?",
            "",
            "## 4. System Design",
            "",
            "- Back-end: trace collection, window summarization, anomaly scoring, shift localization, controller comparison, ablation, robustness evaluation, and task-performance aggregation.",
            "- Front-end: coordinated views where every main chart is clickable, linking ranked incident evidence, behavior space, temporal trends, comparison plots, notebook annotations, task logging, and export panels.",
            "- Execution scenario: a user selects one of five systems, inspects risky windows, compares controllers, and exports decision-ready evidence.",
            "",
            "## 5. Evaluation",
            "",
            "- Main anomaly result: {0}.".format(
                "{env} / {policy} with anomaly AUC {score}".format(**best_anomaly) if best_anomaly else "n/a"
            ),
            "- Main shift result: {0}.".format(
                "{env} / {policy} with shift AUC {score}".format(**best_shift) if best_shift else "n/a"
            ),
            "- Strongest policy-separation result: {0}.".format(
                "{env} {pair} with top-cluster JS divergence {score}".format(**best_cluster) if best_cluster else "n/a"
            ),
            "- Compare RLVA against reward-only or simpler external baselines.",
            "",
            "## 6. Interactivity And Usability",
            "",
            "- Backend weighted throughput: {0:,.0f} records per second.".format(_safe_float(interactivity.get("backend_weighted_records_per_second", 0.0))),
            "- Front-end median response: {0:.1f} ms.".format(_safe_float(interactivity.get("frontend_response_median_ms", 0.0))),
            "- Front-end slowest response: {0:.1f} ms.".format(_safe_float(interactivity.get("frontend_response_max_ms", 0.0))),
            "- Completed user tasks logged: {0}.".format(_safe_int(task_performance.get("completed_tasks", 0))),
            "- Median task completion time: {0:.1f} seconds.".format(_safe_float(task_performance.get("median_duration_seconds", 0.0))),
            "- Support-signal rate: {0:.1%}.".format(_safe_float(task_performance.get("support_signal_rate", 0.0))),
            "- Explain that click selections update focused windows, signals, systems/controllers, score metrics, and comparison groups while staying within the classroom-response target.",
            "",
            "## 7. Limitations And Next Steps",
            "",
            "- Explain that RLVA is a behavior-analysis engine, not a claim of state-of-the-art RL reward performance.",
            "- Propose broader traffic datasets, more policy families, larger user studies, or streaming extensions as follow-up work.",
            "",
            "## 8. Reproducibility Appendix",
            "",
            "- Environment activation: `source .venv/bin/activate`.",
            "- Demo command: `PYTHONPATH=. streamlit run rlva/src/app.py`.",
            "- Export command: `PYTHONPATH=. python -m rlva.src.run_paper_pipeline --steps evaluate,aggregate,ablation,robustness,exports`.",
            "",
        ]
    )


def build_presentation_storyboard(
    dataset_summary: Dict[str, object],
    highlights: Dict[str, Any],
) -> str:
    best_anomaly = highlights.get("best_anomaly", {})
    best_shift = highlights.get("best_shift", {})
    best_baseline = highlights.get("best_baseline", {})
    interactivity = highlights.get("interactivity", {})
    task_performance = highlights.get("task_performance", {})
    return "\n".join(
        [
            "# CS526 Presentation Storyboard",
            "",
            "## Slide 1. User And Problem",
            "",
            "- Introduce the five primary user groups: traffic, inventory, queueing, control QA, and robotics safety reviewers.",
            "- State the product name and the three user questions: risky window, behavior-change boundary, and safer controller.",
            "",
            "## Slide 2. Data",
            "",
            "- Show that the project uses a persistent course-scale raw trace corpus close to 1 GB and explain why time-varying traces need summarization.",
            "- Mention {0} processed trace records and {1} benchmark trace files.".format(
                dataset_summary.get("total_trace_rows", 0),
                dataset_summary.get("trace_file_count", 0),
            ),
            "",
            "## Slide 3. Workflow Demo",
            "",
            "- Show the user flow: choose system -> click a chart -> inspect focused evidence -> compare controllers -> export memo.",
            "- Emphasize that the interface is coordinated, task-driven, and produces work products for an operator.",
            "",
            "## Slide 4. Live Incident Triage Story",
            "",
            "- Lead with one selected system, then connect it to the strongest anomaly evidence: {0}.".format(
                "{env} / {policy} anomaly AUC {score}".format(**best_anomaly) if best_anomaly else "the current best anomaly case"
            ),
            "- Use ranked evidence, the behavior map, and the timeline to connect the number back to a specific risky window.",
            "",
            "## Slide 5. Behavior Shift Confirmation",
            "",
            "- Lead with {0}.".format(
                "{env} / {policy} shift AUC {score}".format(**best_shift) if best_shift else "the current best shift case"
            ),
            "- Show temporal evidence and explain where the operating-regime boundary appears.",
            "",
            "## Slide 6. Why This Is More Than A Dashboard",
            "",
            "- Compare against {0}.".format(
                "{method} on {env} / {policy} ({score})".format(**best_baseline) if best_baseline else "the strongest baseline detector"
            ),
            "- Explain why behavior summaries plus operator workflow expose structure that reward-only views miss.",
            "",
            "## Slide 7. Usability And Interactivity",
            "",
            "- Report completed user tasks: {0}.".format(_safe_int(task_performance.get("completed_tasks", 0))),
            "- Report median task completion time: {0:.1f} seconds.".format(_safe_float(task_performance.get("median_duration_seconds", 0.0))),
            "- Report backend weighted throughput: {0:,.0f} records/s.".format(_safe_float(interactivity.get("backend_weighted_records_per_second", 0.0))),
            "- Report front-end median response: {0:.1f} ms.".format(_safe_float(interactivity.get("frontend_response_median_ms", 0.0))),
            "- State that every main Plotly chart is clickable and linked to a user decision.",
            "- State clearly that operators can both complete tasks and interact within the one-second classroom-response target by reusing precomputed summaries.",
            "",
            "## Slide 8. Closing",
            "",
            "- Summarize the value of extracted information, methods/models used, task performance, and interface interactivity.",
            "- End with the five-system user dashboard inside the Streamlit application.",
            "",
        ]
    )


def build_video_script(
    highlights: Dict[str, Any],
) -> str:
    best_anomaly = highlights.get("best_anomaly", {})
    best_shift = highlights.get("best_shift", {})
    interactivity = highlights.get("interactivity", {})
    task_performance = highlights.get("task_performance", {})
    return "\n".join(
        [
            "# CS526 Video Script",
            "",
            "## 0:00-0:30 Opening",
            "",
            "This project is Decision System Review Studio, a user-facing visual analytics system powered by RLVA. Instead of trusting only total reward, the system helps users inspect risky windows, detect operating changes, and compare controllers behaviorally across five decision systems.",
            "",
            "## 0:30-1:10 Data And Method",
            "",
            "The project uses a persistent raw trace corpus close to one gigabyte, plus seeded benchmark summaries for five environments and five policy families. RLVA converts long traces into window-level summaries, then uses those windows for incident ranking, shift localization, and cross-strategy comparison.",
            "",
            "## 1:10-2:00 Evidence",
            "",
            "One strong anomaly example is {0}. One strong shift-detection example is {1}. These results are backed by clickable linked views that move from benchmark scores to concrete windows, selected controllers, signals, and comparison groups.".format(
                "{env} / {policy} with anomaly AUC {score}".format(**best_anomaly) if best_anomaly else "the current best anomaly case",
                "{env} / {policy} with shift AUC {score}".format(**best_shift) if best_shift else "the current best shift case",
            ),
            "",
            "## 2:00-2:40 Usability And Interactivity",
            "",
            "For course evaluation, the system also measures whether users complete the intended tasks. The exported reports summarize {0} completed tasks, median task time around {1:.1f} seconds, backend throughput around {2:,.0f} records per second, and median front-end response around {3:.1f} milliseconds.".format(
                _safe_int(task_performance.get("completed_tasks", 0)),
                _safe_float(task_performance.get("median_duration_seconds", 0.0)),
                _safe_float(interactivity.get("backend_weighted_records_per_second", 0.0)),
                _safe_float(interactivity.get("frontend_response_median_ms", 0.0)),
            ),
            "",
            "## 2:40-3:20 Closing",
            "",
            "The key contribution is not a higher reward score. The contribution is a five-system review product that extracts behavior-level evidence, supports interactive exploration, records task performance, and provides quantitative validation plus presentation-ready artifacts for classroom demonstration.",
            "",
        ]
    )


def build_submission_checklist(
    out_path: Path,
    submission_dir: Path,
) -> str:
    return "\n".join(
        [
            "# CS526 Submission Checklist",
            "",
            "## Core Deliverables",
            "",
            "- [ ] Working code runs from `source .venv/bin/activate`.",
            "- [ ] Streamlit demo launches with `PYTHONPATH=. streamlit run rlva/src/app.py`.",
            "- [ ] Final project brief exists at `{0}`.".format(str(out_path)),
            "- [ ] Submission package directory exists at `{0}`.".format(str(submission_dir)),
            "- [ ] Interactivity report exists at `rlva/outputs/benchmark/reports/course_interactivity_report.md`.",
            "- [ ] User task performance report exists at `rlva/outputs/benchmark/reports/course_task_performance_report.md`.",
            "",
            "## Documentation",
            "",
            "- [ ] Final report follows `cs526_final_report_outline.md`.",
            "- [ ] Presentation follows `cs526_presentation_storyboard.md`.",
            "- [ ] Video narration follows `cs526_video_script.md` and lasts at least 3 minutes.",
            "- [ ] Report and presentation cite the 1GB course dataset, benchmark tables, and user-task metrics.",
            "",
            "## Evidence To Show",
            "",
            "- [ ] Fundamental data questions are stated explicitly.",
            "- [ ] Target users and execution scenarios are stated explicitly.",
            "- [ ] Coordinated views and interaction components are demonstrated live.",
            "- [ ] Quantitative evidence includes anomaly, shift, baseline, and robustness results.",
            "- [ ] Interactivity evidence includes backend throughput and front-end response time.",
            "- [ ] Usability evidence includes completed user tasks, task time, and export/notebook usage.",
            "",
        ]
    )


def build_submission_package_contents(
    dataset_summary: Dict[str, object],
    benchmark_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    robustness_cluster_df: pd.DataFrame,
    interactivity_metrics: Optional[Dict[str, object]] = None,
    task_performance_metrics: Optional[Dict[str, object]] = None,
    out_path: Optional[Path] = None,
    submission_dir: Optional[Path] = None,
) -> Dict[str, str]:
    highlights = _course_highlights(
        benchmark_df=benchmark_df,
        baseline_df=baseline_df,
        robustness_cluster_df=robustness_cluster_df,
        interactivity_metrics=interactivity_metrics,
        task_performance_metrics=task_performance_metrics,
    )
    out_path = out_path or (BENCHMARK_REPORT_DIR / "cs526_final_project_brief.md")
    submission_dir = submission_dir or COURSE_SUBMISSION_DIR
    return {
        "cs526_submission_readme.md": build_submission_readme(dataset_summary=dataset_summary, highlights=highlights),
        "cs526_final_report_outline.md": build_final_report_outline(dataset_summary=dataset_summary, highlights=highlights),
        "cs526_presentation_storyboard.md": build_presentation_storyboard(dataset_summary=dataset_summary, highlights=highlights),
        "cs526_video_script.md": build_video_script(highlights=highlights),
        "cs526_submission_checklist.md": build_submission_checklist(out_path=out_path, submission_dir=submission_dir),
    }


def write_submission_package(submission_dir: Path, contents: Dict[str, str]) -> List[Path]:
    submission_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    for filename, content in contents.items():
        path = submission_dir / filename
        path.write_text(content + "\n", encoding="utf-8")
        written_paths.append(path)
    return written_paths


def main() -> None:
    args = parse_args()
    benchmark_df = _read_csv(args.report_dir / "benchmark_table.csv")
    ablation_df = _read_csv(args.report_dir / "ablation_table.csv")
    baseline_df = _read_csv(args.report_dir / "baseline_detector_table.csv")
    detection_df = _read_csv(args.report_dir / "detection_metrics.csv")
    sensitivity_df = _read_csv(args.report_dir / "budget_sensitivity_table.csv")
    seed_df = _read_csv(args.report_dir / "benchmark_seed_metrics.csv")
    robustness_cluster_df = _read_csv(args.report_dir / "robustness_cluster_table.csv")
    robustness_window_df = _read_csv(args.report_dir / "robustness_window_table.csv")
    dataset_summary = summarize_dataset(trace_dir=args.trace_dir, summary_dir=args.summary_dir)
    interactivity_metrics = _read_interactivity_metrics()
    task_performance_metrics = _read_task_performance_metrics()

    brief = build_course_brief(
        benchmark_df=benchmark_df,
        ablation_df=ablation_df,
        baseline_df=baseline_df,
        robustness_cluster_df=robustness_cluster_df,
        robustness_window_df=robustness_window_df,
        dataset_summary=dataset_summary,
        interactivity_metrics=interactivity_metrics,
        task_performance_metrics=task_performance_metrics,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(brief, encoding="utf-8")
    evidence_report = build_course_evidence_report(
        benchmark_df=benchmark_df,
        ablation_df=ablation_df,
        baseline_df=baseline_df,
        detection_df=detection_df,
        sensitivity_df=sensitivity_df,
        seed_df=seed_df,
        robustness_cluster_df=robustness_cluster_df,
        robustness_window_df=robustness_window_df,
        dataset_summary=dataset_summary,
        interactivity_metrics=interactivity_metrics,
        task_performance_metrics=task_performance_metrics,
        submission_dir=args.submission_dir,
        out_path=args.evidence_out,
    )
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(evidence_report, encoding="utf-8")
    contents = build_submission_package_contents(
        dataset_summary=dataset_summary,
        benchmark_df=benchmark_df,
        baseline_df=baseline_df,
        robustness_cluster_df=robustness_cluster_df,
        interactivity_metrics=interactivity_metrics,
        task_performance_metrics=task_performance_metrics,
        out_path=args.out,
        submission_dir=args.submission_dir,
    )
    written_paths = write_submission_package(args.submission_dir, contents)
    print("Saved CS526 brief to: {0}".format(args.out))
    print("Saved course evidence report to: {0}".format(args.evidence_out))
    print("Saved CS526 submission package to: {0}".format(args.submission_dir))
    for path in written_paths:
        print("  - {0}".format(path))


if __name__ == "__main__":
    main()
