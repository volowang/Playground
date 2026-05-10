from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from rlva.src.config import (
    COURSE_TASK_PERFORMANCE_METRICS_PATH,
    COURSE_TASK_PERFORMANCE_REPORT_PATH,
    USER_TASK_LOG_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed user-task logs for the CS526 final project")
    parser.add_argument("--log-path", type=Path, default=USER_TASK_LOG_PATH)
    parser.add_argument("--out", type=Path, default=COURSE_TASK_PERFORMANCE_METRICS_PATH)
    parser.add_argument("--markdown-out", type=Path, default=COURSE_TASK_PERFORMANCE_REPORT_PATH)
    return parser.parse_args()


def _read_log_rows(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.median())


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.mean())


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = frame[column].astype(bool)
    if values.empty:
        return 0.0
    return float(values.mean())


def summarize_task_performance(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "completed_tasks": 0,
            "median_duration_seconds": 0.0,
            "median_focus_changes": 0.0,
            "median_filter_changes": 0.0,
            "median_comparison_changes": 0.0,
            "median_exports_triggered": 0.0,
            "average_user_confidence": 0.0,
            "support_signal_rate": 0.0,
            "export_completion_rate": 0.0,
            "notebook_usage_rate": 0.0,
            "tasks_by_type": {},
            "roles": {},
            "systems": {},
        }

    return {
        "completed_tasks": int(len(frame)),
        "median_duration_seconds": _median(frame, "duration_seconds"),
        "median_focus_changes": _median(frame, "focus_changes"),
        "median_filter_changes": _median(frame, "filter_changes"),
        "median_comparison_changes": _median(frame, "comparison_changes"),
        "median_exports_triggered": _median(frame, "exports_triggered"),
        "average_user_confidence": _mean(frame, "user_confidence"),
        "support_signal_rate": _rate(frame, "support_signal"),
        "export_completion_rate": _rate(frame, "export_used"),
        "notebook_usage_rate": _rate(frame, "notebook_used"),
        "tasks_by_type": {
            str(name): int(count) for name, count in frame["task"].value_counts(dropna=False).to_dict().items()
        }
        if "task" in frame.columns
        else {},
        "roles": {
            str(name): int(count) for name, count in frame["role"].value_counts(dropna=False).to_dict().items()
        }
        if "role" in frame.columns
        else {},
        "systems": {
            str(name): int(count) for name, count in frame["system"].value_counts(dropna=False).to_dict().items()
        }
        if "system" in frame.columns
        else {},
    }


def build_task_performance_report(frame: pd.DataFrame, summary: Dict[str, Any], log_path: Path) -> str:
    lines = [
        "# CS526 User Task Performance Report",
        "",
        "## Summary",
        "",
        "- Source log: `{0}`.".format(str(log_path)),
        "- Completed tasks: {0}.".format(_safe_int(summary.get("completed_tasks", 0))),
        "- Median completion time: {0:.1f} seconds.".format(_safe_float(summary.get("median_duration_seconds", 0.0))),
        "- Median focus changes: {0:.1f}.".format(_safe_float(summary.get("median_focus_changes", 0.0))),
        "- Median filter changes: {0:.1f}.".format(_safe_float(summary.get("median_filter_changes", 0.0))),
        "- Median comparison changes: {0:.1f}.".format(_safe_float(summary.get("median_comparison_changes", 0.0))),
        "- Average operator confidence: {0:.2f}/5.".format(_safe_float(summary.get("average_user_confidence", 0.0))),
        "- Support-signal rate: {0:.1%}.".format(_safe_float(summary.get("support_signal_rate", 0.0))),
        "- Export usage rate: {0:.1%}.".format(_safe_float(summary.get("export_completion_rate", 0.0))),
        "- Notebook usage rate: {0:.1%}.".format(_safe_float(summary.get("notebook_usage_rate", 0.0))),
        "",
        "## Interpretation",
        "",
        "These metrics quantify whether a traffic-operations user can move from suspicious windows to a documented review decision, not only whether the charts render quickly.",
        "",
        "## Completed Tasks By Type",
        "",
    ]

    tasks_by_type = summary.get("tasks_by_type", {}) if isinstance(summary, dict) else {}
    if tasks_by_type:
        for task_name, count in tasks_by_type.items():
            lines.append("- `{0}`: {1}".format(task_name, _safe_int(count)))
    else:
        lines.append("- No completed task logs were found.")

    lines.extend(["", "## Recent Task Records", ""])
    if frame.empty:
        lines.append("No task-completion entries are available yet.")
    else:
        preview_columns = [
            "completed_at",
            "workflow",
            "role",
            "task",
            "system",
            "controller",
            "duration_seconds",
            "focus_changes",
            "filter_changes",
            "exports_triggered",
            "user_confidence",
            "support_signal",
        ]
        preview = frame[[col for col in preview_columns if col in frame.columns]].head(8)
        for _, row in preview.iterrows():
            lines.append(
                "- `{0}` | {1} | {2} | {3} | {4:.1f}s | confidence {5:.1f}/5 | support `{6}`".format(
                    str(row.get("completed_at", "")),
                    str(row.get("workflow", "")),
                    str(row.get("system", "")),
                    str(row.get("controller", "")),
                    _safe_float(row.get("duration_seconds", 0.0)),
                    _safe_float(row.get("user_confidence", 0.0)),
                    str(bool(row.get("support_signal", False))).lower(),
                )
            )
    lines.append("")
    return "\n".join(lines)


def measure_course_task_performance(log_path: Path) -> Dict[str, Any]:
    rows = _read_log_rows(log_path)
    frame = pd.DataFrame(rows)
    summary = summarize_task_performance(frame)
    return {
        "log_path": str(log_path),
        "rows": rows,
        "summary": summary,
    }


def write_task_performance_outputs(metrics: Dict[str, Any], out_path: Path, markdown_out: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    frame = pd.DataFrame(metrics.get("rows", []))
    report = build_task_performance_report(frame, metrics.get("summary", {}), log_path=Path(str(metrics.get("log_path", USER_TASK_LOG_PATH))))
    markdown_out.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics = measure_course_task_performance(log_path=args.log_path)
    write_task_performance_outputs(metrics=metrics, out_path=args.out, markdown_out=args.markdown_out)
    print("Saved task performance metrics to: {0}".format(args.out))
    print("Saved task performance report to: {0}".format(args.markdown_out))


if __name__ == "__main__":
    main()
