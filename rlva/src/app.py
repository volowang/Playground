from __future__ import annotations

from datetime import datetime
import inspect
from pathlib import Path
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rlva.src.compare import compare_summary_clusters, load_summary
from rlva.src.config import (
    BENCHMARK_POLICIES,
    BENCHMARK_REPORT_DIR,
    COURSE_DATASET_METADATA_PATH,
    COURSE_DATASET_PATH,
    COURSE_INTERACTIVITY_METRICS_PATH,
    COURSE_TASK_PERFORMANCE_METRICS_PATH,
    COURSE_SUBMISSION_DIR,
    USER_CASE_NOTEBOOK_PATH,
    USER_TASK_LOG_PATH,
    ENVIRONMENTS,
    ENV_LABELS,
    resolve_course_dataset_path,
)
from rlva.src.explain import build_cluster_explain_input, build_window_explain_input, explain_stub
from rlva.src.viz import (
    build_ablation_heatmap,
    build_behavior_space_figure,
    build_benchmark_overview_figure,
    build_temporal_figure,
)


def _extract_points(selection_event: Any) -> List[Dict[str, Any]]:
    if selection_event is None:
        return []
    if isinstance(selection_event, dict):
        return selection_event.get("selection", {}).get("points", [])
    selection_obj = getattr(selection_event, "selection", None)
    if selection_obj is None:
        return []
    if isinstance(selection_obj, dict):
        return selection_obj.get("points", [])
    points_obj = getattr(selection_obj, "points", None)
    return points_obj if isinstance(points_obj, list) else []


def _selected_window(selection_event: Any) -> Optional[int]:
    points = _extract_points(selection_event)
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(custom, (list, tuple)) and custom:
        try:
            return int(custom[0])
        except (TypeError, ValueError):
            return None
    point_index = point.get("point_index") if isinstance(point, dict) else None
    if point_index is None and isinstance(point, dict):
        point_index = point.get("pointNumber")
    if point_index is not None:
        try:
            return int(point_index)
        except (TypeError, ValueError):
            return None
    x_value = point.get("x") if isinstance(point, dict) else None
    try:
        return int(x_value)
    except (TypeError, ValueError):
        return None


def _selected_cluster(selection_event: Any) -> Optional[int]:
    points = _extract_points(selection_event)
    if not points:
        return None
    point = points[0]
    x_value = point.get("x") if isinstance(point, dict) else None
    try:
        return int(x_value)
    except (TypeError, ValueError):
        return None


def _format_policy_name(policy: str) -> str:
    display_names = {
        "pg": "Policy Gradient",
        "ppo": "PPO",
        "dqn": "DQN",
        "random": "Random Baseline",
        "heuristic": "Rule Baseline",
    }
    return display_names.get(policy, policy.replace("_", " ").title())


def _format_feature_group(name: str) -> str:
    return name.replace("_", " ").title()


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return "{0:.2f} {1}".format(value, unit)


def _streamlit_supports_plot_selection() -> bool:
    try:
        return "on_select" in inspect.signature(st.plotly_chart).parameters
    except (TypeError, ValueError):
        return False


def _plotly_chart_with_optional_selection(
    figure: go.Figure,
    *,
    key: Optional[str] = None,
    use_container_width: bool = True,
) -> Any:
    kwargs: Dict[str, Any] = {"use_container_width": use_container_width}
    if key is not None:
        kwargs["key"] = key
    if _streamlit_supports_plot_selection():
        return st.plotly_chart(
            figure,
            on_select="rerun",
            selection_mode="points",
            **kwargs,
        )
    st.plotly_chart(figure, **kwargs)
    return None


METRIC_LABELS = {
    "anomaly_auc": "Anomaly AUC",
    "shift_auc": "Shift AUC",
    "reward_mean": "Reward Mean",
    "reward_jump_auc": "Reward-Jump AUC",
    "shift_localization_error": "Shift Localization Error",
}

LOWER_IS_BETTER_METRICS = {"shift_localization_error"}

PRIMARY_ANALYST_ROLE = "Traffic operations analyst"
PRIMARY_SUPERVISOR_ROLE = "Traffic operations supervisor"
PRIMARY_OPERATIONS_OWNER = "traffic incident review team"
PRIMARY_SUPERVISOR_OWNER = "signal strategy review board"
PRIMARY_PRODUCT_NAME = "Traffic Operations Review Studio"
PRIMARY_PRODUCT_ENGINE = "RLVA"

CASE_STUDY_NOTES: Dict[Tuple[str, str, int], Dict[str, str]] = {
    (
        "inventory",
        "dqn",
        11,
    ): {
        "headline": "Inventory / DQN shows a high-value operational anomaly case.",
        "why": "This exported case is useful because inventory runs produce strong anomaly signal while still separating learned and baseline-style behaviors under demand perturbations.",
    },
    (
        "traffic",
        "dqn",
        23,
    ): {
        "headline": "Traffic / DQN is the clearest regime-shift example in the benchmark.",
        "why": "Traffic interventions create visible phase changes, so this case is a strong qualitative companion to the high shift AUC results in the benchmark tables.",
    },
    (
        "lunarlander",
        "dqn",
        11,
    ): {
        "headline": "LunarLander / DQN is the strongest standard-control environment story.",
        "why": "This case demonstrates that the method is not restricted to the custom operational environments and still surfaces meaningful behavior changes on a standard RL task.",
    },
}

OVERVIEW_TOPICS: Dict[str, str] = {
    "Problem": "Traffic operations teams need to detect corridor incidents, confirm when signal behavior changes, and justify whether the active strategy should remain deployed. The project summarizes controller traces into reviewable windows instead of relying on aggregate reward alone.",
    "Data": "The project uses 5 environments (`queue`, `inventory`, `traffic`, `cartpole`, `lunarlander`) and 5 policy families (`pg`, `ppo`, `dqn`, `random`, `heuristic`). The traffic-control workspace is the primary user-facing scenario, while the other environments provide transfer and robustness evidence.",
    "Method": "Each trace is converted into window-level behavior summaries. Those summaries drive incident ranking, shift localization, controller comparison, ablation studies, and robustness checks.",
    "Evaluation": "The benchmark evaluates anomaly AUC, shift AUC, reward-jump AUC, localization error, seed stability, feature ablations, robustness to analysis-budget changes, and now completed user-task performance.",
    "Interactivity": "The dashboard links ranked incidents, behavior-space views, temporal views, controller-comparison charts, case notebooks, and review exports so operators can move from alert to documented action.",
}

ASSET_DESCRIPTIONS: Dict[str, str] = {
    "main_benchmark": "Main benchmark summary used to explain the strongest environment-level results.",
    "baseline_comparison": "Comparison between RLVA and simpler baseline detectors.",
    "seed_stability": "Seed-level spread showing whether the result holds across repeated runs.",
    "budget_sensitivity": "Sensitivity to tighter analysis budgets and shorter traces.",
}

TASK_FOCUS_COPY: Dict[str, str] = {
    "Anomaly detection": "Use ranked incident windows, operator summaries, and exported cases to understand where the traffic-review workflow most clearly surfaces abnormal controller behavior.",
    "Shift localization": "Use change-oriented plots, localization summaries, and robustness views to understand how well the system identifies traffic regime boundaries.",
    "Method comparison": "Use baseline comparisons, seed stability, and controller comparisons to understand whether the traffic-review workflow adds value beyond simpler reward-only detectors.",
    "Robustness": "Use sensitivity and robustness sections to test whether the main traffic-review conclusions hold under different analysis settings.",
    "Case studies": "Use exported figures and source artifacts to connect benchmark scores back to concrete traffic operations examples.",
}

DEMO_TASK_COPY: Dict[str, str] = {
    "Find anomalies": "Locate suspicious traffic-control windows that deserve incident review before they escalate into corridor-wide problems.",
    "Locate shift": "Pinpoint when traffic conditions or controller behavior changed and whether the review system finds the boundary quickly.",
    "Compare policies": "Contrast alternative traffic-signal strategies behaviorally, not just by reward, using cluster-level divergence.",
    "Tell the project story": "Follow a guided final-presentation path that moves from traffic-operations problem to evidence to operator action.",
}

DEMO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Traffic incident": {"env": "traffic", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
    "Traffic shift": {"env": "traffic", "policy": "dqn", "seed": 23, "task": "Locate shift", "metric": "regime_shift_score"},
    "Traffic strategy comparison": {"env": "traffic", "policy": "dqn", "seed": 23, "task": "Compare policies", "metric": "anomaly_score"},
}

USER_ROLE_COPY: Dict[str, str] = {
    PRIMARY_ANALYST_ROLE: "Use this mode to triage suspicious corridor behavior, confirm whether the signal system changed state, and collect evidence for an incident handoff.",
    PRIMARY_SUPERVISOR_ROLE: "Use this mode to review corridor strategy performance, compare candidate controllers, and decide whether the active signal plan should stay deployed.",
}

ROLE_DEFAULTS: Dict[str, Dict[str, str]] = {
    PRIMARY_ANALYST_ROLE: {"task": "Find anomalies", "preset": "Traffic incident"},
    PRIMARY_SUPERVISOR_ROLE: {"task": "Compare policies", "preset": "Traffic strategy comparison"},
}

USER_WORKFLOW_COPY: Dict[str, Dict[str, str]] = {
    "Triage corridor incident": {
        "role": PRIMARY_ANALYST_ROLE,
        "preset": "Traffic incident",
        "summary": "Use this workflow when a corridor starts behaving abnormally and the operator needs to isolate the suspicious time segments quickly.",
    },
    "Confirm corridor shift": {
        "role": PRIMARY_ANALYST_ROLE,
        "preset": "Traffic shift",
        "summary": "Use this workflow when traffic conditions or signal timing appear to change and the operator needs to confirm the boundary before escalating.",
    },
    "Compare controller strategies": {
        "role": PRIMARY_SUPERVISOR_ROLE,
        "preset": "Traffic strategy comparison",
        "summary": "Use this workflow when a supervisor needs to compare the active controller against an alternative strategy before approving a deployment decision.",
    },
}

TASK_QUESTION_BANK: Dict[str, List[str]] = {
    "Find anomalies": [
        "Which time segments are unusual enough to require review?",
        "Is the suspicious behavior isolated or persistent across nearby windows?",
        "Should the user escalate this controller behavior now or continue monitoring?",
    ],
    "Locate shift": [
        "Where does the operating regime appear to change?",
        "Does the focused boundary line up with the strongest evidence in nearby windows?",
        "Does the user have enough evidence to hand the case to incident review?",
    ],
    "Compare policies": [
        "Which controller behaves most differently in the current operating context?",
        "Is the difference visible only in reward, or also in behavior structure?",
        "Which controller should the user trust more for this scenario?",
    ],
}

ROLE_SCENARIO_COPY: Dict[str, str] = {
    PRIMARY_ANALYST_ROLE: "Live traffic-operations triage: investigate a suspicious corridor interval and prepare a handoff for the incident team.",
    PRIMARY_SUPERVISOR_ROLE: "Traffic-strategy review: decide whether the active signal controller should remain deployed or whether an alternate plan is safer.",
}

TASK_SUCCESS_COPY: Dict[str, str] = {
    "Find anomalies": "Success means the operator can identify the highest-risk corridor window, inspect the linked evidence, and record a concrete incident-review action.",
    "Locate shift": "Success means the operator can point to the most likely change boundary, verify it with neighboring evidence, and escalate the corridor event if needed.",
    "Compare policies": "Success means the supervisor can explain which traffic controller is behaviorally safer or more stable using linked evidence instead of reward alone.",
}

COURSE_PACKAGE_DESCRIPTIONS: Dict[str, str] = {
    "cs526_submission_readme.md": "Index of the course-facing deliverables and demo command.",
    "cs526_final_report_outline.md": "Rubric-aligned report structure for the written final submission.",
    "cs526_presentation_storyboard.md": "Slide-by-slide walkthrough for the live demo or gala presentation.",
    "cs526_video_script.md": "Narration draft for the required 3+ minute project video.",
    "cs526_submission_checklist.md": "Final handoff checklist covering code, data, metrics, report, and media.",
}

USER_CASE_STATUS_OPTIONS = [
    "Needs review",
    "Escalated",
    "Monitoring",
    "Approved",
    "Resolved",
]

USER_NOTE_TAG_OPTIONS = [
    "Suspicious corridor behavior",
    "Traffic regime change",
    "Strategy comparison evidence",
    "Follow-up required",
    "Exported to review packet",
]


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(208, 230, 244, 0.45), transparent 28%),
                radial-gradient(circle at top right, rgba(243, 211, 180, 0.40), transparent 24%),
                linear-gradient(180deg, #f7f3ed 0%, #f6f8fb 52%, #eef3f8 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }
        .hero-shell {
            padding: 1.35rem 1.5rem 1.15rem 1.5rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(20, 54, 86, 0.96), rgba(49, 87, 122, 0.92));
            color: #f7fbff;
            box-shadow: 0 24px 60px rgba(18, 34, 51, 0.18);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            font-size: 0.75rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            opacity: 0.74;
            margin-bottom: 0.45rem;
        }
        .hero-title {
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 700;
            margin: 0 0 0.55rem 0;
        }
        .hero-copy {
            max-width: 62rem;
            font-size: 1rem;
            line-height: 1.55;
            opacity: 0.92;
            margin: 0;
        }
        .section-chip {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: rgba(53, 92, 125, 0.10);
            color: #27445d;
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(39, 68, 93, 0.08);
            border-radius: 16px;
            padding: 0.85rem 0.9rem 0.75rem 0.9rem;
            box-shadow: 0 14px 34px rgba(39, 68, 93, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }
        div[data-testid="stTabs"] button {
            border-radius: 999px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .asset-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(39, 68, 93, 0.10);
            border-radius: 16px;
            padding: 0.55rem 0.55rem 0.35rem 0.55rem;
            box-shadow: 0 10px 24px rgba(39, 68, 93, 0.05);
        }
        .asset-card-active {
            border: 2px solid rgba(184, 74, 57, 0.55);
            box-shadow: 0 14px 28px rgba(184, 74, 57, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


@st.cache_data(show_spinner=False)
def cached_available_seeds(env_name: str, policy: str) -> List[int]:
    benchmark_dir = BENCHMARK_REPORT_DIR.parent / "summaries" / env_name / policy
    seeds: List[int] = []
    for path in sorted(benchmark_dir.glob("seed_*.csv")):
        try:
            seeds.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return seeds


@st.cache_data(show_spinner=False)
def cached_summary(env_name: str, policy: str, seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        path = BENCHMARK_REPORT_DIR.parent / "summaries" / env_name / policy / "seed_{0}.csv".format(seed)
        if path.exists():
            return pd.read_csv(path)
    return load_summary(env_name=env_name, policy=policy)


@st.cache_data(show_spinner=False)
def cached_compare(env_name: str, p1: str, p2: str, k_clusters: int, seed: Optional[int] = None) -> pd.DataFrame:
    return compare_summary_clusters(cached_summary(env_name, p1, seed=seed), cached_summary(env_name, p2, seed=seed), n_clusters=k_clusters)


@st.cache_data(show_spinner=False)
def cached_report_csv(filename: str) -> pd.DataFrame:
    path = BENCHMARK_REPORT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_artifact_coverage() -> pd.DataFrame:
    benchmark_root = BENCHMARK_REPORT_DIR.parent
    model_policies = ["pg", "ppo", "dqn"]
    tracked_envs = ENVIRONMENTS
    rows: List[Dict[str, Any]] = []

    for env_name in tracked_envs:
        model_files = []
        trace_files = []
        summary_files = []
        for policy in model_policies:
            model_files.extend(sorted((benchmark_root / "models" / env_name / policy).glob("*.pt")))
        for policy in BENCHMARK_POLICIES:
            trace_files.extend(sorted((benchmark_root / "traces" / env_name / policy).glob("*.csv")))
            summary_files.extend(sorted((benchmark_root / "summaries" / env_name / policy).glob("*.csv")))

        trace_steps = 0
        for path in trace_files:
            trace_steps += len(pd.read_csv(path))

        summary_windows = 0
        for path in summary_files:
            summary_windows += len(pd.read_csv(path))

        rows.append(
            {
                "env": env_name,
                "model_policies": len(model_policies),
                "trace_policies": len(BENCHMARK_POLICIES),
                "model_files": len(model_files),
                "trace_files": len(trace_files),
                "trace_steps": trace_steps,
                "summary_files": len(summary_files),
                "summary_windows": summary_windows,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_course_dataset_metadata() -> Dict[str, Any]:
    if not COURSE_DATASET_METADATA_PATH.exists():
        return {}
    return json.loads(COURSE_DATASET_METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cached_course_interactivity_metrics() -> Dict[str, Any]:
    if not COURSE_INTERACTIVITY_METRICS_PATH.exists():
        return {}
    return json.loads(COURSE_INTERACTIVITY_METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cached_course_task_performance_metrics() -> Dict[str, Any]:
    if not COURSE_TASK_PERFORMANCE_METRICS_PATH.exists():
        return {}
    return json.loads(COURSE_TASK_PERFORMANCE_METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def cached_submission_package_inventory() -> pd.DataFrame:
    if not COURSE_SUBMISSION_DIR.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(COURSE_SUBMISSION_DIR.glob("*.md")):
        rows.append(
            {
                "artifact": path.name,
                "purpose": COURSE_PACKAGE_DESCRIPTIONS.get(path.name, "Course-facing markdown artifact."),
                "size": _format_bytes(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)


def _load_case_notebook_rows() -> List[Dict[str, Any]]:
    if not USER_CASE_NOTEBOOK_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in USER_CASE_NOTEBOOK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _save_case_notebook_entry(entry: Dict[str, Any]) -> None:
    USER_CASE_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_CASE_NOTEBOOK_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _case_notebook_frame() -> pd.DataFrame:
    frame = pd.DataFrame(_load_case_notebook_rows())
    if frame.empty:
        return frame
    sort_cols = [col for col in ["created_at", "title"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    return frame.reset_index(drop=True)


def _load_user_task_rows() -> List[Dict[str, Any]]:
    if not USER_TASK_LOG_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in USER_TASK_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _save_user_task_entry(entry: Dict[str, Any]) -> None:
    USER_TASK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_TASK_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def _user_task_frame() -> pd.DataFrame:
    frame = pd.DataFrame(_load_user_task_rows())
    if frame.empty:
        return frame
    sort_cols = [col for col in ["completed_at", "workflow"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    return frame.reset_index(drop=True)


def _case_note_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "created_at",
        "workflow",
        "role",
        "task",
        "system",
        "controller",
        "run",
        "window_label",
        "status",
        "owner",
        "risk_level",
        "risk_flag",
        "change_confidence",
        "average_outcome",
        "title",
        "note",
        "tags",
        "recommended_action",
    ]
    available = [col for col in columns if col in frame.columns]
    export_df = frame[available].copy()
    if "tags" in export_df.columns:
        export_df["tags"] = export_df["tags"].apply(
            lambda value: ", ".join(value) if isinstance(value, list) else str(value)
        )
    return export_df


@st.cache_data(show_spinner=False)
def cached_course_dataset_sample(nrows: int = 12) -> pd.DataFrame:
    dataset_path = resolve_course_dataset_path()
    if not dataset_path.exists():
        return pd.DataFrame()
    return pd.read_csv(dataset_path, nrows=nrows)


def _course_dataset_inventory_frame() -> pd.DataFrame:
    meta = cached_course_dataset_metadata()
    inventory = meta.get("env_inventory", {}) if meta else {}
    if not inventory:
        return pd.DataFrame()
    rows = []
    total_source_files = int(meta.get("source_trace_files", 0)) if meta else 0
    replica_passes = int(meta.get("replica_passes", 0)) if meta else 0
    total_records = int(meta.get("total_records", 0)) if meta else 0
    num_envs = max(1, len(inventory))
    for env_name, count in sorted(inventory.items()):
        rows.append(
            {
                "env": env_name,
                "env_label": _format_env_name(env_name),
                "source_trace_files": int(count),
                "replica_passes": replica_passes,
                "estimated_records_share": total_records / float(num_envs),
                "estimated_bytes_share": int(meta.get("actual_bytes", 0)) / float(num_envs) if meta else 0.0,
                "source_file_fraction": (float(count) / float(total_source_files)) if total_source_files else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_dataset_inventory_figure(inventory_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        inventory_df,
        x="env_label",
        y="source_trace_files",
        color="replica_passes",
        title="1GB Dataset Provenance by Environment",
        labels={"env_label": "Environment", "source_trace_files": "Source trace files", "replica_passes": "Replica passes"},
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(template="plotly_white", coloraxis_showscale=False)
    return fig


def _build_dataflow_figure() -> go.Figure:
    node_labels = [
        "1GB raw trace corpus",
        "seeded benchmark traces",
        "window summaries",
        "interactive demo views",
        "presentation exports",
    ]
    fig = go.Figure(
        go.Sankey(
            node={"label": node_labels, "pad": 18, "thickness": 18, "color": ["#143656", "#355c7d", "#6c8ebf", "#b84a39", "#f0a202"]},
            link={
                "source": [0, 1, 2, 2, 3],
                "target": [1, 2, 3, 4, 4],
                "value": [10, 9, 8, 4, 6],
            },
        )
    )
    fig.update_layout(template="plotly_white", title="Project Dataflow: Raw Dataset To Interactive Evidence")
    return fig


def _build_schema_frame(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()
    rows = []
    for column in sample_df.columns:
        values = sample_df[column]
        preview = values.iloc[0]
        rows.append(
            {
                "field": column,
                "dtype": str(values.dtype),
                "example": str(preview),
            }
        )
    return pd.DataFrame(rows)


def _build_window_evidence_markdown(env_name: str, policy: str, task_name: str, window_row: pd.Series) -> str:
    return "\n".join(
        [
            "# RLVA Evidence Snapshot",
            "",
            "- Task: {0}".format(task_name),
            "- Environment: {0}".format(_format_env_name(env_name)),
            "- Policy: {0}".format(_format_policy_name(policy)),
            "- Window: k={0}".format(int(window_row["k"])),
            "- Span: t={0}-{1}".format(int(window_row["t_start"]), int(window_row["t_end"])),
            "- Anomaly score: {0:.6f}".format(float(window_row["anomaly_score"])),
            "- Shift score: {0:.6f}".format(float(window_row["regime_shift_score"])),
            "- Reward: {0:.6f}".format(float(window_row["r_bar"])),
            "- Entropy: {0:.6f}".format(float(window_row["H_bar"])),
            "",
            "## Interpretation",
            "",
            _window_explain(pd.DataFrame([window_row]), policy=policy, env_name=env_name, window_row=window_row),
            "",
        ]
    )


def _build_cluster_evidence_markdown(env_name: str, p1: str, p2: str, cluster_count: int, cluster_row: pd.Series) -> str:
    return "\n".join(
        [
            "# RLVA Policy Comparison Snapshot",
            "",
            "- Environment: {0}".format(_format_env_name(env_name)),
            "- Policy 1: {0}".format(_format_policy_name(p1)),
            "- Policy 2: {0}".format(_format_policy_name(p2)),
            "- Cluster count: {0}".format(cluster_count),
            "- Focused cluster: {0}".format(int(cluster_row["cluster"])),
            "- JS divergence: {0:.6f}".format(float(cluster_row["js_div"])),
            "- Reward gap: {0:.6f}".format(float(cluster_row["reward_gap"])),
            "- Aggregated windows: {0:.0f}".format(float(cluster_row["n_total"])),
            "",
            "## Interpretation",
            "",
            _cluster_explain(cluster_row, env_name=env_name, p1=p1, p2=p2, cluster_count=cluster_count),
            "",
        ]
    )


def _window_explain(summary_df: pd.DataFrame, policy: str, env_name: str, window_row: pd.Series) -> str:
    explain_input = build_window_explain_input(
        policy="{0}:{1}".format(env_name, policy),
        compare_to=None,
        filters={"env": env_name},
        window=window_row.to_dict(),
        neighbors=None,
    )
    return explain_stub(explain_input)


def _cluster_explain(compare_row: pd.Series, env_name: str, p1: str, p2: str, cluster_count: int) -> str:
    explain_input = build_cluster_explain_input(
        policy_1="{0}:{1}".format(env_name, p1),
        policy_2="{0}:{1}".format(env_name, p2),
        filters={"env": env_name, "cluster_count": cluster_count},
        cluster_row=compare_row.to_dict(),
    )
    return explain_stub(explain_input)


def _summary_metrics(summary_df: pd.DataFrame) -> Dict[str, float]:
    anomaly_rate = float(summary_df["is_anomaly"].mean()) if len(summary_df) else 0.0
    regime_rate = float(summary_df["is_regime_shift"].mean()) if len(summary_df) else 0.0
    return {
        "windows": float(len(summary_df)),
        "reward_mean": float(summary_df["r_bar"].mean()),
        "anomaly_max": float(summary_df["anomaly_score"].max()),
        "regime_rate": regime_rate,
        "anomaly_rate": anomaly_rate,
    }


def _top_window_finding(summary_df: pd.DataFrame) -> pd.Series:
    ranked = summary_df.sort_values(["anomaly_score", "regime_shift_score", "r_bar"], ascending=[False, False, True])
    return ranked.iloc[0]


def _top_cluster_finding(compare_df: pd.DataFrame) -> Optional[pd.Series]:
    valid = compare_df[compare_df["js_div"].notna()].copy()
    if valid.empty:
        return None
    return valid.sort_values(["js_div", "reward_gap", "n_total"], ascending=[False, False, False]).iloc[0]


def _main_story_rows(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    focus_mask = benchmark_df["env"].isin(["inventory", "traffic"]) | (
        benchmark_df["env"].eq("lunarlander") & benchmark_df["policy"].eq("dqn")
    )
    focus_df = benchmark_df[focus_mask].copy()
    env_order = {"inventory": 0, "traffic": 1, "lunarlander": 2}
    policy_order = {name: idx for idx, name in enumerate(BENCHMARK_POLICIES)}
    focus_df["_env_order"] = focus_df["env"].map(env_order).fillna(99)
    focus_df["_policy_order"] = focus_df["policy"].map(policy_order).fillna(99)
    focus_df = focus_df.sort_values(["_env_order", "_policy_order"]).drop(columns=["_env_order", "_policy_order"])
    return focus_df


def _supplement_rows(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    supplement_df = benchmark_df[benchmark_df["env"].eq("cartpole")].copy()
    policy_order = {name: idx for idx, name in enumerate(BENCHMARK_POLICIES)}
    supplement_df["_policy_order"] = supplement_df["policy"].map(policy_order).fillna(99)
    return supplement_df.sort_values("_policy_order").drop(columns=["_policy_order"])


def _best_row(benchmark_df: pd.DataFrame, env_name: str, metric: str) -> Optional[pd.Series]:
    env_df = benchmark_df[benchmark_df["env"] == env_name].copy()
    if env_df.empty:
        return None
    return env_df.sort_values(f"{metric}_mean", ascending=metric in LOWER_IS_BETTER_METRICS).iloc[0]


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _artifact_fig_dir() -> Path:
    return BENCHMARK_REPORT_DIR.parent / "figs"


def _figure_image_path(stem: str) -> Optional[Path]:
    path = _artifact_fig_dir() / "{0}.png".format(stem)
    return path if path.exists() else None


def _case_study_index() -> pd.DataFrame:
    path = _artifact_fig_dir() / "case_study_index.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _build_main_story_figure(main_story_df: pd.DataFrame) -> go.Figure:
    frame = main_story_df.copy()
    frame["label"] = frame.apply(
        lambda row: "{0}<br>{1}".format(ENV_LABELS.get(str(row["env"]), str(row["env"])), _format_policy_name(str(row["policy"]))),
        axis=1,
    )
    long_df = frame.melt(
        id_vars=["label", "env", "policy"],
        value_vars=["anomaly_auc_mean", "shift_auc_mean", "reward_jump_auc_mean"],
        var_name="metric",
        value_name="score",
    )
    metric_map = {
        "anomaly_auc_mean": "Anomaly AUC",
        "shift_auc_mean": "Shift AUC",
        "reward_jump_auc_mean": "Reward-Jump AUC",
    }
    long_df["metric"] = long_df["metric"].map(metric_map)
    fig = px.bar(
        long_df,
        x="label",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=["#b84a39", "#355c7d", "#f0a202"],
        title="Main Narrative Benchmark",
    )
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="score", legend_title="")
    return fig


def _build_transfer_figure(supplement_df: pd.DataFrame) -> go.Figure:
    frame = supplement_df.copy()
    frame["policy_label"] = frame["policy"].map(_format_policy_name)
    fig = px.scatter(
        frame,
        x="anomaly_auc_mean",
        y="shift_auc_mean",
        color="policy_label",
        size="reward_mean_mean",
        hover_name="policy_label",
        title="CartPole Transferability Supplement",
        labels={"anomaly_auc_mean": "Anomaly AUC", "shift_auc_mean": "Shift AUC", "policy_label": "Policy"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def _build_lunarlander_focus_figure(lunarlander_benchmark: pd.DataFrame) -> go.Figure:
    frame = lunarlander_benchmark[lunarlander_benchmark["policy"].isin(["dqn", "pg", "ppo"])].copy()
    frame["policy_label"] = frame["policy"].map(_format_policy_name)
    fig = px.scatter(
        frame,
        x="anomaly_auc_mean",
        y="shift_auc_mean",
        color="policy_label",
        size="reward_jump_auc_mean",
        hover_name="policy_label",
        title="LunarLander Policy Readout",
        labels={
            "anomaly_auc_mean": "Anomaly AUC",
            "shift_auc_mean": "Shift AUC",
            "reward_jump_auc_mean": "Reward-Jump AUC",
        },
    )
    fig.update_layout(template="plotly_white", legend_title="")
    return fig


def _build_ablation_profile_figure(ablation_slice: pd.DataFrame) -> go.Figure:
    frame = ablation_slice.copy()
    frame["feature_group_label"] = frame["feature_group"].map(_format_feature_group)
    long_df = frame.melt(
        id_vars=["feature_group_label"],
        value_vars=["anomaly_auc_mean", "shift_auc_mean"],
        var_name="metric",
        value_name="score",
    )
    long_df["metric"] = long_df["metric"].map({"anomaly_auc_mean": "Anomaly AUC", "shift_auc_mean": "Shift AUC"})
    fig = px.bar(
        long_df,
        x="feature_group_label",
        y="score",
        color="metric",
        barmode="group",
        color_discrete_sequence=["#b84a39", "#355c7d"],
        title="Feature Ablation Profile",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Feature group", yaxis_title="score", legend_title="")
    return fig


def _build_baseline_comparison_figure(
    benchmark_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    env_name: str,
    policy_name: str,
    metric: str,
) -> go.Figure:
    rlva_row = benchmark_df[(benchmark_df["env"] == env_name) & (benchmark_df["policy"] == policy_name)].copy()
    base_rows = baseline_df[(baseline_df["env"] == env_name) & (baseline_df["policy"] == policy_name)].copy()
    rows: List[Dict[str, Any]] = []
    if not rlva_row.empty:
        row = rlva_row.iloc[0]
        rows.append({"method": "RLVA", "score": float(row["{0}_mean".format(metric)]), "family": "RLVA"})
    for _, row in base_rows.iterrows():
        rows.append(
            {
                "method": str(row["method"]).replace("_", " ").title(),
                "score": float(row["{0}_mean".format(metric)]),
                "family": "Simple baselines" if str(row["method"]).startswith("reward_") else "External detectors",
            }
        )
    ascending = metric in LOWER_IS_BETTER_METRICS
    frame = pd.DataFrame(rows).sort_values(["family", "score"], ascending=[True, ascending])
    fig = px.bar(
        frame,
        x="method",
        y="score",
        color="family",
        title="{0} / {1}: {2}".format(ENV_LABELS.get(env_name, env_name), _format_policy_name(policy_name), _metric_label(metric)),
        color_discrete_map={"RLVA": "#b84a39", "Simple baselines": "#f0a202", "External detectors": "#355c7d"},
    )
    yaxis_title = "lower is better" if metric in LOWER_IS_BETTER_METRICS else "score"
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title=yaxis_title, legend_title="")
    return fig


def _build_sensitivity_figure(
    sensitivity_df: pd.DataFrame,
    env_name: str,
    policy_name: str,
    metric: str,
    fixed_l: int,
    fixed_seed_count: int,
) -> go.Figure:
    frame = sensitivity_df[
        (sensitivity_df["env"] == env_name)
        & (sensitivity_df["policy"] == policy_name)
        & (sensitivity_df["L"] == fixed_l)
        & (sensitivity_df["seed_count"] == fixed_seed_count)
    ].copy()
    fig = px.line(
        frame.sort_values("T"),
        x="T",
        y="{0}_mean".format(metric),
        markers=True,
        title="{0} / {1}: {2} vs Trace Horizon".format(ENV_LABELS.get(env_name, env_name), _format_policy_name(policy_name), _metric_label(metric)),
    )
    fig.update_layout(template="plotly_white", xaxis_title="Trace horizon T", yaxis_title="score")
    return fig


def _build_window_sensitivity_figure(
    sensitivity_df: pd.DataFrame,
    env_name: str,
    policy_name: str,
    metric: str,
    fixed_t: int,
    fixed_seed_count: int,
) -> go.Figure:
    frame = sensitivity_df[
        (sensitivity_df["env"] == env_name)
        & (sensitivity_df["policy"] == policy_name)
        & (sensitivity_df["T"] == fixed_t)
        & (sensitivity_df["seed_count"] == fixed_seed_count)
    ].copy()
    fig = px.line(
        frame.sort_values("L"),
        x="L",
        y="{0}_mean".format(metric),
        markers=True,
        title="{0} / {1}: {2} vs Window Size".format(ENV_LABELS.get(env_name, env_name), _format_policy_name(policy_name), _metric_label(metric)),
    )
    fig.update_layout(template="plotly_white", xaxis_title="Window size L", yaxis_title="score")
    return fig


def _build_seed_stability_figure(seed_df: pd.DataFrame, env_name: str, metric: str) -> go.Figure:
    frame = seed_df[seed_df["env"] == env_name].copy()
    frame["policy_label"] = frame["policy"].map(_format_policy_name)
    fig = px.strip(
        frame,
        x="policy_label",
        y=metric,
        color="policy_label",
        hover_data=["seed"],
        title="{0}: {1} Across Seeds".format(ENV_LABELS.get(env_name, env_name), _metric_label(metric)),
    )
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="seed score", legend_title="")
    return fig


def _build_coverage_figure(coverage_df: pd.DataFrame) -> go.Figure:
    frame = coverage_df.copy()
    frame["env_label"] = frame["env"].map(lambda name: ENV_LABELS.get(name, name))
    long_df = frame.melt(
        id_vars=["env_label"],
        value_vars=["model_files", "trace_files", "summary_files"],
        var_name="artifact",
        value_name="count",
    )
    long_df["artifact"] = long_df["artifact"].map(
        {"model_files": "Models", "trace_files": "Trace files", "summary_files": "Summary files"}
    )
    fig = px.bar(
        long_df,
        x="env_label",
        y="count",
        color="artifact",
        barmode="group",
        title="Benchmark Artifact Inventory",
        color_discrete_sequence=["#355c7d", "#6c8ebf", "#b84a39"],
    )
    fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="artifact count", legend_title="")
    return fig


def _build_window_robustness_figure(window_df: pd.DataFrame, env_name: str, metric: str) -> go.Figure:
    frame = window_df[window_df["env"] == env_name].copy()
    frame["policy_label"] = frame["policy"].map(_format_policy_name)
    fig = px.line(
        frame.sort_values("window_length"),
        x="window_length",
        y="{0}_mean".format(metric),
        color="policy_label",
        markers=True,
        title="{0}: {1} vs Window Length".format(ENV_LABELS.get(env_name, env_name), _metric_label(metric)),
    )
    yaxis_title = "lower is better" if metric in LOWER_IS_BETTER_METRICS else "score"
    fig.update_layout(template="plotly_white", xaxis_title="Window length", yaxis_title=yaxis_title, legend_title="")
    return fig


def _build_cluster_robustness_figure(cluster_df: pd.DataFrame, env_name: str) -> go.Figure:
    frame = cluster_df[cluster_df["env"] == env_name].copy()
    frame["pair"] = frame["policy_1"].map(_format_policy_name) + " vs " + frame["policy_2"].map(_format_policy_name)
    fig = px.line(
        frame.sort_values("cluster_count"),
        x="cluster_count",
        y="top1_js_div_mean",
        color="pair",
        markers=True,
        title="{0}: Top Cluster Divergence vs Cluster Count".format(ENV_LABELS.get(env_name, env_name)),
    )
    fig.update_layout(template="plotly_white", xaxis_title="Cluster count", yaxis_title="Top-1 JS divergence", legend_title="")
    return fig


@st.cache_data(show_spinner=False)
def cached_interaction_timings() -> pd.DataFrame:
    benchmark_df = cached_report_csv("benchmark_table.csv")
    baseline_df = cached_report_csv("baseline_detector_table.csv")

    def _measure(label: str, fn: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {"interaction": label, "latency_ms": elapsed_ms}

    summary_df = cached_summary("inventory", "dqn", seed=11)
    timings = [
        _measure("Load seeded summary", lambda: cached_summary("inventory", "dqn", seed=11)),
        _measure("Build behavior space", lambda: build_behavior_space_figure(summary_df, env_name="inventory", color_by="anomaly_score")),
        _measure("Build temporal trends", lambda: build_temporal_figure(summary_df, selected_k=int(summary_df["k"].iloc[0]))),
        _measure("Policy comparison", lambda: cached_compare("inventory", "dqn", "ppo", 8, seed=11)),
        _measure("Benchmark overview", lambda: build_benchmark_overview_figure(benchmark_df, metric="anomaly_auc")),
        _measure(
            "Baseline comparison",
            lambda: _build_baseline_comparison_figure(
                benchmark_df=benchmark_df,
                baseline_df=baseline_df,
                env_name="inventory",
                policy_name="pg",
                metric="shift_auc",
            ),
        ),
    ]
    return pd.DataFrame(timings)


def _case_study_note(env_name: str, policy: str, seed: int) -> Dict[str, str]:
    return CASE_STUDY_NOTES.get(
        (env_name, policy, seed),
        {
            "headline": "Selected exported case",
            "why": "This figure is included as qualitative evidence linking benchmark scores back to a concrete trace and summary artifact.",
        },
    )


def _metric_ascending(metric: str) -> bool:
    return metric in LOWER_IS_BETTER_METRICS


def _format_env_name(env_name: str) -> str:
    return ENV_LABELS.get(env_name, env_name)


def _filter_like_global(frame: pd.DataFrame, controls: Dict[str, Any], env_col: str = "env", policy_col: str = "policy") -> pd.DataFrame:
    out = frame.copy()
    if env_col in out.columns and controls["env"] != "all":
        out = out[out[env_col] == controls["env"]]
    if policy_col in out.columns and controls["policy"] != "all":
        out = out[out[policy_col] == controls["policy"]]
    return out


def _global_controls(benchmark_df: pd.DataFrame) -> Dict[str, Any]:
    st.sidebar.header("Global Explorer")
    if "pending_global_env" in st.session_state:
        st.session_state["ctl_env"] = st.session_state.pop("pending_global_env")
    if "pending_global_policy" in st.session_state:
        st.session_state["ctl_policy"] = st.session_state.pop("pending_global_policy")
    if "pending_global_metric" in st.session_state:
        st.session_state["ctl_metric"] = st.session_state.pop("pending_global_metric")
    mode = st.sidebar.radio("Interaction mode", options=["Presentation mode", "Exploration mode"], index=1)
    task_focus = st.sidebar.selectbox(
        "What do you want to inspect?",
        options=list(TASK_FOCUS_COPY.keys()),
    )
    env_options = ["all"] + sorted(benchmark_df["env"].unique().tolist())
    policy_options = ["all"] + sorted(benchmark_df["policy"].unique().tolist())
    metric_options = ["anomaly_auc", "shift_auc", "reward_mean", "reward_jump_auc", "shift_localization_error"]
    env_name = st.sidebar.selectbox("Global environment", options=env_options, key="ctl_env", format_func=lambda value: "All environments" if value == "all" else _format_env_name(value))
    policy = st.sidebar.selectbox("Global policy", options=policy_options, key="ctl_policy", format_func=lambda value: "All policies" if value == "all" else _format_policy_name(value))
    metric = st.sidebar.selectbox("Global metric", options=metric_options, key="ctl_metric", format_func=_metric_label)
    top_k = st.sidebar.slider("Top rows to show", min_value=3, max_value=25, value=10)
    return {
        "mode": mode,
        "task_focus": task_focus,
        "env": env_name,
        "policy": policy,
        "metric": metric,
        "top_k": top_k,
    }


def _task_focus_hint(task_focus: str, mode: str) -> str:
    suffix = "Use the interactive controls below to drill down." if mode == "Exploration mode" else "Use the highlighted sections as the suggested presentation path."
    return "{0} {1}".format(TASK_FOCUS_COPY[task_focus], suffix)


def _control_default(options: List[str], preferred: str) -> int:
    return options.index(preferred) if preferred in options else 0


def _queue_global_update(env_name: Optional[str] = None, policy: Optional[str] = None, metric: Optional[str] = None) -> None:
    if env_name is not None:
        st.session_state["pending_global_env"] = env_name
    if policy is not None:
        st.session_state["pending_global_policy"] = policy
    if metric is not None:
        st.session_state["pending_global_metric"] = metric


def _asset_category_from_stem(stem: str) -> str:
    return "case_study" if stem.endswith("_case_study") else "paper_figure"


def _queue_asset_focus(stem: str) -> None:
    category = _asset_category_from_stem(stem)
    st.session_state["pending_asset_category"] = category
    st.session_state["selected_asset_{0}".format(category)] = stem


def _benchmark_table_frame(benchmark_df: pd.DataFrame, controls: Dict[str, Any]) -> pd.DataFrame:
    frame = _filter_like_global(benchmark_df, controls)
    metric = controls["metric"]
    sort_col = "{0}_mean".format(metric)
    frame = frame.sort_values(sort_col, ascending=_metric_ascending(metric)).copy()
    frame["environment"] = frame["env"].map(_format_env_name)
    frame["policy_label"] = frame["policy"].map(_format_policy_name)
    fixed_cols = ["environment", "policy_label", "num_seeds"]
    metric_cols = [sort_col, "anomaly_auc_mean", "shift_auc_mean", "reward_jump_auc_mean", "shift_localization_error_mean"]
    ordered_metric_cols: List[str] = []
    for col in metric_cols:
        if col not in ordered_metric_cols:
            ordered_metric_cols.append(col)
    display_cols = fixed_cols + ordered_metric_cols
    renamed = frame[display_cols].rename(columns={"num_seeds": "seeds"})
    renamed = renamed.rename(columns={sort_col: "selected_metric"})
    rename_map = {
        "anomaly_auc_mean": "anomaly_auc",
        "shift_auc_mean": "shift_auc",
        "reward_jump_auc_mean": "reward_jump_auc",
        "shift_localization_error_mean": "shift_localization_error",
    }
    final_columns: List[str] = []
    used_names: Dict[str, int] = {}
    for col in renamed.columns:
        target = rename_map.get(col, col)
        if col == "selected_metric":
            target = "selected_metric"
        if target in used_names:
            used_names[target] += 1
            target = "{0}_{1}".format(target, used_names[target])
        else:
            used_names[target] = 0
        final_columns.append(target)
    renamed.columns = final_columns
    return renamed.head(int(controls["top_k"]))


def _baseline_comparison_frame(benchmark_df: pd.DataFrame, baseline_df: pd.DataFrame, env_name: str, policy_name: str, metric: str, normalize: bool) -> pd.DataFrame:
    rlva_row = benchmark_df[(benchmark_df["env"] == env_name) & (benchmark_df["policy"] == policy_name)]
    if rlva_row.empty:
        return pd.DataFrame()
    rlva_score = float(rlva_row.iloc[0]["{0}_mean".format(metric)])
    rows = [{"method": "RLVA", "score": rlva_score, "delta_vs_rlva": 0.0, "beats_rlva": False}]
    base_rows = baseline_df[(baseline_df["env"] == env_name) & (baseline_df["policy"] == policy_name)].copy()
    for _, row in base_rows.iterrows():
        score = float(row["{0}_mean".format(metric)])
        delta = score - rlva_score
        if metric in LOWER_IS_BETTER_METRICS:
            beats = score < rlva_score
        else:
            beats = score > rlva_score
        rows.append(
            {
                "method": str(row["method"]).replace("_", " ").title(),
                "score": score,
                "delta_vs_rlva": delta,
                "beats_rlva": beats,
            }
        )
    frame = pd.DataFrame(rows)
    if normalize and rlva_score != 0:
        frame["display_score"] = frame["score"] / rlva_score
    else:
        frame["display_score"] = frame["score"]
    return frame


def _render_related_assets(asset_catalog: pd.DataFrame, env_name: str, policy: str, task_focus: str, key_prefix: str) -> None:
    matches = asset_catalog.copy()
    if env_name != "all":
        matches = matches[(matches["env"].isna()) | (matches["env"] == env_name)]
    if policy != "all":
        matches = matches[(matches["policy"].isna()) | (matches["policy"] == policy)]
    if task_focus == "Case studies":
        matches = matches[matches["category"] == "case_study"]
    elif task_focus in {"Anomaly detection", "Shift localization", "Method comparison", "Robustness"}:
        matches = matches
    if matches.empty:
        st.caption("No directly related exported assets match the current global filters.")
        return
    st.caption("Related exported assets:")
    cols = st.columns(min(4, len(matches.head(4))))
    for idx, (_, row) in enumerate(matches.head(4).iterrows()):
        with cols[idx]:
            st.markdown("`{0}`".format(str(row["stem"])))
            if st.button("Open in Assets", key="related_asset_{0}_{1}_{2}".format(key_prefix, task_focus, row["stem"]), type="secondary"):
                _queue_asset_focus(str(row["stem"]))
                _trigger_rerun()
    st.caption("These shortcuts preload the chosen asset in the Assets tab.")


def _build_timing_figure(timing_df: pd.DataFrame) -> go.Figure:
    frame = timing_df.copy().sort_values("latency_ms", ascending=False)
    fig = px.bar(
        frame,
        x="latency_ms",
        y="interaction",
        orientation="h",
        color="latency_ms",
        color_continuous_scale="Turbo",
        title="Interaction Latency Profile",
    )
    fig.update_layout(template="plotly_white", xaxis_title="Latency (ms)", yaxis_title="", coloraxis_showscale=False)
    return fig


def _build_ranked_bar(frame: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, orientation: str = "v") -> go.Figure:
    fig = px.bar(frame, x=x, y=y, color=color, orientation=orientation, title=title)
    fig.update_layout(template="plotly_white", legend_title="")
    return fig


def _render_metric_cards_from_frame(frame: pd.DataFrame, label_col: str, value_col: str, max_items: int = 3, formatter: str = "{0:.3f}") -> None:
    rows = frame.head(max_items).reset_index(drop=True)
    cols = st.columns(max(1, len(rows)))
    for idx, (_, row) in enumerate(rows.iterrows()):
        cols[idx].metric(str(row[label_col]), formatter.format(float(row[value_col])))


@st.cache_data(show_spinner=False)
def cached_asset_catalog() -> pd.DataFrame:
    fig_dir = _artifact_fig_dir()
    rows: List[Dict[str, Any]] = []
    for path in sorted(fig_dir.glob("*.png")):
        stem = path.stem
        category = "case_study" if stem.endswith("_case_study") else "paper_figure"
        env_name = None
        policy = None
        if category == "case_study":
            parts = stem.split("_")
            if len(parts) >= 3:
                env_name = parts[0]
                policy = parts[1]
        rows.append(
            {
                "stem": stem,
                "category": category,
                "env": env_name,
                "policy": policy,
                "png_path": str(path),
                "pdf_path": str(fig_dir / "{0}.pdf".format(stem)),
                "description": ASSET_DESCRIPTIONS.get(stem, ""),
            }
        )
    return pd.DataFrame(rows)


def _asset_label(row: pd.Series) -> str:
    env_part = _format_env_name(str(row["env"])) if pd.notna(row["env"]) and str(row["env"]) != "None" else "Global"
    policy_part = _format_policy_name(str(row["policy"])) if pd.notna(row["policy"]) and str(row["policy"]) != "None" else "Figure"
    return "{0} | {1}".format(env_part, policy_part)


def _render_active_dataset_banner() -> None:
    meta = cached_course_dataset_metadata()
    if not meta:
        st.error("The persistent 1GB course dataset metadata is missing. This final demo is expected to run with the stored course dataset.")
        return
    dataset_path = resolve_course_dataset_path()
    size_bytes = int(meta.get("actual_bytes", 0))
    total_records = int(meta.get("total_records", 0))
    bytes_per_record = float(meta.get("bytes_per_record", 0.0))
    st.markdown(
        (
            "**Active project dataset.** The final demo uses the persistent raw trace corpus at "
            "`{path}` with `{records}` records, `{size}` (`{bytes}` bytes), and `{bpr:.3f}` bytes per record. "
            "This file is generated once and then reused for every presentation."
        ).format(
            path=dataset_path,
            records=total_records,
            size=_format_bytes(size_bytes),
            bytes=size_bytes,
            bpr=bytes_per_record,
        )
    )


def _reset_task_session(case_signature: Optional[Dict[str, Any]] = None) -> None:
    st.session_state["task_session_started_at"] = time.time()
    st.session_state["task_focus_changes"] = 0
    st.session_state["task_filter_changes"] = 0
    st.session_state["task_comparison_changes"] = 0
    st.session_state["task_export_count"] = 0
    st.session_state["task_notebook_saves"] = 0
    st.session_state["task_tracking_signature"] = {}
    st.session_state["task_case_signature"] = case_signature or {}


def _ensure_task_session(case_signature: Dict[str, Any]) -> None:
    if "task_session_started_at" not in st.session_state:
        _reset_task_session(case_signature)
        return
    if st.session_state.get("task_case_signature") != case_signature:
        _reset_task_session(case_signature)


def _track_task_session_state(
    selected_window_k: Optional[int],
    filter_signature: Dict[str, Any],
    compare_signature: Dict[str, Any],
) -> None:
    current_signature = {
        "selected_window_k": selected_window_k,
        "filter_signature": filter_signature,
        "compare_signature": compare_signature,
    }
    previous_signature = st.session_state.get("task_tracking_signature", {})
    if previous_signature:
        if previous_signature.get("selected_window_k") != current_signature["selected_window_k"]:
            st.session_state["task_focus_changes"] = int(st.session_state.get("task_focus_changes", 0)) + 1
        if previous_signature.get("filter_signature") != current_signature["filter_signature"]:
            st.session_state["task_filter_changes"] = int(st.session_state.get("task_filter_changes", 0)) + 1
        if previous_signature.get("compare_signature") != current_signature["compare_signature"]:
            st.session_state["task_comparison_changes"] = int(st.session_state.get("task_comparison_changes", 0)) + 1
    st.session_state["task_tracking_signature"] = current_signature


def _register_export_event() -> None:
    st.session_state["task_export_count"] = int(st.session_state.get("task_export_count", 0)) + 1


def _sync_demo_state(default_env: str, default_policy: str) -> None:
    st.session_state.setdefault("demo_role", PRIMARY_ANALYST_ROLE)
    st.session_state.setdefault("demo_env", default_env)
    st.session_state.setdefault("demo_policy", default_policy)
    st.session_state.setdefault("demo_seed", None)
    st.session_state.setdefault("demo_task", "Tell the project story")
    st.session_state.setdefault("selected_k", None)
    st.session_state.setdefault("selected_cluster", None)
    _ensure_task_session(
        {
            "workflow": st.session_state.get("demo_workflow", ""),
            "role": st.session_state.get("demo_role", PRIMARY_ANALYST_ROLE),
            "task": st.session_state.get("demo_task", "Tell the project story"),
            "env": st.session_state.get("demo_env", default_env),
            "policy": st.session_state.get("demo_policy", default_policy),
            "seed": st.session_state.get("demo_seed"),
        }
    )


def _apply_demo_preset(label: str) -> None:
    preset = DEMO_PRESETS[label]
    st.session_state["demo_env"] = str(preset["env"])
    st.session_state["demo_policy"] = str(preset["policy"])
    st.session_state["demo_seed"] = int(preset["seed"])
    st.session_state["demo_task"] = str(preset["task"])
    st.session_state["demo_color_by"] = str(preset["metric"])
    st.session_state["selected_k"] = None
    st.session_state["selected_cluster"] = None
    _reset_task_session(
        {
            "workflow": st.session_state.get("demo_workflow", ""),
            "role": st.session_state.get("demo_role", PRIMARY_ANALYST_ROLE),
            "task": st.session_state.get("demo_task", ""),
            "env": st.session_state.get("demo_env", ""),
            "policy": st.session_state.get("demo_policy", ""),
            "seed": st.session_state.get("demo_seed"),
        }
    )


def _apply_role_preset(role_name: str) -> None:
    defaults = ROLE_DEFAULTS[role_name]
    st.session_state["demo_role"] = role_name
    st.session_state["demo_task"] = defaults["task"]
    _apply_demo_preset(defaults["preset"])


def _user_task_label(task_name: str) -> str:
    mapping = {
        "Find anomalies": "Find unusual traffic behavior",
        "Locate shift": "Find where the traffic system changed",
        "Compare policies": "Compare two signal strategies",
        "Tell the project story": "Walk through the traffic-review system end-to-end",
    }
    return mapping.get(task_name, task_name)


def _color_label(color_by: str) -> str:
    mapping = {
        "anomaly_score": "Suspicion level",
        "regime_shift_score": "Change-point confidence",
        "switch_rate": "Action-switch rate",
        "r_bar": "Average reward",
        "H_bar": "Behavior entropy",
    }
    return mapping.get(color_by, color_by)


def _mode_subset_label(mode_name: str) -> str:
    mapping = {
        "All filtered windows": "All relevant time segments",
        "Only anomalous windows": "Only unusual time segments",
        "Only shift windows": "Only system-change time segments",
        "Only intervention windows": "Only labeled intervention time segments",
    }
    return mapping.get(mode_name, mode_name)


def _window_label(window_row: pd.Series) -> str:
    return "k={0} | t={1}-{2}".format(int(window_row["k"]), int(window_row["t_start"]), int(window_row["t_end"]))


def _entry_summary_line(entry: Dict[str, Any]) -> str:
    return "{0} | {1} | {2} | {3}".format(
        entry.get("created_at", ""),
        entry.get("status", ""),
        entry.get("system", ""),
        entry.get("title", ""),
    )


def _user_problem_statement(role_name: str, task_name: str, env_name: str, policy: str) -> str:
    task_label = _user_task_label(task_name)
    if role_name == PRIMARY_SUPERVISOR_ROLE:
        return "Determine whether `{0}` in `{1}` should remain deployed, be revised, or be compared further using `{2}` evidence.".format(
            _format_policy_name(policy),
            _format_env_name(env_name),
            task_label.lower(),
        )
    return "Determine whether the current `{0}` traffic case in `{1}` is severe enough to escalate using `{2}` evidence.".format(
        _format_policy_name(policy),
        _format_env_name(env_name),
        task_label.lower(),
    )


def _task_question_rows(
    task_name: str,
    role_name: str,
    env_name: str,
    policy: str,
    filtered_df: pd.DataFrame,
    window_row: pd.Series,
    compare_row: Optional[pd.Series],
) -> pd.DataFrame:
    questions = TASK_QUESTION_BANK.get(task_name, TASK_QUESTION_BANK["Find anomalies"])
    suspicious_count = int((filtered_df["anomaly_score"] >= float(window_row["anomaly_score"])).sum())
    persistent_count = int(
        filtered_df["k"].between(max(int(window_row["k"]) - 1, int(filtered_df["k"].min())), min(int(window_row["k"]) + 1, int(filtered_df["k"].max()))).sum()
    )
    intervention_overlap = bool(window_row["gt_intervention"]) if "gt_intervention" in window_row.index else False

    if task_name == "Locate shift":
        rows = [
            {
                "question": questions[0],
                "current answer": "Focused boundary is `{0}` with change confidence `{1:.3f}`.".format(
                    _window_label(window_row),
                    float(window_row["regime_shift_score"]),
                ),
                "evidence source": "Timeline + linked focus window",
            },
            {
                "question": questions[1],
                "current answer": "{0} nearby windows stay inside the focused neighborhood, which helps verify the transition.".format(
                    persistent_count
                ),
                "evidence source": "Timeline neighborhood around the selected window",
            },
            {
                "question": questions[2],
                "current answer": "Escalation signal is `{0}` for the current case.".format(
                    "strong" if float(window_row["regime_shift_score"]) >= 0.8 or intervention_overlap else "moderate"
                ),
                "evidence source": "Decision summary + intervention overlap",
            },
        ]
    elif task_name == "Compare policies":
        compare_answer = "Comparison is not available yet for the current controller pair."
        compare_evidence = "Controller-comparison view"
        if compare_row is not None:
            compare_answer = "Behavior group `{0}` shows difference `{1:.3f}` with outcome gap `{2:.3f}`.".format(
                int(compare_row["cluster"]),
                float(compare_row["js_div"]),
                float(compare_row["reward_gap"]),
            )
        rows = [
            {
                "question": questions[0],
                "current answer": "Current anchor case is `{0}` for `{1}` in `{2}`.".format(
                    _window_label(window_row),
                    _format_policy_name(policy),
                    _format_env_name(env_name),
                ),
                "evidence source": "Ranked windows + linked focus",
            },
            {
                "question": questions[1],
                "current answer": compare_answer,
                "evidence source": compare_evidence,
            },
            {
                "question": questions[2],
                "current answer": "User should trust the controller with lower risk windows and weaker cross-policy divergence in the active scenario.",
                "evidence source": "Decision summary + comparison export",
            },
        ]
    else:
        rows = [
            {
                "question": questions[0],
                "current answer": "Focused case `{0}` has suspicion `{1:.3f}` and ranks within the top `{2}` visible windows.".format(
                    _window_label(window_row),
                    float(window_row["anomaly_score"]),
                    max(1, suspicious_count),
                ),
                "evidence source": "Ranked window list",
            },
            {
                "question": questions[1],
                "current answer": "{0} nearby windows remain in view, so the user can test whether the signal persists.".format(
                    persistent_count
                ),
                "evidence source": "Timeline + filtered neighborhood",
            },
            {
                "question": questions[2],
                "current answer": "{0} should `{1}` based on the current evidence.".format(
                    role_name,
                    "escalate now" if float(window_row["anomaly_score"]) >= 0.8 or intervention_overlap else "continue review",
                ),
                "evidence source": "Action tab + notebook workflow",
            },
        ]
    return pd.DataFrame(rows)


def _question_answer_records(question_frame: pd.DataFrame) -> List[Dict[str, str]]:
    if question_frame.empty:
        return []
    rows: List[Dict[str, str]] = []
    for _, row in question_frame.iterrows():
        rows.append(
            {
                "question": str(row.get("question", "")),
                "current_answer": str(row.get("current answer", "")),
                "evidence_source": str(row.get("evidence source", "")),
            }
        )
    return rows


def _question_answer_markdown_lines(question_frame: pd.DataFrame, heading: str = "## Fundamental Question Answers") -> List[str]:
    lines = [heading, ""]
    records = _question_answer_records(question_frame)
    if not records:
        lines.extend(["No question answers are available for the current case.", ""])
        return lines
    for record in records:
        lines.extend(
            [
                "### {0}".format(record["question"]),
                "",
                "- Current answer: {0}".format(record["current_answer"]),
                "- Evidence source: {0}".format(record["evidence_source"]),
                "",
            ]
        )
    return lines


def _decision_readiness(task_name: str, filtered_df: pd.DataFrame, window_row: pd.Series, compare_row: Optional[pd.Series]) -> Dict[str, Any]:
    intervention_overlap = bool(window_row["gt_intervention"]) if "gt_intervention" in window_row.index else False
    checks = [
        {
            "label": "Focused evidence selected",
            "ready": bool(len(filtered_df) > 0),
            "detail": "A single time segment is active across the linked views.",
        },
        {
            "label": "Risk or change threshold met",
            "ready": bool(float(window_row["anomaly_score"]) >= 0.5 or float(window_row["regime_shift_score"]) >= 0.5),
            "detail": "The current window exceeds at least one user-facing review threshold.",
        },
        {
            "label": "Reference support available",
            "ready": intervention_overlap,
            "detail": "The focused case overlaps a labeled intervention or change marker.",
        },
        {
            "label": "Comparison evidence available",
            "ready": bool(compare_row is not None) if task_name == "Compare policies" else True,
            "detail": "Cross-controller evidence is available when the current task needs it.",
        },
    ]
    ready_count = sum(1 for check in checks if check["ready"])
    if ready_count >= 4:
        level = "Ready to export"
    elif ready_count >= 3:
        level = "Ready for review"
    else:
        level = "Needs more evidence"
    return {
        "level": level,
        "score": ready_count,
        "total": len(checks),
        "checks": checks,
    }


def _compose_case_note_entry(
    workflow_name: str,
    role_name: str,
    task_name: str,
    env_name: str,
    policy: str,
    selected_seed: Optional[int],
    window_row: pd.Series,
    decision: Dict[str, Any],
    note_title: str,
    note_body: str,
    note_status: str,
    note_owner: str,
    note_tags: List[str],
    compare_row: Optional[pd.Series],
    question_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    problem_statement = _user_problem_statement(role_name, task_name, env_name, policy)
    entry: Dict[str, Any] = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "workflow": workflow_name,
        "role": role_name,
        "task": _user_task_label(task_name),
        "execution_scenario": ROLE_SCENARIO_COPY.get(role_name, ""),
        "user_problem": problem_statement,
        "system": _format_env_name(env_name),
        "controller": _format_policy_name(policy),
        "run": int(selected_seed) if selected_seed is not None else None,
        "window_label": _window_label(window_row),
        "window_k": int(window_row["k"]),
        "window_start": int(window_row["t_start"]),
        "window_end": int(window_row["t_end"]),
        "status": note_status,
        "owner": note_owner,
        "risk_level": decision["severity"],
        "risk_flag": float(window_row["anomaly_score"]),
        "change_confidence": float(window_row["regime_shift_score"]),
        "average_outcome": float(window_row["r_bar"]),
        "title": note_title.strip() or "Untitled case note",
        "note": note_body.strip(),
        "tags": note_tags,
        "decision": decision["decision"],
        "risk_summary": decision["risk"],
        "recommended_action": decision["recommendation"],
        "question_answers": _question_answer_records(question_frame if question_frame is not None else pd.DataFrame()),
    }
    if compare_row is not None:
        entry["behavior_group"] = int(compare_row["cluster"])
        entry["behavior_difference"] = float(compare_row["js_div"])
        entry["outcome_gap"] = float(compare_row["reward_gap"])
    return entry


def _build_case_note_markdown(entry: Dict[str, Any]) -> str:
    lines = [
        "# Saved Case Note",
        "",
        "- Created: {0}".format(entry.get("created_at", "")),
        "- Workflow: {0}".format(entry.get("workflow", "")),
        "- Role: {0}".format(entry.get("role", "")),
        "- Task: {0}".format(entry.get("task", "")),
        "- Execution scenario: {0}".format(entry.get("execution_scenario", "")),
        "- System: {0}".format(entry.get("system", "")),
        "- Controller: {0}".format(entry.get("controller", "")),
        "- Run: {0}".format(entry.get("run", "n/a")),
        "- Focused window: {0}".format(entry.get("window_label", "")),
        "- Status: {0}".format(entry.get("status", "")),
        "- Owner: {0}".format(entry.get("owner", "")),
        "- Risk level: {0}".format(entry.get("risk_level", "")),
        "- Risk flag: {0:.3f}".format(float(entry.get("risk_flag", 0.0))),
        "- Change confidence: {0:.3f}".format(float(entry.get("change_confidence", 0.0))),
        "- Average outcome: {0:.3f}".format(float(entry.get("average_outcome", 0.0))),
        "",
        "## User problem",
        "",
        str(entry.get("user_problem", "")),
        "",
    ]
    question_answers = entry.get("question_answers", [])
    if question_answers:
        lines.extend(["## Fundamental question answers", ""])
        for record in question_answers:
            lines.extend(
                [
                    "### {0}".format(str(record.get("question", ""))),
                    "",
                    "- Current answer: {0}".format(str(record.get("current_answer", ""))),
                    "- Evidence source: {0}".format(str(record.get("evidence_source", ""))),
                    "",
                ]
            )
    lines.extend(
        [
            "## Title",
            "",
            str(entry.get("title", "")),
            "",
            "## Analyst note",
            "",
            str(entry.get("note", "")) or "No note entered.",
            "",
            "## Decision",
            "",
            str(entry.get("decision", "")),
            "",
            "## Risk summary",
            "",
            str(entry.get("risk_summary", "")),
            "",
            "## Recommended action",
            "",
            str(entry.get("recommended_action", "")),
            "",
        ]
    )
    tags = entry.get("tags", [])
    if tags:
        lines.extend(["## Tags", "", "- " + "\n- ".join(str(tag) for tag in tags), ""])
    if "behavior_group" in entry:
        lines.extend(
            [
                "## Comparison evidence",
                "",
                "- Behavior group: {0}".format(entry.get("behavior_group", "")),
                "- Behavior difference: {0:.3f}".format(float(entry.get("behavior_difference", 0.0))),
                "- Outcome gap: {0:.3f}".format(float(entry.get("outcome_gap", 0.0))),
                "",
            ]
        )
    return "\n".join(lines)


def _build_review_memo_markdown(
    role_name: str,
    workflow_name: str,
    env_name: str,
    policy: str,
    selected_seed: Optional[int],
    task_name: str,
    window_row: pd.Series,
    decision: Dict[str, Any],
    note_title: str,
    note_body: str,
    question_frame: Optional[pd.DataFrame] = None,
) -> str:
    header = "Signal Strategy Review Memo" if role_name == PRIMARY_SUPERVISOR_ROLE else "Traffic Incident Review Memo"
    problem_statement = _user_problem_statement(role_name, task_name, env_name, policy)
    lines = [
        "# {0}".format(header),
        "",
        "- Workflow: {0}".format(workflow_name),
        "- Prepared for: {0}".format(role_name),
        "- Execution scenario: {0}".format(ROLE_SCENARIO_COPY.get(role_name, "")),
        "- System: {0}".format(_format_env_name(env_name)),
        "- Controller: {0}".format(_format_policy_name(policy)),
        "- Run: {0}".format(selected_seed if selected_seed is not None else "n/a"),
        "- Focused window: {0}".format(_window_label(window_row)),
        "- User question: {0}".format(_user_task_label(task_name)),
        "- Risk level: {0}".format(decision["severity"]),
        "- Risk flag: {0:.3f}".format(float(window_row["anomaly_score"])),
        "- Change confidence: {0:.3f}".format(float(window_row["regime_shift_score"])),
        "- Average outcome: {0:.3f}".format(float(window_row["r_bar"])),
        "",
        "## Decision",
        "",
        decision["decision"],
        "",
        "## User problem",
        "",
        problem_statement,
        "",
    ]
    lines.extend(_question_answer_markdown_lines(question_frame if question_frame is not None else pd.DataFrame()))
    lines.extend(
        [
            "## Risk assessment",
            "",
            decision["risk"],
            "",
            "## Recommended action",
            "",
            decision["recommendation"],
            "",
            "## Analyst summary",
            "",
            note_title.strip() or "No custom title entered.",
            "",
            note_body.strip() or "No additional analyst note entered.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_task_answer_brief_markdown(
    workflow_name: str,
    role_name: str,
    task_name: str,
    env_name: str,
    policy: str,
    selected_seed: Optional[int],
    window_row: pd.Series,
    decision: Dict[str, Any],
    question_frame: pd.DataFrame,
) -> str:
    lines = [
        "# Workflow Answer Brief",
        "",
        "- Workflow: {0}".format(workflow_name),
        "- Primary user: {0}".format(role_name),
        "- Decision goal: {0}".format(_user_task_label(task_name)),
        "- System: {0}".format(_format_env_name(env_name)),
        "- Controller: {0}".format(_format_policy_name(policy)),
        "- Run: {0}".format(selected_seed if selected_seed is not None else "n/a"),
        "- Focused window: {0}".format(_window_label(window_row)),
        "- Decision level: {0}".format(decision["severity"]),
        "",
        "## User problem",
        "",
        _user_problem_statement(role_name, task_name, env_name, policy),
        "",
    ]
    lines.extend(_question_answer_markdown_lines(question_frame, heading="## Current Answers"))
    lines.extend(
        [
            "## Recommended action",
            "",
            decision["recommendation"],
            "",
            "## Decision summary",
            "",
            decision["decision"],
            "",
        ]
    )
    return "\n".join(lines)


def _task_support_signal(task_name: str, window_row: pd.Series, compare_row: Optional[pd.Series]) -> Tuple[bool, str]:
    intervention_overlap = bool(window_row["gt_intervention"]) if "gt_intervention" in window_row.index else False
    shift_start = bool(window_row["gt_shift_start"]) if "gt_shift_start" in window_row.index else False
    if task_name == "Locate shift":
        return shift_start or float(window_row.get("regime_shift_score", 0.0)) >= 0.8, "shift boundary support"
    if task_name == "Compare policies":
        compare_ready = bool(compare_row is not None)
        return compare_ready, "controller comparison evidence"
    return intervention_overlap or float(window_row.get("anomaly_score", 0.0)) >= 0.8, "incident evidence support"


def _compose_task_log_entry(
    workflow_name: str,
    role_name: str,
    task_name: str,
    env_name: str,
    policy: str,
    selected_seed: Optional[int],
    window_row: pd.Series,
    decision: Dict[str, Any],
    question_frame: pd.DataFrame,
    compare_row: Optional[pd.Series],
    user_confidence: float,
    reviewer_name: str,
    tracking_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tracking_snapshot = tracking_snapshot or {}
    support_signal, support_reason = _task_support_signal(task_name, window_row, compare_row)
    started_at = float(tracking_snapshot.get("task_session_started_at", st.session_state.get("task_session_started_at", time.time())))
    completed_at = time.time()
    focus_changes = int(tracking_snapshot.get("task_focus_changes", st.session_state.get("task_focus_changes", 0)))
    filter_changes = int(tracking_snapshot.get("task_filter_changes", st.session_state.get("task_filter_changes", 0)))
    comparison_changes = int(tracking_snapshot.get("task_comparison_changes", st.session_state.get("task_comparison_changes", 0)))
    export_count = int(tracking_snapshot.get("task_export_count", st.session_state.get("task_export_count", 0)))
    notebook_saves = int(tracking_snapshot.get("task_notebook_saves", st.session_state.get("task_notebook_saves", 0)))
    entry: Dict[str, Any] = {
        "started_at": datetime.utcfromtimestamp(started_at).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "completed_at": datetime.utcfromtimestamp(completed_at).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "workflow": workflow_name,
        "role": role_name,
        "reviewer_name": reviewer_name.strip() or role_name,
        "task": _user_task_label(task_name),
        "task_key": task_name,
        "execution_scenario": ROLE_SCENARIO_COPY.get(role_name, ""),
        "system": _format_env_name(env_name),
        "controller": _format_policy_name(policy),
        "run": int(selected_seed) if selected_seed is not None else None,
        "focused_window": _window_label(window_row),
        "window_k": int(window_row["k"]),
        "duration_seconds": max(0.0, float(completed_at - started_at)),
        "focus_changes": focus_changes,
        "filter_changes": filter_changes,
        "comparison_changes": comparison_changes,
        "exports_triggered": export_count,
        "notebook_saves": notebook_saves,
        "export_used": bool(export_count > 0),
        "notebook_used": bool(notebook_saves > 0),
        "decision_level": str(decision.get("severity", "")),
        "decision": str(decision.get("decision", "")),
        "recommended_action": str(decision.get("recommendation", "")),
        "user_confidence": float(user_confidence),
        "anomaly_score": float(window_row.get("anomaly_score", 0.0)),
        "regime_shift_score": float(window_row.get("regime_shift_score", 0.0)),
        "support_signal": bool(support_signal),
        "support_reason": support_reason,
        "question_answers": _question_answer_records(question_frame),
    }
    if compare_row is not None:
        entry["behavior_group"] = int(compare_row["cluster"])
        entry["behavior_difference"] = float(compare_row["js_div"])
        entry["outcome_gap"] = float(compare_row["reward_gap"])
    return entry


def _build_comparison_brief_markdown(
    env_name: str,
    p1: str,
    p2: str,
    cluster_count: int,
    cluster_row: pd.Series,
) -> str:
    lines = [
        "# Controller Comparison Brief",
        "",
        "- System: {0}".format(_format_env_name(env_name)),
        "- Controller 1: {0}".format(_format_policy_name(p1)),
        "- Controller 2: {0}".format(_format_policy_name(p2)),
        "- Behavior groups: {0}".format(cluster_count),
        "- Focused group: {0}".format(int(cluster_row["cluster"])),
        "- Behavior difference: {0:.3f}".format(float(cluster_row["js_div"])),
        "- Outcome gap: {0:.3f}".format(float(cluster_row["reward_gap"])),
        "- Group size: {0:.0f}".format(float(cluster_row["n_total"])),
        "",
        "## Plain-language summary",
        "",
        _cluster_explain(cluster_row, env_name=env_name, p1=p1, p2=p2, cluster_count=cluster_count),
        "",
    ]
    return "\n".join(lines)


def _build_notebook_digest_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Case Notebook Digest",
        "",
        "- Saved cases: {0}".format(len(frame)),
        "",
        "## Cases",
        "",
    ]
    if frame.empty:
        lines.append("No saved cases yet.")
        lines.append("")
        return "\n".join(lines)
    for _, row in frame.iterrows():
        lines.append(
            "- `{0}` | {1} | {2} | {3} | {4}".format(
                str(row.get("created_at", "")),
                str(row.get("status", "")),
                str(row.get("system", "")),
                str(row.get("controller", "")),
                str(row.get("title", "")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _action_recommendations(task_name: str, window_row: pd.Series) -> List[str]:
    steps: List[str] = []
    if task_name == "Find anomalies":
        steps.append("Inspect adjacent traffic windows around the focused segment to see whether the suspicious corridor behavior is isolated or persistent.")
        steps.append("Compare the active controller against a baseline strategy in the same corridor to decide whether the issue is controller-specific.")
    elif task_name == "Locate shift":
        steps.append("Check the windows immediately before and after the focused segment to confirm the traffic regime changed rather than fluctuated briefly.")
        steps.append("Use the controller-comparison view to test whether alternate strategies react differently around the same corridor boundary.")
    elif task_name == "Compare policies":
        steps.append("Export the focused comparison cluster and use it as evidence for which signal strategy is more stable or more behaviorally distinct.")
        steps.append("Return to the linked analysis view to inspect whether the same difference appears in the highest-risk traffic windows.")
    else:
        steps.append("Use the dataset overview first, then show the linked views, and finish by exporting the current corridor-review evidence snapshot.")
        steps.append("Use one traffic preset per user role so the audience sees how the system supports multiple operations workflows.")
    if "gt_intervention" in window_row.index and bool(window_row["gt_intervention"]):
        steps.append("Because this window overlaps a labeled intervention, treat it as a high-priority traffic review case.")
    return steps


def _diagnostic_summary(task_name: str, env_name: str, policy: str, window_row: pd.Series) -> Dict[str, Any]:
    if task_name == "Locate shift":
        happened = "The traffic corridor appears to change operating mode around this window."
        why = "The change-point confidence is high and the surrounding windows show a visible transition in corridor summary behavior."
    elif task_name == "Compare policies":
        happened = "The focused signal strategy reaches a high-signal behavior region that should be compared against alternate plans."
        why = "This window is a good anchor for cross-controller comparison because it is both behaviorally distinctive and easy to trace in corridor time."
    elif task_name == "Tell the project story":
        happened = "This window is the clearest anchor point for explaining how the full traffic-review system turns raw traces into evidence."
        why = "It connects the user question, the linked visual views, and the exported evidence in one place."
    else:
        happened = "The system flags this corridor window as behaviorally suspicious."
        why = "Its suspicion level is high relative to the visible windows and it coincides with a meaningful change in traffic summary behavior."
    next_steps = _action_recommendations(task_name, window_row)
    return {
        "happened": happened,
        "why": why,
        "next_steps": next_steps,
        "user_env": _format_env_name(env_name),
        "user_policy": _format_policy_name(policy),
    }


def _decision_summary(role_name: str, task_name: str, env_name: str, policy: str, window_row: pd.Series, compare_row: Optional[pd.Series]) -> Dict[str, Any]:
    suspicion = float(window_row["anomaly_score"])
    change_conf = float(window_row["regime_shift_score"])
    reward = float(window_row["r_bar"])
    intervention_flag = bool(window_row["gt_intervention"]) if "gt_intervention" in window_row.index else False

    if suspicion >= 0.8 or change_conf >= 0.8:
        severity = "High"
    elif suspicion >= 0.5 or change_conf >= 0.5:
        severity = "Moderate"
    else:
        severity = "Low"

    if role_name == PRIMARY_SUPERVISOR_ROLE:
        decision = "Pause rollout of the active signal strategy and review the focused behavior." if severity == "High" else "The strategy is reviewable, but inspect the highlighted corridor evidence before keeping it live."
        owner = PRIMARY_SUPERVISOR_OWNER
    elif role_name == PRIMARY_ANALYST_ROLE:
        decision = "Escalate this corridor segment for incident review." if severity == "High" else "Monitor this segment and compare it against normal corridor windows."
        owner = PRIMARY_OPERATIONS_OWNER
    else:
        decision = "Use this case as the main teaching and presentation example." if severity != "Low" else "Use this case as supporting evidence rather than the main demo example."
        owner = "presentation / instruction owner"

    evidence = [
        "Focused window: k={0}, t={1}-{2}".format(int(window_row["k"]), int(window_row["t_start"]), int(window_row["t_end"])),
        "Suspicion level: {0:.3f}".format(suspicion),
        "Change-point confidence: {0:.3f}".format(change_conf),
        "Average outcome: {0:.3f}".format(reward),
    ]
    if intervention_flag:
        evidence.append("The focused window overlaps a labeled intervention.")
    if compare_row is not None:
        evidence.append(
            "Comparison evidence: behavior group {0} shows behavior difference {1:.3f} and outcome gap {2:.3f}.".format(
                int(compare_row["cluster"]),
                float(compare_row["js_div"]),
                float(compare_row["reward_gap"]),
            )
        )

    risk_line = "Current severity is {0} for {1} in {2}.".format(severity, _format_policy_name(policy), _format_env_name(env_name))
    if severity == "High":
        risk_line += " The current evidence suggests the behavior should not be treated as routine."
    elif severity == "Moderate":
        risk_line += " The current evidence suggests a meaningful but not yet conclusive issue."
    else:
        risk_line += " The current evidence is useful mainly as context or supporting material."

    if task_name == "Compare policies":
        recommendation = "Export both the focused window and the focused controller-comparison snapshot for the signal strategy review."
    elif task_name == "Locate shift":
        recommendation = "Review the windows immediately before and after the change boundary to confirm the traffic transition."
    else:
        recommendation = "Use the linked views to inspect neighboring traffic windows, then export the current corridor evidence snapshot."

    if role_name == PRIMARY_SUPERVISOR_ROLE:
        action_title = "Recommended action: strategy review"
    else:
        action_title = "Recommended action: incident follow-up"

    return {
        "decision": decision,
        "severity": severity,
        "owner": owner,
        "evidence": evidence,
        "risk": risk_line,
        "recommendation": recommendation,
        "action_title": action_title,
    }


def _focus_selected_window(filtered_df: pd.DataFrame, task_name: str) -> pd.Series:
    if task_name == "Locate shift" and filtered_df["regime_shift_score"].notna().any():
        return filtered_df.sort_values(["regime_shift_score", "anomaly_score"], ascending=[False, False]).iloc[0]
    if task_name == "Find anomalies" and filtered_df["anomaly_score"].notna().any():
        return filtered_df.sort_values(["anomaly_score", "regime_shift_score"], ascending=[False, False]).iloc[0]
    return _top_window_finding(filtered_df)


def _window_rank_frame(filtered_df: pd.DataFrame, task_name: str, top_k: int) -> pd.DataFrame:
    frame = filtered_df.copy()
    if task_name == "Locate shift":
        frame = frame.sort_values(["regime_shift_score", "anomaly_score", "r_bar"], ascending=[False, False, True])
    else:
        frame = frame.sort_values(["anomaly_score", "regime_shift_score", "r_bar"], ascending=[False, False, True])
    return frame.head(top_k).reset_index(drop=True)


def _recommended_compare_pair(task_name: str, active_policy: str) -> Tuple[str, str]:
    if task_name == "Compare policies":
        return active_policy, "ppo" if active_policy != "ppo" else "pg"
    if task_name == "Locate shift":
        return active_policy, "heuristic" if active_policy != "heuristic" else "pg"
    return active_policy, "random" if active_policy != "random" else "heuristic"


def _render_demo_panel() -> None:
    _sync_demo_state(default_env="traffic", default_policy="dqn")
    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">User Workspace</div>
        <div class="hero-title">Traffic Operations Review Studio</div>
        <p class="hero-copy">
            This workspace is built for live traffic operations. It helps corridor analysts isolate suspicious windows,
            confirm when the signal system changed regime, compare alternate signal strategies, and leave with a review memo
            rather than a pile of charts. `RLVA` remains the engine, but the product is framed for a concrete operations user.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**Who this is for.** Traffic operations analysts and supervisors reviewing corridor behavior and signal-strategy safety.")
    st.markdown("**What the user wants.** Triage a suspicious corridor interval, confirm whether the system changed, compare alternate signal plans, and leave with a handoff-ready memo.")
    st.markdown("**How the workspace behaves.** One focused traffic case drives the ranked evidence, charts, strategy comparison, decision summary, notebook, task log, and exports.")
    st.caption("The `traffic` environment is the primary product scenario. The other systems remain available as transfer and robustness evidence for the course demo.")

    quick_cols = st.columns(4)
    quick_cols[0].button("Start incident triage", use_container_width=True, on_click=_apply_demo_preset, args=("Traffic incident",))
    quick_cols[1].button("Start change review", use_container_width=True, on_click=_apply_demo_preset, args=("Traffic shift",))
    quick_cols[2].button("Start strategy comparison", use_container_width=True, on_click=_apply_demo_preset, args=("Traffic strategy comparison",))
    if quick_cols[3].button("Reset workspace", use_container_width=True):
        _sync_demo_state(default_env="traffic", default_policy="dqn")
        st.session_state["demo_env"] = "traffic"
        st.session_state["demo_policy"] = "dqn"
        st.session_state["demo_seed"] = 23
        st.session_state["demo_task"] = "Find anomalies"
        st.session_state["demo_role"] = PRIMARY_ANALYST_ROLE
        st.session_state["demo_color_by"] = "anomaly_score"
        st.session_state["selected_k"] = None
        st.session_state["selected_cluster"] = None
        _reset_task_session(
            {
                "workflow": st.session_state.get("demo_workflow", ""),
                "role": PRIMARY_ANALYST_ROLE,
                "task": "Find anomalies",
                "env": "traffic",
                "policy": "dqn",
                "seed": 23,
            }
        )
        _trigger_rerun()

    workflow_cols = st.columns([1.35, 3, 1.2])
    with workflow_cols[0]:
        workflow_name = st.selectbox("Workflow", options=list(USER_WORKFLOW_COPY.keys()), key="demo_workflow")
    selected_workflow = USER_WORKFLOW_COPY[workflow_name]
    with workflow_cols[1]:
        st.info("**Workflow summary.** {0}".format(selected_workflow["summary"]))
    if workflow_cols[2].button("Start workflow", use_container_width=True):
        st.session_state["demo_role"] = selected_workflow["role"]
        _apply_demo_preset(selected_workflow["preset"])
        _trigger_rerun()

    role_cols = st.columns([1.2, 3])
    role_name = st.session_state.get("demo_role", selected_workflow["role"])
    if role_name not in USER_ROLE_COPY:
        role_name = selected_workflow["role"]
        st.session_state["demo_role"] = role_name
    with role_cols[0]:
        role_name = st.selectbox("Primary user", options=list(USER_ROLE_COPY.keys()), key="demo_role")
    with role_cols[1]:
        st.info("**Primary user need.** {0}".format(USER_ROLE_COPY[role_name]))

    control_cols = st.columns([1.25, 1.15, 1.0, 1.0, 1.0, 1.0])
    with control_cols[0]:
        task_options = ["Find anomalies", "Locate shift", "Compare policies"]
        if st.session_state.get("demo_task") not in task_options:
            st.session_state["demo_task"] = ROLE_DEFAULTS[role_name]["task"]
        task_name = st.selectbox("Decision goal", options=task_options, key="demo_task", format_func=_user_task_label)
    with control_cols[1]:
        env_name = st.selectbox("System", options=ENVIRONMENTS, key="demo_env", format_func=lambda name: ENV_LABELS.get(name, name))
    with control_cols[2]:
        policy = st.selectbox("Controller", options=BENCHMARK_POLICIES, key="demo_policy", format_func=_format_policy_name)
    available_seeds = cached_available_seeds(env_name, policy)
    if available_seeds:
        if st.session_state.get("demo_seed") not in available_seeds:
            st.session_state["demo_seed"] = int(available_seeds[0])
    else:
        st.session_state["demo_seed"] = None
    with control_cols[3]:
        selected_seed = st.selectbox("Run", options=available_seeds, key="demo_seed") if available_seeds else None
    with control_cols[4]:
        color_by = st.selectbox(
            "Map emphasis",
            options=["anomaly_score", "regime_shift_score", "switch_rate", "r_bar", "H_bar"],
            key="demo_color_by",
            format_func=_color_label,
        )
    with control_cols[5]:
        focus_preset = st.selectbox(
            "Quick focus",
            options=["Balanced view", "Highest risk", "System change", "Current task only"],
            key="demo_focus_preset",
        )

    case_signature = {
        "workflow": workflow_name,
        "role": role_name,
        "task": task_name,
        "env": env_name,
        "policy": policy,
        "seed": selected_seed,
    }
    _ensure_task_session(case_signature)

    st.info(
        "**Active mission.** {0} **Current scene.** `{1}` / `{2}` / run `{3}`.".format(
            DEMO_TASK_COPY[task_name],
            _format_env_name(env_name),
            _format_policy_name(policy),
            selected_seed if selected_seed is not None else "default",
        )
    )

    summary_df = cached_summary(env_name, policy, seed=selected_seed).sort_values("k").reset_index(drop=True)
    if summary_df.empty:
        st.warning("No time-segment summaries are available for the current selection.")
        return

    k_min, k_max = int(summary_df["k"].min()), int(summary_df["k"].max())
    default_color = "regime_shift_score" if task_name == "Locate shift" else color_by
    if default_color != color_by:
        st.session_state["demo_color_by"] = default_color
        color_by = default_color
    anomaly_default = float(summary_df["anomaly_score"].quantile(0.50 if task_name == "Find anomalies" else 0.25))
    shift_default = float(summary_df["regime_shift_score"].quantile(0.50 if task_name == "Locate shift" else 0.25))
    if focus_preset == "Highest risk":
        anomaly_default = float(summary_df["anomaly_score"].quantile(0.75))
        shift_default = float(summary_df["regime_shift_score"].quantile(0.60))
    elif focus_preset == "System change":
        anomaly_default = float(summary_df["anomaly_score"].quantile(0.25))
        shift_default = float(summary_df["regime_shift_score"].quantile(0.75))
    elif focus_preset == "Current task only":
        if task_name == "Locate shift":
            anomaly_default = float(summary_df["anomaly_score"].quantile(0.20))
            shift_default = float(summary_df["regime_shift_score"].quantile(0.75))
        elif task_name == "Find anomalies":
            anomaly_default = float(summary_df["anomaly_score"].quantile(0.75))
            shift_default = float(summary_df["regime_shift_score"].quantile(0.20))

    with st.expander("Focus filters", expanded=True):
        filter_cols = st.columns([1.35, 1.0, 1.0, 1.1])
        with filter_cols[0]:
            k_range = st.slider("Time segment range", min_value=k_min, max_value=k_max, value=(k_min, k_max))
        with filter_cols[1]:
            top_k_windows = st.slider("Top time segments", min_value=3, max_value=min(20, len(summary_df)), value=min(8, len(summary_df)))
        with filter_cols[2]:
            focus_mode = st.selectbox(
                "Window scope",
                options=["All filtered windows", "Only anomalous windows", "Only shift windows", "Only intervention windows"],
                format_func=_mode_subset_label,
            )
        with filter_cols[3]:
            if task_name == "Compare policies":
                st.info("Comparison uses the same focused windows, then switches to behavior-group evidence.")
            elif task_name == "Locate shift":
                st.info("Change review emphasizes windows with stronger change confidence.")
            else:
                st.info("Risk review emphasizes windows with stronger unusual-behavior signal.")
        threshold_cols = st.columns(2)
        with threshold_cols[0]:
            anomaly_threshold = st.slider(
                "Risk flag threshold",
                min_value=float(summary_df["anomaly_score"].min()),
                max_value=float(summary_df["anomaly_score"].max()),
                value=anomaly_default,
            )
        with threshold_cols[1]:
            shift_threshold = st.slider(
                "Change confidence threshold",
                min_value=float(summary_df["regime_shift_score"].min()),
                max_value=float(summary_df["regime_shift_score"].max()),
                value=shift_default,
            )

    filtered_df = summary_df[summary_df["k"].between(k_range[0], k_range[1])].copy()
    filtered_df = filtered_df[
        filtered_df["anomaly_score"].ge(float(anomaly_threshold)) | filtered_df["regime_shift_score"].ge(float(shift_threshold))
    ].copy()
    if focus_mode == "Only anomalous windows":
        filtered_df = filtered_df[filtered_df["is_anomaly"]]
    elif focus_mode == "Only shift windows":
        filtered_df = filtered_df[filtered_df["is_regime_shift"]]
    elif focus_mode == "Only intervention windows" and "gt_intervention" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["gt_intervention"]]

    if filtered_df.empty:
        st.warning("No time segments remain after the current filters.")
        return

    top_window = _focus_selected_window(filtered_df, task_name)
    if st.session_state.get("selected_k") not in filtered_df["k"].astype(int).tolist():
        st.session_state["selected_k"] = int(top_window["k"])
    selected_window_row = filtered_df[filtered_df["k"] == int(st.session_state["selected_k"])]
    if selected_window_row.empty:
        selected_window_row = filtered_df.iloc[[0]]
        st.session_state["selected_k"] = int(selected_window_row.iloc[0]["k"])
    selected_window = selected_window_row.iloc[0]

    st.session_state.setdefault("case_note_title", "")
    st.session_state.setdefault("case_note_body", "")
    st.session_state.setdefault("case_note_status", "Needs review")
    default_case_owner = PRIMARY_SUPERVISOR_OWNER if role_name == PRIMARY_SUPERVISOR_ROLE else PRIMARY_OPERATIONS_OWNER
    if st.session_state.get("case_note_owner") in {None, "", "model validation team", "operations review team"}:
        st.session_state["case_note_owner"] = default_case_owner
    st.session_state.setdefault("case_note_tags", [])

    metrics = _summary_metrics(filtered_df)
    headline_cols = st.columns(5)
    headline_cols[0].metric("Visible time segments", int(metrics["windows"]))
    headline_cols[1].metric("Average outcome", "{0:.3f}".format(metrics["reward_mean"]))
    headline_cols[2].metric("Peak risk flag", "{0:.3f}".format(float(filtered_df["anomaly_score"].max())))
    headline_cols[3].metric("Peak change confidence", "{0:.3f}".format(float(filtered_df["regime_shift_score"].max())))
    headline_cols[4].metric("Reference overlap", "{0:.1%}".format(float(filtered_df["gt_intervention"].mean()) if "gt_intervention" in filtered_df.columns else 0.0))
    task_cols = st.columns(4)
    task_cols[0].metric("Task timer", "{0:.1f}s".format(max(0.0, float(time.time() - float(st.session_state.get("task_session_started_at", time.time()))))))
    task_cols[1].metric("Focus changes", int(st.session_state.get("task_focus_changes", 0)))
    task_cols[2].metric("Filter changes", int(st.session_state.get("task_filter_changes", 0)))
    task_cols[3].metric("Comparison changes", int(st.session_state.get("task_comparison_changes", 0)))

    st.caption(
        "Active filters: windows `{0}-{1}` | scope `{2}` | risk flag `>= {3:.3f}` | change confidence `>= {4:.3f}` | emphasis `{5}`.".format(
            k_range[0],
            k_range[1],
            _mode_subset_label(focus_mode),
            float(anomaly_threshold),
            float(shift_threshold),
            _color_label(color_by),
        )
    )

    diagnosis = _diagnostic_summary(task_name, env_name, policy, selected_window)
    problem_statement = _user_problem_statement(role_name, task_name, env_name, policy)
    scenario_copy = ROLE_SCENARIO_COPY.get(role_name, "Concrete review scenario for the active user.")
    st.caption("Mission -> Evidence -> Action -> Notebook -> Export -> Task Performance")
    framing_cols = st.columns([1.4, 1.4, 1.6])
    with framing_cols[0]:
        st.markdown("**User problem**")
        st.write(problem_statement)
    with framing_cols[1]:
        st.markdown("**Execution scenario**")
        st.write(scenario_copy)
    with framing_cols[2]:
        st.markdown("**Success condition**")
        st.write(TASK_SUCCESS_COPY.get(task_name, TASK_SUCCESS_COPY["Find anomalies"]))
    summary_cols = st.columns([1.2, 1.2, 1.6])
    with summary_cols[0]:
        st.markdown("**What happened**")
        st.write(diagnosis["happened"])
    with summary_cols[1]:
        st.markdown("**Why it matters**")
        st.write(diagnosis["why"])
    with summary_cols[2]:
        st.markdown("**What to do next**")
        for step in diagnosis["next_steps"]:
            st.markdown("- {0}".format(step))

    default_p1, default_p2 = _recommended_compare_pair(task_name, policy)
    p1 = st.session_state.get("cmp_p1", default_p1)
    if p1 not in BENCHMARK_POLICIES:
        p1 = default_p1
    p2_candidates = [name for name in BENCHMARK_POLICIES if name != p1]
    p2_default = default_p2 if default_p2 in p2_candidates else p2_candidates[0]
    p2 = st.session_state.get("cmp_p2", p2_default)
    if p2 not in p2_candidates:
        p2 = p2_default
    k_clusters = int(st.session_state.get("cmp_k_clusters", 8))
    compare_df = cached_compare(env_name=env_name, p1=p1, p2=p2, k_clusters=k_clusters, seed=selected_seed)
    compare_df = compare_df[compare_df["js_div"].notna()].copy()
    compare_row = None
    if not compare_df.empty:
        compare_clusters = compare_df["cluster"].astype(int).tolist()
        if st.session_state.get("selected_cluster") not in compare_clusters:
            st.session_state["selected_cluster"] = int(compare_clusters[0])
        compare_row = compare_df[compare_df["cluster"] == int(st.session_state["selected_cluster"])].iloc[0]
    _track_task_session_state(
        selected_window_k=int(selected_window["k"]),
        filter_signature={
            "k_range": tuple(k_range),
            "focus_mode": focus_mode,
            "anomaly_threshold": round(float(anomaly_threshold), 6),
            "shift_threshold": round(float(shift_threshold), 6),
            "top_k_windows": int(top_k_windows),
        },
        compare_signature={
            "p1": p1,
            "p2": p2,
            "k_clusters": int(k_clusters),
            "selected_cluster": int(st.session_state.get("selected_cluster")) if st.session_state.get("selected_cluster") is not None else None,
        },
    )
    question_frame = _task_question_rows(
        task_name=task_name,
        role_name=role_name,
        env_name=env_name,
        policy=policy,
        filtered_df=filtered_df,
        window_row=selected_window,
        compare_row=compare_row,
    )

    demo_tabs = st.tabs(["Mission", "Evidence", "Compare", "Action", "Notebook", "Export"])

    with demo_tabs[0]:
        st.markdown("**Purpose.** Start from the operator's question, confirm the current case, and understand how the linked evidence will support the decision.")
        mission_cols = st.columns(4)
        mission_cols[0].info("**Primary user**\n\n{0}".format(role_name))
        mission_cols[1].info("**Decision goal**\n\n{0}".format(_user_task_label(task_name)))
        mission_cols[2].info(
            "**Current case**\n\n{0} / {1} / run {2}".format(
                _format_env_name(env_name),
                _format_policy_name(policy),
                selected_seed if selected_seed is not None else "default",
            )
        )
        mission_cols[3].info("**Focused window**\n\n{0}".format(_window_label(selected_window)))
        scenario_cols = st.columns([1.2, 1.3, 1.5])
        with scenario_cols[0]:
            st.markdown("**User scenario**")
            st.write(scenario_copy)
        with scenario_cols[1]:
            st.markdown("**User problem**")
            st.write(problem_statement)
        with scenario_cols[2]:
            st.markdown("**Working rhythm**")
            st.markdown("- Narrow the windows.")
            st.markdown("- Focus one case.")
            st.markdown("- Check controller differences if needed.")
            st.markdown("- Save a note and export a memo.")
        st.markdown("**Fundamental questions this workspace answers**")
        st.dataframe(question_frame, use_container_width=True, hide_index=True)

    with demo_tabs[1]:
        st.markdown("**Purpose.** Answer the current question by moving from ranked evidence to linked charts to a plain-language explanation.")
        walkthrough = [
            "1. Start from the ranked windows that best answer the active question.",
            "2. Click the behavior map or the timeline to focus one time segment.",
            "3. Read the explanation panel in plain language.",
            "4. Move to Action or Notebook when the focused case is ready.",
        ]
        for line in walkthrough:
            st.markdown(line)
        st.markdown("**Question tracker**")
        st.dataframe(question_frame, use_container_width=True, hide_index=True)
        ranked_df = _window_rank_frame(filtered_df, task_name, top_k_windows)
        st.subheader("Top Time Segments For This Question")
        ranked_cols = st.columns([2.6, 1.2, 1.2, 1, 1])
        ranked_cols[0].markdown("**Time segment**")
        ranked_cols[1].markdown("**Risk flag**")
        ranked_cols[2].markdown("**Change**")
        ranked_cols[3].markdown("**Outcome**")
        ranked_cols[4].markdown("**Focus**")
        for idx, (_, row) in enumerate(ranked_df.iterrows()):
            row_cols = st.columns([2.6, 1.2, 1.2, 1, 1])
            row_cols[0].markdown("`{0}`".format(_window_label(row)))
            row_cols[1].markdown("{0:.3f}".format(float(row["anomaly_score"])))
            row_cols[2].markdown("{0:.3f}".format(float(row["regime_shift_score"])))
            row_cols[3].markdown("{0:.3f}".format(float(row["r_bar"])))
            if row_cols[4].button("Focus", key="focus_window_{0}".format(idx), type="secondary"):
                st.session_state["selected_k"] = int(row["k"])
                _trigger_rerun()

        linked_cols = st.columns([3, 2])
        with linked_cols[0]:
            st.subheader("Behavior Map")
            st.caption("Click any point to update the timeline, case details, action summary, notebook, and exports.")
            behavior_selection = _plotly_chart_with_optional_selection(
                build_behavior_space_figure(filtered_df, env_name=env_name, color_by=color_by),
                key="behavior_space",
                use_container_width=True,
            )
            picked_k = _selected_window(behavior_selection)
            if picked_k is not None:
                st.session_state["selected_k"] = int(picked_k)
                selected_window = filtered_df[filtered_df["k"] == int(picked_k)].iloc[0]
            st.subheader("Timeline")
            temporal_selection = _plotly_chart_with_optional_selection(
                build_temporal_figure(filtered_df, int(st.session_state["selected_k"])),
                key="temporal_view",
                use_container_width=True,
            )
            timeline_k = _selected_window(temporal_selection)
            if timeline_k is not None and timeline_k in filtered_df["k"].astype(int).tolist():
                st.session_state["selected_k"] = int(timeline_k)
                selected_window = filtered_df[filtered_df["k"] == int(timeline_k)].iloc[0]
        with linked_cols[1]:
            st.subheader("Explanation")
            st.caption("This panel turns the focused chart selection into a short explanation for a non-specialist user.")
            window_options = filtered_df["k"].astype(int).tolist()
            selected_now = st.selectbox(
                "Focused time segment",
                options=window_options,
                index=window_options.index(int(st.session_state["selected_k"])) if int(st.session_state["selected_k"]) in window_options else 0,
                key="window_detail_select",
            )
            st.session_state["selected_k"] = int(selected_now)
            selected_window = filtered_df[filtered_df["k"] == int(selected_now)].iloc[0]
            detail_cols = st.columns(2)
            detail_cols[0].metric("Time segment", int(selected_window["k"]))
            detail_cols[1].metric("Span", "{0}-{1}".format(int(selected_window["t_start"]), int(selected_window["t_end"])))
            detail_cols = st.columns(2)
            detail_cols[0].metric("Risk flag", "{0:.3f}".format(float(selected_window["anomaly_score"])))
            detail_cols[1].metric("Change confidence", "{0:.3f}".format(float(selected_window["regime_shift_score"])))
            detail_cols = st.columns(2)
            detail_cols[0].metric("Outcome", "{0:.3f}".format(float(selected_window["r_bar"])))
            detail_cols[1].metric("Action-switch rate", "{0:.3f}".format(float(selected_window["switch_rate"])))
            detail_cols = st.columns(2)
            detail_cols[0].metric("Behavior entropy", "{0:.3f}".format(float(selected_window["H_bar"])))
            detail_cols[1].metric("Decision concentration", "{0:.3f}".format(float(selected_window["dominance_gap"])))
            if "gt_intervention" in selected_window.index:
                st.markdown(
                    "**Reference marker.** intervention=`{0}` | change-start=`{1}`".format(
                        bool(selected_window["gt_intervention"]),
                        bool(selected_window["gt_shift_start"]),
                    )
                )
            st.markdown(_window_explain(filtered_df, policy=policy, env_name=env_name, window_row=selected_window))
            st.caption("The behavior map, ranked list, filters, and explanation panel are coordinated views of the same focused time segment.")

    with demo_tabs[2]:
        st.markdown("**Purpose.** Use this page when the question is not only what happened, but whether another controller behaves differently.")
        cmp_cols = st.columns([1, 1, 1.2, 1])
        with cmp_cols[0]:
            p1 = st.selectbox("Controller 1", options=BENCHMARK_POLICIES, index=_control_default(BENCHMARK_POLICIES, default_p1), key="cmp_p1")
        with cmp_cols[1]:
            p2_candidates = [name for name in BENCHMARK_POLICIES if name != p1]
            p2 = st.selectbox(
                "Controller 2",
                options=p2_candidates,
                index=_control_default(p2_candidates, default_p2 if default_p2 != p1 else p2_candidates[0]),
                key="cmp_p2",
            )
        with cmp_cols[2]:
            k_clusters = st.slider("Behavior groups", min_value=3, max_value=20, value=int(st.session_state.get("cmp_k_clusters", 8)), key="cmp_k_clusters")
        with cmp_cols[3]:
            compare_top_k = st.slider("Top behavior groups", min_value=3, max_value=12, value=8)

        compare_df = cached_compare(env_name=env_name, p1=p1, p2=p2, k_clusters=k_clusters, seed=selected_seed)
        compare_df = compare_df[compare_df["js_div"].notna()].copy()
        top_df = compare_df.head(min(compare_top_k, len(compare_df)))
        if compare_df.empty or top_df.empty:
            st.warning("No valid controller-comparison groups are available for the current pair.")
        else:
            compare_bar = px.bar(
                top_df.assign(behavior_group=top_df["cluster"].astype(int).astype(str)),
                x="behavior_group",
                y="js_div",
                color="reward_gap",
                title="Behavior Difference By Group",
                labels={"behavior_group": "Behavior group", "js_div": "Behavior difference", "reward_gap": "Outcome gap"},
                color_continuous_scale="Tealgrn",
            )
            compare_bar.update_layout(template="plotly_white", coloraxis_colorbar_title="Outcome gap")
            compare_selection = _plotly_chart_with_optional_selection(
                compare_bar,
                key="comparison_bar",
                use_container_width=True,
            )
            selected_cluster = _selected_cluster(compare_selection)
            if selected_cluster is not None:
                st.session_state["selected_cluster"] = int(selected_cluster)
            if st.session_state.get("selected_cluster") not in top_df["cluster"].astype(int).tolist():
                st.session_state["selected_cluster"] = int(top_df.iloc[0]["cluster"])

            compare_cols = st.columns([3, 2])
            with compare_cols[0]:
                cluster_fig = px.scatter(
                    top_df,
                    x="reward_gap",
                    y="js_div",
                    size="n_total",
                    color="cluster",
                    hover_name="cluster",
                    title="Behavior Group Frontier",
                    labels={"reward_gap": "Outcome gap", "js_div": "Behavior difference"},
                )
                cluster_fig.update_layout(template="plotly_white", legend_title="Behavior group")
                st.plotly_chart(cluster_fig, use_container_width=True)
            with compare_cols[1]:
                cluster_options = top_df["cluster"].astype(int).tolist()
                cluster_now = st.selectbox(
                    "Focused behavior group",
                    options=cluster_options,
                    index=cluster_options.index(int(st.session_state["selected_cluster"])) if int(st.session_state["selected_cluster"]) in cluster_options else 0,
                    key="cluster_detail_select",
                )
                st.session_state["selected_cluster"] = int(cluster_now)
                cluster_row = compare_df[compare_df["cluster"] == int(cluster_now)].iloc[0]
                compare_row = cluster_row
                detail_cols = st.columns(2)
                detail_cols[0].metric("Behavior difference", "{0:.3f}".format(float(cluster_row["js_div"])))
                detail_cols[1].metric("Outcome gap", "{0:.3f}".format(float(cluster_row["reward_gap"])))
                detail_cols = st.columns(2)
                detail_cols[0].metric("Visible time segments", int(float(cluster_row["n_total"])))
                detail_cols[1].metric("Behavior group", int(cluster_row["cluster"]))
                st.markdown(_cluster_explain(cluster_row, env_name=env_name, p1=p1, p2=p2, cluster_count=k_clusters))

            top_cluster = _top_cluster_finding(compare_df)
            if top_cluster is not None:
                st.markdown(
                    "The strongest visible separation is behavior group `{0}`: **{1}** vs **{2}** reaches behavior difference `{3:.3f}` and outcome gap `{4:.3f}`.".format(
                        int(top_cluster["cluster"]),
                        _format_policy_name(p1),
                        _format_policy_name(p2),
                        float(top_cluster["js_div"]),
                        float(top_cluster["reward_gap"]),
                    )
                )

    decision = _decision_summary(
        role_name=role_name,
        task_name=task_name,
        env_name=env_name,
        policy=policy,
        window_row=selected_window,
        compare_row=compare_row,
    )

    with demo_tabs[3]:
        st.markdown("**Purpose.** Turn the focused evidence into a clear action a user can take or hand off.")
        decision = _decision_summary(
            role_name=role_name,
            task_name=task_name,
            env_name=env_name,
            policy=policy,
            window_row=selected_window,
            compare_row=compare_row,
        )
        readiness = _decision_readiness(task_name=task_name, filtered_df=filtered_df, window_row=selected_window, compare_row=compare_row)
        st.subheader("Decision Summary")
        top_cols = st.columns(3)
        top_cols[0].metric("Decision level", decision["severity"])
        top_cols[1].metric("Primary owner", decision["owner"])
        top_cols[2].metric("Focused case", "{0} / {1}".format(_format_env_name(env_name), _format_policy_name(policy)))
        st.metric("Decision readiness", readiness["level"], "{0}/{1} checks".format(readiness["score"], readiness["total"]))
        st.progress(float(readiness["score"]) / float(readiness["total"]))
        summary_cols = st.columns([1.4, 1.2])
        with summary_cols[0]:
            st.markdown("**Decision**")
            st.write(decision["decision"])
            st.markdown("**Risk assessment**")
            st.write(decision["risk"])
            st.markdown("**{0}**".format(decision["action_title"]))
            st.write(decision["recommendation"])
        with summary_cols[1]:
            st.markdown("**Evidence used for this decision**")
            for item in decision["evidence"]:
                st.markdown("- {0}".format(item))
            st.markdown("**Readiness checks**")
            for check in readiness["checks"]:
                status = "Ready" if check["ready"] else "Missing"
                st.markdown("- **{0}**: {1}. {2}".format(status, check["label"], check["detail"]))
        st.markdown("**Current answers to the user's core questions**")
        for record in _question_answer_records(question_frame):
            st.markdown("**{0}**".format(record["question"]))
            st.write(record["current_answer"])
            st.caption("Evidence source: {0}".format(record["evidence_source"]))
        st.markdown("**Action checklist**")
        for step in diagnosis["next_steps"]:
            st.markdown("- {0}".format(step))
        support_signal, support_reason = _task_support_signal(task_name, selected_window, compare_row)
        perf_cols = st.columns(4)
        perf_cols[0].metric("Task timer", "{0:.1f}s".format(max(0.0, float(time.time() - float(st.session_state.get("task_session_started_at", time.time()))))))
        perf_cols[1].metric("Exports used", int(st.session_state.get("task_export_count", 0)))
        perf_cols[2].metric("Notebook saves", int(st.session_state.get("task_notebook_saves", 0)))
        perf_cols[3].metric("Support signal", "Yes" if support_signal else "No")
        st.caption("Support reason: {0}".format(support_reason))
        st.session_state.setdefault("task_reviewer_name", "")
        st.session_state.setdefault("task_user_confidence", 4.0)
        with st.form("task_completion_form", clear_on_submit=False):
            completion_cols = st.columns([1.2, 1.0])
            with completion_cols[0]:
                reviewer_name = st.text_input(
                    "Operator / reviewer",
                    key="task_reviewer_name",
                    placeholder="Name or team completing this review",
                )
            with completion_cols[1]:
                user_confidence = st.slider(
                    "Confidence",
                    min_value=1.0,
                    max_value=5.0,
                    value=float(st.session_state.get("task_user_confidence", 4.0)),
                    step=0.5,
                    key="task_user_confidence",
                )
            log_task_completion = st.form_submit_button("Log completed user task", use_container_width=True)
        if log_task_completion:
            task_entry = _compose_task_log_entry(
                workflow_name=workflow_name,
                role_name=role_name,
                task_name=task_name,
                env_name=env_name,
                policy=policy,
                selected_seed=selected_seed,
                window_row=selected_window,
                decision=decision,
                question_frame=question_frame,
                compare_row=compare_row,
                user_confidence=float(user_confidence),
                reviewer_name=reviewer_name,
            )
            _save_user_task_entry(task_entry)
            st.success("Logged the completed task for course task-performance analysis.")
            _reset_task_session(case_signature)
        st.caption("The current focused case is shared with the Evidence, Compare, Notebook, and Export tabs.")

    with demo_tabs[4]:
        st.markdown("**Purpose.** Save annotations, mark case status, and build a reusable notebook of review cases.")
        with st.form("case_note_form", clear_on_submit=False):
            note_cols = st.columns([1.3, 1.0, 1.0])
            with note_cols[0]:
                note_title = st.text_input(
                    "Case title",
                    key="case_note_title",
                    placeholder="Short title for this case",
                )
            with note_cols[1]:
                note_status = st.selectbox(
                    "Case status",
                    options=USER_CASE_STATUS_OPTIONS,
                    key="case_note_status",
                )
            with note_cols[2]:
                note_owner = st.text_input("Owner", key="case_note_owner")
            note_tags = st.multiselect(
                "Tags",
                options=USER_NOTE_TAG_OPTIONS,
                key="case_note_tags",
            )
            note_body = st.text_area(
                "Analyst note",
                key="case_note_body",
                height=180,
                placeholder="Explain what you saw, why it matters, and what should happen next.",
            )
            save_case = st.form_submit_button("Save case to notebook", use_container_width=True)
        if save_case:
            entry = _compose_case_note_entry(
                workflow_name=workflow_name,
                role_name=role_name,
                task_name=task_name,
                env_name=env_name,
                policy=policy,
                selected_seed=selected_seed,
                window_row=selected_window,
                decision=decision,
                note_title=note_title,
                note_body=note_body,
                note_status=note_status,
                note_owner=note_owner,
                note_tags=note_tags,
                compare_row=compare_row,
                question_frame=question_frame,
            )
            _save_case_notebook_entry(entry)
            st.session_state["task_notebook_saves"] = int(st.session_state.get("task_notebook_saves", 0)) + 1
            st.success("Saved the current case to the notebook.")

        notebook_frame = _case_notebook_frame()
        if notebook_frame.empty:
            st.info("No saved cases yet. Save the current case to start the notebook.")
        else:
            filter_cols = st.columns([1.0, 1.0, 2.2])
            with filter_cols[0]:
                status_filter = st.multiselect(
                    "Filter by status",
                    options=sorted(notebook_frame["status"].dropna().unique().tolist()),
                    default=sorted(notebook_frame["status"].dropna().unique().tolist()),
                )
            with filter_cols[1]:
                role_filter = st.multiselect(
                    "Filter by role",
                    options=sorted(notebook_frame["role"].dropna().unique().tolist()),
                    default=sorted(notebook_frame["role"].dropna().unique().tolist()),
                )
            visible_notebook = notebook_frame.copy()
            if status_filter:
                visible_notebook = visible_notebook[visible_notebook["status"].isin(status_filter)]
            if role_filter:
                visible_notebook = visible_notebook[visible_notebook["role"].isin(role_filter)]
            st.dataframe(_case_note_export_frame(visible_notebook), use_container_width=True, hide_index=True)
            if not visible_notebook.empty:
                case_options = visible_notebook.apply(lambda row: _entry_summary_line(row.to_dict()), axis=1).tolist()
                selected_case = st.selectbox("Saved case detail", options=case_options)
                case_idx = case_options.index(selected_case)
                case_entry = visible_notebook.iloc[case_idx].to_dict()
                st.markdown(_build_case_note_markdown(case_entry))
                saved_case_downloaded = st.download_button(
                    "Download selected saved case",
                    data=_build_case_note_markdown(case_entry).encode("utf-8"),
                    file_name="saved_case_note.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                if saved_case_downloaded:
                    _register_export_event()

    with demo_tabs[5]:
        st.markdown("**Purpose.** Export work products a user can hand to a reviewer, incident owner, or operations lead.")
        decision_md = "\n".join(
            [
                "# RLVA Decision Summary",
                "",
                "- User role: {0}".format(role_name),
                "- User task: {0}".format(_user_task_label(task_name)),
                "- System: {0}".format(_format_env_name(env_name)),
                "- Controller: {0}".format(_format_policy_name(policy)),
                "- Decision level: {0}".format(decision["severity"]),
                "- Primary owner: {0}".format(decision["owner"]),
                "",
                "## Decision",
                "",
                decision["decision"],
                "",
                "## Risk assessment",
                "",
                decision["risk"],
                "",
                "## Recommended next action",
                "",
                decision["recommendation"],
                "",
                "## Evidence",
                "",
            ]
            + ["- {0}".format(item) for item in decision["evidence"]]
            + [""]
        )
        review_memo_md = _build_review_memo_markdown(
            role_name=role_name,
            workflow_name=workflow_name,
            env_name=env_name,
            policy=policy,
            selected_seed=selected_seed,
            task_name=task_name,
            window_row=selected_window,
            decision=decision,
            note_title=st.session_state.get("case_note_title", ""),
            note_body=st.session_state.get("case_note_body", ""),
            question_frame=question_frame,
        )
        task_answer_md = _build_task_answer_brief_markdown(
            workflow_name=workflow_name,
            role_name=role_name,
            task_name=task_name,
            env_name=env_name,
            policy=policy,
            selected_seed=selected_seed,
            window_row=selected_window,
            decision=decision,
            question_frame=question_frame,
        )
        window_md = _build_window_evidence_markdown(env_name=env_name, policy=policy, task_name=task_name, window_row=selected_window)
        export_cols = st.columns(3)
        decision_downloaded = export_cols[0].download_button(
            "Download decision summary",
            data=decision_md.encode("utf-8"),
            file_name="{0}_{1}_decision_summary.md".format(env_name, policy),
            mime="text/markdown",
            use_container_width=True,
        )
        if decision_downloaded:
            _register_export_event()
        review_memo_downloaded = export_cols[1].download_button(
            "Download review memo",
            data=review_memo_md.encode("utf-8"),
            file_name="{0}_{1}_review_memo.md".format(env_name, policy),
            mime="text/markdown",
            use_container_width=True,
        )
        if review_memo_downloaded:
            _register_export_event()
        answer_brief_downloaded = export_cols[2].download_button(
            "Download workflow answer brief",
            data=task_answer_md.encode("utf-8"),
            file_name="{0}_{1}_workflow_answer_brief.md".format(env_name, policy),
            mime="text/markdown",
            use_container_width=True,
        )
        if answer_brief_downloaded:
            _register_export_event()
        window_md = _build_window_evidence_markdown(env_name=env_name, policy=policy, task_name=task_name, window_row=selected_window)
        window_evidence_downloaded = st.download_button(
            "Download focused time-segment evidence",
            data=window_md.encode("utf-8"),
            file_name="{0}_{1}_window_k{2}_evidence.md".format(env_name, policy, int(selected_window["k"])),
            mime="text/markdown",
            use_container_width=True,
        )
        if window_evidence_downloaded:
            _register_export_event()
        filtered_csv_downloaded = st.download_button(
            "Download filtered time segments CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="{0}_{1}_{2}_filtered_windows.csv".format(env_name, policy, task_name.replace(" ", "_").lower()),
            mime="text/csv",
            use_container_width=True,
        )
        if filtered_csv_downloaded:
            _register_export_event()
        if not compare_df.empty and st.session_state.get("selected_cluster") in compare_df["cluster"].astype(int).tolist():
            cluster_row = compare_df[compare_df["cluster"] == int(st.session_state["selected_cluster"])].iloc[0]
            comparison_brief_md = _build_comparison_brief_markdown(
                env_name=env_name,
                p1=p1,
                p2=p2,
                cluster_count=k_clusters,
                cluster_row=cluster_row,
            )
            cluster_md = _build_cluster_evidence_markdown(env_name=env_name, p1=p1, p2=p2, cluster_count=k_clusters, cluster_row=cluster_row)
            compare_export_cols = st.columns(2)
            comparison_brief_downloaded = compare_export_cols[0].download_button(
                "Download controller comparison brief",
                data=comparison_brief_md.encode("utf-8"),
                file_name="{0}_{1}_vs_{2}_comparison_brief.md".format(env_name, p1, p2),
                mime="text/markdown",
                use_container_width=True,
            )
            if comparison_brief_downloaded:
                _register_export_event()
            cluster_evidence_downloaded = compare_export_cols[1].download_button(
                "Download focused behavior-group evidence",
                data=cluster_md.encode("utf-8"),
                file_name="{0}_{1}_vs_{2}_cluster_{3}.md".format(env_name, p1, p2, int(cluster_row["cluster"])),
                mime="text/markdown",
                use_container_width=True,
            )
            if cluster_evidence_downloaded:
                _register_export_event()
            comparison_csv_downloaded = st.download_button(
                "Download controller-comparison CSV",
                data=compare_df.to_csv(index=False).encode("utf-8"),
                file_name="{0}_{1}_vs_{2}_clusters.csv".format(env_name, p1, p2),
                mime="text/csv",
                use_container_width=True,
            )
            if comparison_csv_downloaded:
                _register_export_event()
        notebook_frame = _case_notebook_frame()
        if not notebook_frame.empty:
            notebook_export_cols = st.columns(2)
            notebook_digest_downloaded = notebook_export_cols[0].download_button(
                "Download notebook digest",
                data=_build_notebook_digest_markdown(notebook_frame).encode("utf-8"),
                file_name="case_notebook_digest.md",
                mime="text/markdown",
                use_container_width=True,
            )
            if notebook_digest_downloaded:
                _register_export_event()
            notebook_csv_downloaded = notebook_export_cols[1].download_button(
                "Download notebook CSV",
                data=_case_note_export_frame(notebook_frame).to_csv(index=False).encode("utf-8"),
                file_name="case_notebook.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if notebook_csv_downloaded:
                _register_export_event()


def _render_validation_panel() -> None:
    _render_active_dataset_banner()
    benchmark_df = cached_report_csv("benchmark_table.csv")
    ablation_df = cached_report_csv("ablation_table.csv")
    detection_df = cached_report_csv("detection_metrics.csv")
    baseline_df = cached_report_csv("baseline_detector_table.csv")
    sensitivity_df = cached_report_csv("budget_sensitivity_table.csv")
    seed_df = cached_report_csv("benchmark_seed_metrics.csv")
    robustness_cluster_df = cached_report_csv("robustness_cluster_table.csv")
    robustness_window_df = cached_report_csv("robustness_window_table.csv")
    coverage_df = cached_artifact_coverage()
    case_index_df = _case_study_index()
    timing_df = cached_interaction_timings()
    course_interactivity = cached_course_interactivity_metrics()
    submission_inventory = cached_submission_package_inventory()

    if benchmark_df.empty:
        st.warning("Benchmark reports are not available yet. Run the benchmark pipeline first.")
        return

    controls = _global_controls(benchmark_df)
    asset_catalog = cached_asset_catalog()
    main_story_df = _main_story_rows(benchmark_df)
    supplement_df = _supplement_rows(benchmark_df)
    lunarlander_benchmark = benchmark_df[benchmark_df["env"] == "lunarlander"].copy()

    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">Project Showcase</div>
        <div class="hero-title">Why Trust This System</div>
        <p class="hero-copy">
            This workspace explains, in user language, why the diagnostic system is worth trusting: whether it
            catches unusual behavior, whether it catches operating changes, and whether it stays reliable and
            responsive during real use.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    headline_cols = st.columns(4)
    headline_cols[0].metric("Main environments", 3, "inventory / traffic / lunarlander")
    headline_cols[1].metric("Learned policies", 3, "pg / ppo / dqn")
    headline_cols[2].metric("Exported figures", len(list(_artifact_fig_dir().glob("*.png"))))
    headline_cols[3].metric("Report tables", len(list(BENCHMARK_REPORT_DIR.glob("*.csv"))))
    st.caption(_task_focus_hint(controls["task_focus"], controls["mode"]))

    tabs = st.tabs(
        [
            "Does it catch unusual behavior?",
            "Does it catch operating changes?",
            "Does it stay reliable and responsive?",
        ]
    )

    with tabs[0]:
        st.markdown("**Purpose.** This page answers the first trust question: can the system reliably flag unusual behavior when it matters?")
        st.markdown('<div class="section-chip">Unusual Behavior</div>', unsafe_allow_html=True)
        st.subheader("How The System Catches Unusual Behavior")
        topic = st.selectbox("Context for this question", options=list(OVERVIEW_TOPICS.keys()))
        st.info(OVERVIEW_TOPICS[topic])
        dataset_meta = cached_course_dataset_metadata()
        dataset_sample = cached_course_dataset_sample(8)
        st.markdown(
            "The system looks for unusual short-term behavior inside long traces. Instead of trusting only a final outcome score, it breaks the run into time segments and summarizes what changed inside each one."
        )
        overview_cols = st.columns(2)
        with overview_cols[0]:
            st.markdown(
                """
                **What This Evidence Is Based On**

                - Data source: trace records exported from five benchmark systems: `queue`, `inventory`, `traffic`, `cartpole`, and `lunarlander`.
                - Controllers: learned controllers (`pg`, `ppo`, `dqn`) plus simple reference behaviors (`random`, `heuristic`).
                - Reference labels: benchmark summaries include known intervention periods, so we can check whether the method catches suspicious behavior in the right places.
                - Main user question here: can the system point to behavior that really looks unusual?
                """
            )
        with overview_cols[1]:
            st.markdown(
                """
                **What The User Gets**

                - A ranked view of suspicious time segments instead of one opaque final score.
                - Linked views that let the user inspect behavior, time, and explanation together.
                - A clear sign of whether unusual behavior is being detected in the systems where it matters most.
                """
            )
        if dataset_meta:
            st.markdown('<div class="section-chip">Dataset Layer</div>', unsafe_allow_html=True)
            dataset_cols = st.columns(4)
            dataset_cols[0].metric("Persistent raw dataset", _format_bytes(int(dataset_meta.get("actual_bytes", 0))))
            dataset_cols[1].metric("Raw records", int(dataset_meta.get("total_records", 0)))
            dataset_cols[2].metric("Bytes / record", "{0:.3f}".format(float(dataset_meta.get("bytes_per_record", 0.0))))
            dataset_cols[3].metric("Source trace files", int(dataset_meta.get("source_trace_files", 0)))
            overview_cols = st.columns([2, 3])
            with overview_cols[0]:
                inventory_df = _course_dataset_inventory_frame()
                if not inventory_df.empty:
                    st.plotly_chart(_build_dataset_inventory_figure(inventory_df), use_container_width=True)
            with overview_cols[1]:
                if not dataset_sample.empty:
                    st.dataframe(dataset_sample, use_container_width=True, hide_index=True)
            st.caption("The final project is anchored in the stored 1GB raw trace corpus; the interface stays usable by working from derived summaries instead of re-reading the entire dataset every time.")

    with tabs[1]:
        st.markdown("**Purpose.** This page answers the second trust question: can the system catch meaningful operating changes, not just isolated odd moments?")
        st.markdown('<div class="section-chip">Operating Changes</div>', unsafe_allow_html=True)
        st.subheader("How The System Catches Operating Changes")
        active_env = controls["env"] if controls["env"] != "all" else "inventory"
        active_policy = controls["policy"] if controls["policy"] != "all" else "dqn"
        active_metric = controls["metric"]
        active_env_df = benchmark_df[benchmark_df["env"] == active_env].copy()
        active_row = benchmark_df[(benchmark_df["env"] == active_env) & (benchmark_df["policy"] == active_policy)].copy()
        if active_row.empty and not active_env_df.empty:
            active_row = active_env_df.sort_values("{0}_mean".format(active_metric), ascending=_metric_ascending(active_metric)).head(1)
            active_policy = str(active_row.iloc[0]["policy"])
        lunarlander_dqn = benchmark_df[(benchmark_df["env"] == "lunarlander") & (benchmark_df["policy"] == "dqn")]
        inventory_anomaly = _best_row(benchmark_df, env_name="inventory", metric="anomaly_auc")
        traffic_shift = _best_row(benchmark_df, env_name="traffic", metric="shift_auc")
        presentation_focus = st.selectbox("What operating-change evidence to emphasize", options=["Overall story", "Anomaly detection", "Shift localization", "Method comparison"])

        st.markdown(
            """
        <div class="hero-shell">
            <div class="hero-kicker">Trust Layer</div>
            <div class="hero-title">Does It Catch Operating Changes?</div>
            <p class="hero-copy">
                This evidence focuses on larger state changes. It shows whether the system can notice when behavior
                moves into a new operating mode, and whether that signal stays visible across multiple environments.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        highlight_cols = st.columns(3)
        with highlight_cols[0]:
            if not lunarlander_dqn.empty:
                lunarlander_row = lunarlander_dqn.iloc[0]
                st.metric("LunarLander DQN unusual-behavior score", f"{float(lunarlander_row['anomaly_auc_mean']):.3f}")
                st.caption("Strongest standard-control example in the current benchmark.")
        with highlight_cols[1]:
            if inventory_anomaly is not None:
                st.metric(
                    "Inventory best unusual-behavior score",
                    f"{float(inventory_anomaly['anomaly_auc_mean']):.3f}",
                    _format_policy_name(str(inventory_anomaly["policy"])),
                )
                st.caption("Operational setting with the most stable signal for unusual behavior.")
        with highlight_cols[2]:
            if traffic_shift is not None:
                st.metric(
                    "Traffic best operating-change score",
                    f"{float(traffic_shift['shift_auc_mean']):.3f}",
                    _format_policy_name(str(traffic_shift["policy"])),
                )
                st.caption("Operational setting with the clearest operating-change signal.")

        st.markdown("Traffic is the clearest operating-change story, inventory helps show the contrast with unusual-behavior detection, and LunarLander shows the idea on a familiar control task.")
        if presentation_focus == "Anomaly detection":
            st.info("Current emphasis: use inventory first if you want to contrast unusual behavior with operating change before moving into the shift story.")
        elif presentation_focus == "Shift localization":
            st.info("Current emphasis: foreground traffic and the change-sensitive plots, then use robustness views to show the change signal is not fragile.")
        elif presentation_focus == "Method comparison":
            st.info("Current emphasis: use simpler-method comparison and example cases to show that RLVA adds value beyond basic checks.")
        else:
            st.info("Current emphasis: give a balanced summary of where the system catches operating changes and why the evidence is believable.")
        st.plotly_chart(_build_main_story_figure(main_story_df), use_container_width=True)
        if not active_row.empty:
            row = active_row.iloc[0]
            st.markdown(
                (
                    "**Current takeaway.** Under the active filters, **{env} / {policy}** has "
                    "`{metric}` = `{score:.3f}` with anomaly AUC `{anomaly:.3f}` and shift AUC `{shift:.3f}`."
                ).format(
                    env=_format_env_name(str(row["env"])),
                    policy=_format_policy_name(str(row["policy"])),
                    metric=_metric_label(active_metric),
                    score=float(row["{0}_mean".format(active_metric)]),
                    anomaly=float(row["anomaly_auc_mean"]),
                    shift=float(row["shift_auc_mean"]),
                )
            )
        st.markdown("**What we learned:** the method is strongest where interventions produce clear behavior changes, and short behavior summaries expose those changes earlier and more clearly than a final outcome score alone.")

        story_cols = st.columns([3, 2])
        with story_cols[0]:
            if not lunarlander_benchmark.empty:
                st.plotly_chart(_build_lunarlander_focus_figure(lunarlander_benchmark), use_container_width=True)
        with story_cols[1]:
            st.plotly_chart(_build_transfer_figure(supplement_df), use_container_width=True)
        st.caption("These two views complement the main benchmark: LunarLander is the strongest standard-control example, while CartPole acts as supporting evidence rather than the headline result.")

        lunarlander_policy_options = sorted(
            ablation_df[ablation_df["env"] == "lunarlander"]["policy"].unique().tolist()
        ) if not ablation_df.empty else []
        if lunarlander_policy_options:
            selected_lunarlander_policy = st.selectbox(
                "LunarLander feature-importance policy",
                options=lunarlander_policy_options,
                index=_control_default(lunarlander_policy_options, controls["policy"]),
                format_func=_format_policy_name,
            )
            lunarlander_ablation = ablation_df[
                (ablation_df["env"] == "lunarlander") & (ablation_df["policy"] == selected_lunarlander_policy)
            ].copy()
        else:
            selected_lunarlander_policy = "dqn"
            lunarlander_ablation = pd.DataFrame()

        if not lunarlander_ablation.empty:
            st.subheader(
                "What Information Matters: LunarLander / {0}".format(
                    _format_policy_name(selected_lunarlander_policy)
                )
            )
            action_only = lunarlander_ablation[lunarlander_ablation["feature_group"] == "action_only"]
            full_only = lunarlander_ablation[lunarlander_ablation["feature_group"] == "full"]
            if not action_only.empty and not full_only.empty:
                action_row = action_only.iloc[0]
                full_row = full_only.iloc[0]
                st.markdown(
                    "Action-only information reaches **{0}** unusual-behavior score and **{1}** operating-change score; full features reach **{2}** and **{3}**. "
                    "This suggests the environment already exposes a substantial signal through action distribution and switching behavior.".format(
                        str(action_row["anomaly_auc_summary"]),
                        str(action_row["shift_auc_summary"]),
                        str(full_row["anomaly_auc_summary"]),
                        str(full_row["shift_auc_summary"]),
                    )
                )
            st.plotly_chart(_build_ablation_profile_figure(lunarlander_ablation), use_container_width=True)
            st.caption("This view is interactive: use the controller selector above to compare which kinds of behavior information matter for different LunarLander controllers.")

        if not baseline_df.empty:
            st.markdown('<div class="section-chip">Method Comparison</div>', unsafe_allow_html=True)
            st.subheader("RLVA Vs Simpler Checks")
            baseline_controls = st.columns(3)
            with baseline_controls[0]:
                baseline_env_options = sorted(baseline_df["env"].unique().tolist())
                baseline_env = st.selectbox(
                    "System for comparison",
                    options=baseline_env_options,
                    index=_control_default(baseline_env_options, controls["env"]),
                    format_func=lambda name: ENV_LABELS.get(name, name),
                )
            with baseline_controls[1]:
                env_policies = sorted(baseline_df[baseline_df["env"] == baseline_env]["policy"].unique().tolist())
                baseline_policy = st.selectbox("Controller for comparison", options=env_policies, index=_control_default(env_policies, controls["policy"]), format_func=_format_policy_name)
            with baseline_controls[2]:
                baseline_metric = st.selectbox(
                    "Comparison score",
                    options=["anomaly_auc", "shift_auc", "shift_localization_error"],
                    index=_control_default(["anomaly_auc", "shift_auc", "shift_localization_error"], controls["metric"] if controls["metric"] in ["anomaly_auc", "shift_auc", "shift_localization_error"] else "shift_auc"),
                    format_func=_metric_label,
                )
            baseline_extra = st.columns(2)
            with baseline_extra[0]:
                normalize_baseline = st.checkbox("Normalize relative to RLVA", value=False)
            with baseline_extra[1]:
                only_beating = st.checkbox("Show only methods that beat RLVA", value=False)
            st.plotly_chart(
                _build_baseline_comparison_figure(
                    benchmark_df=benchmark_df,
                    baseline_df=baseline_df,
                    env_name=baseline_env,
                    policy_name=baseline_policy,
                    metric=baseline_metric,
                ),
                use_container_width=True,
            )
            if baseline_metric in LOWER_IS_BETTER_METRICS:
                st.caption("For localization error, smaller values are better, so the chart is sorted accordingly.")
            else:
                st.caption("This comparison matters because it shows whether the project method adds value beyond simpler checks.")
            baseline_table = _baseline_comparison_frame(benchmark_df, baseline_df, baseline_env, baseline_policy, baseline_metric, normalize_baseline)
            if only_beating and not baseline_table.empty:
                baseline_table = baseline_table[baseline_table["method"].eq("RLVA") | baseline_table["beats_rlva"]]
            if not baseline_table.empty:
                baseline_plot = baseline_table.copy()
                baseline_plot = baseline_plot.sort_values("display_score", ascending=_metric_ascending(baseline_metric))
                st.plotly_chart(
                    _build_ranked_bar(
                        baseline_plot,
                        x="method",
                        y="display_score",
                        color="beats_rlva",
                        title="Simpler-Method Drill-Down",
                    ),
                    use_container_width=True,
                )
                _render_metric_cards_from_frame(
                    baseline_plot.sort_values("display_score", ascending=_metric_ascending(baseline_metric)),
                    label_col="method",
                    value_col="display_score",
                    max_items=min(3, len(baseline_plot)),
                )

        if not sensitivity_df.empty:
            st.markdown('<div class="section-chip">Robustness</div>', unsafe_allow_html=True)
            st.subheader("Sensitivity To Analysis Budget")
            sens_controls = st.columns(5)
            with sens_controls[0]:
                sens_env_options = sorted(sensitivity_df["env"].unique().tolist())
                sens_env = st.selectbox(
                    "System",
                    options=sens_env_options,
                    index=_control_default(sens_env_options, controls["env"]),
                    format_func=lambda name: ENV_LABELS.get(name, name),
                )
            with sens_controls[1]:
                sens_policy_options = sorted(sensitivity_df[sensitivity_df["env"] == sens_env]["policy"].unique().tolist())
                sens_policy = st.selectbox(
                    "Controller",
                    options=sens_policy_options,
                    index=_control_default(sens_policy_options, controls["policy"]),
                    format_func=_format_policy_name,
                )
            with sens_controls[2]:
                sens_metric = st.selectbox(
                    "Score to inspect",
                    options=["anomaly_auc", "shift_auc", "shift_localization_error"],
                    index=_control_default(["anomaly_auc", "shift_auc", "shift_localization_error"], controls["metric"] if controls["metric"] in ["anomaly_auc", "shift_auc", "shift_localization_error"] else "shift_auc"),
                    format_func=_metric_label,
                )
            with sens_controls[3]:
                fixed_l = st.selectbox("Fixed segment length", options=sorted(sensitivity_df["L"].unique().tolist()), index=1 if len(sensitivity_df["L"].unique()) > 1 else 0)
            with sens_controls[4]:
                fixed_seed_count = st.selectbox("Number of repeated runs", options=sorted(sensitivity_df["seed_count"].unique().tolist()), index=len(sensitivity_df["seed_count"].unique().tolist()) - 1)
            sens_cols = st.columns(2)
            with sens_cols[0]:
                st.plotly_chart(
                    _build_sensitivity_figure(
                        sensitivity_df=sensitivity_df,
                        env_name=sens_env,
                        policy_name=sens_policy,
                        metric=sens_metric,
                        fixed_l=int(fixed_l),
                        fixed_seed_count=int(fixed_seed_count),
                    ),
                    use_container_width=True,
                )
            with sens_cols[1]:
                fixed_t_options = sorted(sensitivity_df["T"].unique().tolist())
                fixed_t = 600 if 600 in fixed_t_options else fixed_t_options[-1]
                st.plotly_chart(
                    _build_window_sensitivity_figure(
                        sensitivity_df=sensitivity_df,
                        env_name=sens_env,
                        policy_name=sens_policy,
                        metric=sens_metric,
                        fixed_t=int(fixed_t),
                        fixed_seed_count=int(fixed_seed_count),
                    ),
                    use_container_width=True,
                )
            st.caption("Some `nan` regions mean the shortened trace never enters a labeled intervention period. Read these as insufficient reference coverage, not as a method failure.")
            detail_slice = sensitivity_df[
                (sensitivity_df["env"] == sens_env)
                & (sensitivity_df["policy"] == sens_policy)
                & (sensitivity_df["seed_count"] == fixed_seed_count)
            ].copy()
            if not detail_slice.empty:
                best_setting = detail_slice.sort_values("{0}_mean".format(sens_metric), ascending=_metric_ascending(sens_metric)).iloc[0]
                st.markdown(
                    "Best visible setting under the current drill-down: `T={0}`, `L={1}`, `{2}={3:.3f}`.".format(
                        int(best_setting["T"]),
                        int(best_setting["L"]),
                        _metric_label(sens_metric),
                        float(best_setting["{0}_mean".format(sens_metric)]),
                    )
                )

        if not case_index_df.empty:
            st.markdown('<div class="section-chip">Qualitative Evidence</div>', unsafe_allow_html=True)
            st.subheader("Example Browser")
            st.caption("Exported example figures with linked source artifacts.")
            case_controls = st.columns(3)
            with case_controls[0]:
                case_env_options = sorted(case_index_df["env"].unique().tolist())
                case_env = st.selectbox(
                    "System",
                    options=case_env_options,
                    index=_control_default(case_env_options, controls["env"]),
                    format_func=lambda name: ENV_LABELS.get(name, name),
                )
            with case_controls[1]:
                case_policy_options = sorted(case_index_df[case_index_df["env"] == case_env]["policy"].unique().tolist())
                case_policy = st.selectbox(
                    "Controller",
                    options=case_policy_options,
                    index=_control_default(case_policy_options, controls["policy"]),
                    format_func=_format_policy_name,
                )
            with case_controls[2]:
                case_rows = case_index_df[(case_index_df["env"] == case_env) & (case_index_df["policy"] == case_policy)].copy()
                case_seed = st.selectbox("Scenario run", options=case_rows["seed"].astype(int).tolist())

            selected_case = case_rows[case_rows["seed"].astype(int) == int(case_seed)]
            case_image = _figure_image_path("{0}_{1}_case_study".format(case_env, case_policy))
            if case_image is not None:
                caption = "{0} / {1} / exported seed {2}".format(
                    ENV_LABELS.get(case_env, case_env),
                    _format_policy_name(case_policy),
                    case_seed,
                )
                st.image(str(case_image), caption=caption)
                if not selected_case.empty:
                    case_record = selected_case.iloc[0]
                    st.caption(
                        "Trace: `{0}` | Summary: `{1}`".format(
                            Path(str(case_record["trace_path"])).name,
                            Path(str(case_record["summary_path"])).name,
                        )
                    )
                    note = _case_study_note(case_env, case_policy, int(case_seed))
                    st.markdown("**Why this case matters.** {0} {1}".format(note["headline"], note["why"]))
                    linked_case_metrics = benchmark_df[
                        (benchmark_df["env"] == case_env) & (benchmark_df["policy"] == case_policy)
                    ]
                    if not linked_case_metrics.empty:
                        case_metrics = linked_case_metrics.iloc[0]
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("Anomaly AUC", "{0:.3f}".format(float(case_metrics["anomaly_auc_mean"])))
                        metric_cols[1].metric("Shift AUC", "{0:.3f}".format(float(case_metrics["shift_auc_mean"])))
                        metric_cols[2].metric("Localization error", "{0:.3f}".format(float(case_metrics["shift_localization_error_mean"])))

    with tabs[2]:
        st.markdown("**Purpose.** This page answers the third trust question: does the system stay reliable and responsive when a real user interacts with it?")
        st.markdown('<div class="section-chip">Reliability And Responsiveness</div>', unsafe_allow_html=True)
        st.subheader("Does The System Stay Reliable And Responsive?")
        analysis_focus = st.selectbox(
            "What trust evidence to inspect",
            options=["Reliability overview", "Repeated-run stability", "Detection snapshot", "Information importance", "Robustness"],
        )
        st.caption("Use this selector to inspect different kinds of trust evidence.")
        explorer_cols = st.columns(2)
        with explorer_cols[0]:
            metric = st.selectbox("Summary score", options=["anomaly_auc", "shift_auc", "reward_mean"], key="benchmark_metric", index=["anomaly_auc", "shift_auc", "reward_mean"].index(controls["metric"]) if controls["metric"] in ["anomaly_auc", "shift_auc", "reward_mean"] else 0)
            bench_plot_df = _filter_like_global(benchmark_df, controls)
            if bench_plot_df.empty:
                bench_plot_df = benchmark_df.copy()
            st.plotly_chart(build_benchmark_overview_figure(bench_plot_df, metric=metric), use_container_width=True)
        with explorer_cols[1]:
            env_metric_df = _filter_like_global(benchmark_df, controls)
            if env_metric_df.empty:
                env_metric_df = benchmark_df.copy()
            env_metric_df["env_label"] = env_metric_df["env"].map(lambda name: ENV_LABELS.get(name, name))
            env_metric_df["policy_label"] = env_metric_df["policy"].map(_format_policy_name)
            spread_fig = px.scatter(
                env_metric_df,
                x="anomaly_auc_mean",
                y="shift_auc_mean",
                color="env_label",
                symbol="policy_label",
                size="reward_jump_auc_mean",
                hover_name="policy_label",
                title="Controller Frontier",
                labels={"anomaly_auc_mean": "Anomaly AUC", "shift_auc_mean": "Shift AUC"},
            )
            spread_fig.update_traces(customdata=env_metric_df[["env", "policy"]].to_numpy(), hovertemplate="%{hovertext}<br>Anomaly=%{x:.3f}<br>Shift=%{y:.3f}<extra></extra>")
            spread_fig.update_layout(template="plotly_white", legend_title="")
            spread_selection = _plotly_chart_with_optional_selection(
                spread_fig,
                use_container_width=True,
            )
            spread_points = _extract_points(spread_selection)
            if spread_points:
                custom = spread_points[0].get("customdata") if isinstance(spread_points[0], dict) else None
                if isinstance(custom, (list, tuple)) and len(custom) >= 2:
                    _queue_global_update(env_name=str(custom[0]), policy=str(custom[1]))
                    _trigger_rerun()
        st.subheader("Ranking View")
        benchmark_table = _benchmark_table_frame(benchmark_df, controls)
        if not benchmark_table.empty:
            ranking_plot = benchmark_table.copy()
            ranking_plot["label"] = ranking_plot["environment"] + " / " + ranking_plot["policy_label"]
            ranking_fig = _build_ranked_bar(
                ranking_plot,
                x="selected_metric",
                y="label",
                title="Ranking View",
                color="environment",
                orientation="h",
            )
            ranking_fig.update_traces(customdata=ranking_plot[["environment", "policy_label"]].to_numpy(), hovertemplate="%{y}<br>score=%{x:.3f}<extra></extra>")
            ranking_selection = _plotly_chart_with_optional_selection(
                ranking_fig,
                use_container_width=True,
            )
            ranking_points = _extract_points(ranking_selection)
            if ranking_points:
                custom = ranking_points[0].get("customdata") if isinstance(ranking_points[0], dict) else None
                if isinstance(custom, (list, tuple)) and len(custom) >= 2:
                    env_label = str(custom[0])
                    policy_label = str(custom[1])
                    env_match = next((name for name, label in ENV_LABELS.items() if label == env_label), None)
                    policy_match = next((name for name in BENCHMARK_POLICIES if _format_policy_name(name) == policy_label), None)
                    if env_match or policy_match:
                        _queue_global_update(env_name=env_match or controls["env"], policy=policy_match or controls["policy"], metric=metric)
                        _trigger_rerun()
            best_table_row = benchmark_table.iloc[0]
            st.markdown(
                "Top visible row under the current filters: **{0} / {1}** with selected score `{2:.3f}`.".format(
                    best_table_row["environment"],
                    best_table_row["policy_label"],
                    float(best_table_row["selected_metric"]),
                )
            )
            _render_metric_cards_from_frame(
                ranking_plot.rename(columns={"label": "rank_label"}),
                label_col="rank_label",
                value_col="selected_metric",
                max_items=min(3, len(ranking_plot)),
            )

        if not timing_df.empty:
            st.markdown("**Responsiveness snapshot.** These measurements show whether the interface stays fast enough for live exploration.")
            timing_cols = st.columns(3)
            timing_cols[0].metric("Median response", "{0:.1f} ms".format(float(timing_df["latency_ms"].median())))
            timing_cols[1].metric("Slowest response", "{0:.1f} ms".format(float(timing_df["latency_ms"].max())))
            timing_cols[2].metric("Measured actions", int(len(timing_df)))
            latency_threshold = st.slider("Response-time reference (ms)", min_value=0, max_value=1500, value=300, step=50)
            slow_rows = timing_df[timing_df["latency_ms"] >= float(latency_threshold)]
            interaction_filter = st.multiselect("Interaction types", options=timing_df["interaction"].tolist(), default=timing_df["interaction"].tolist())
            filtered_timings = timing_df[timing_df["interaction"].isin(interaction_filter)]
            if not filtered_timings.empty:
                st.plotly_chart(_build_timing_figure(filtered_timings), use_container_width=True)
            if slow_rows.empty:
                st.success("No measured interaction exceeds the current response-time reference.")
            else:
                st.warning("Some interactions exceed the current response-time reference: {0}".format(", ".join(slow_rows["interaction"].tolist())))

        if course_interactivity:
            st.markdown("**Course-scale interactivity report.** These exported metrics are the report-ready evidence for backend throughput and front-end response time.")
            interactivity_summary = course_interactivity.get("summary", {})
            summary_cols = st.columns(4)
            summary_cols[0].metric(
                "Weighted backend throughput",
                "{0:,.0f} rows/s".format(float(interactivity_summary.get("backend_weighted_records_per_second", 0.0))),
            )
            summary_cols[1].metric(
                "Peak backend throughput",
                "{0:,.0f} rows/s".format(float(interactivity_summary.get("backend_peak_records_per_second", 0.0))),
            )
            summary_cols[2].metric(
                "Median front-end response",
                "{0:.1f} ms".format(float(interactivity_summary.get("frontend_response_median_ms", 0.0))),
            )
            summary_cols[3].metric(
                "Sub-second actions",
                "{0}/{1}".format(
                    int(interactivity_summary.get("frontend_actions_under_one_second", 0)),
                    int(interactivity_summary.get("frontend_measurements", 0)),
                ),
            )
            measurement_df = pd.DataFrame(course_interactivity.get("measurements", []))
            if not measurement_df.empty:
                visible_cols = ["interaction", "layer", "records_processed", "records_per_second", "latency_ms", "note"]
                st.dataframe(measurement_df[visible_cols], use_container_width=True, hide_index=True)
            if bool(interactivity_summary.get("frontend_meets_one_second_reference", False)):
                st.success("The exported course interactivity report satisfies the one-second classroom-response reference.")
            else:
                st.warning("At least one exported front-end action is above the one-second classroom-response reference.")
        else:
            st.info("Run `python -m rlva.src.measure_course_interactivity` to export course-report throughput and response metrics.")

        if not submission_inventory.empty:
            st.markdown("**Submission package.** These files turn the project into a gradeable final-project bundle instead of just a code repository.")
            st.dataframe(submission_inventory, use_container_width=True, hide_index=True)
            st.caption("The package lives under `rlva/outputs/benchmark/reports/cs526_submission_package/` and is regenerated by `python -m rlva.src.export_course_deliverables`.")
        else:
            st.info("Run `python -m rlva.src.export_course_deliverables` to generate the report outline, presentation storyboard, video script, and submission checklist.")

        if not seed_df.empty:
            if analysis_focus == "Repeated-run stability":
                st.info("Recommended reading order: inspect the repeated-run spread first, then compare against the overall benchmark means.")
            stability_cols = st.columns(2)
            with stability_cols[0]:
                stability_env_options = sorted(_filter_like_global(seed_df, controls)["env"].unique().tolist()) if not _filter_like_global(seed_df, controls).empty else sorted(seed_df["env"].unique().tolist())
                stability_env = st.selectbox(
                    "Repeated-run system",
                    options=stability_env_options,
                    index=_control_default(stability_env_options, controls["env"]),
                    format_func=lambda name: ENV_LABELS.get(name, name),
                )
            with stability_cols[1]:
                stability_metric = st.selectbox(
                    "Repeated-run score",
                    options=["anomaly_auc", "shift_auc", "reward_jump_auc"],
                    format_func=_metric_label,
                )
            st.plotly_chart(
                _build_seed_stability_figure(seed_df=seed_df, env_name=stability_env, metric=stability_metric),
                use_container_width=True,
            )

        if not detection_df.empty:
            if analysis_focus == "Detection snapshot":
                st.info("Recommended reading order: start from unusual-behavior and operating-change scores here, then connect them back to the benchmark and simpler-method views.")
            detection_view = detection_df[detection_df["env"] != "all"].copy()
            detection_view["label"] = detection_view.apply(
                lambda row: "{0}<br>{1}".format(ENV_LABELS.get(str(row["env"]), str(row["env"])), _format_policy_name(str(row["policy"]))),
                axis=1,
            )
            long_detection = detection_view.melt(
                id_vars=["label"],
                value_vars=["anomaly_auc", "shift_auc", "reward_jump_auc"],
                var_name="metric",
                value_name="score",
            )
            long_detection["metric"] = long_detection["metric"].map(
                {"anomaly_auc": "Anomaly AUC", "shift_auc": "Shift AUC", "reward_jump_auc": "Reward-Jump AUC"}
            )
            detection_fig = px.bar(
                long_detection,
                x="label",
                y="score",
                color="metric",
                barmode="group",
                color_discrete_sequence=["#b84a39", "#355c7d", "#f0a202"],
                title="Detection Snapshot",
            )
            detection_fig.update_layout(template="plotly_white", xaxis_title="", yaxis_title="score", legend_title="")
            st.plotly_chart(detection_fig, use_container_width=True)

        if not ablation_df.empty:
            if analysis_focus == "Information importance":
                st.info("Recommended reading order: use this section to explain which pieces of summary information matter and whether a reward-only view is sufficient.")
            st.subheader("What Information Matters")
            ablation_controls = st.columns(3)
            with ablation_controls[0]:
                ablation_env_options = sorted(_filter_like_global(ablation_df, controls)["env"].unique().tolist()) if not _filter_like_global(ablation_df, controls).empty else sorted(ablation_df["env"].unique().tolist())
                ablation_env = st.selectbox("System", options=ablation_env_options, index=_control_default(ablation_env_options, controls["env"]), format_func=lambda name: ENV_LABELS.get(name, name))
            with ablation_controls[1]:
                env_policies = sorted(ablation_df[ablation_df["env"] == ablation_env]["policy"].unique().tolist())
                ablation_policy = st.selectbox("Controller", options=env_policies, index=_control_default(env_policies, controls["policy"]), format_func=_format_policy_name)
            with ablation_controls[2]:
                ablation_metric = st.selectbox("Score to inspect", options=["anomaly_auc", "shift_auc", "reward_mean"], key="ablation_metric")
            st.plotly_chart(
                build_ablation_heatmap(ablation_df, env_name=ablation_env, policy_name=ablation_policy, metric=ablation_metric),
                use_container_width=True,
            )
            ablation_slice = ablation_df[(ablation_df["env"] == ablation_env) & (ablation_df["policy"] == ablation_policy)].copy()
            best_group = ablation_slice.sort_values("{0}_mean".format(ablation_metric), ascending=False).iloc[0]
            st.markdown(
                (
                    "**Takeaway.** For {env} / {policy}, the strongest `{metric}` comes from "
                    "**{feature_group}** with score `{score:.3f}`."
                ).format(
                    env=ENV_LABELS.get(ablation_env, ablation_env),
                    policy=_format_policy_name(ablation_policy),
                    metric=ablation_metric,
                    feature_group=_format_feature_group(str(best_group["feature_group"])),
                    score=float(best_group["{0}_mean".format(ablation_metric)]),
                )
            )
            ablation_rank = ablation_slice.sort_values("{0}_mean".format(ablation_metric), ascending=_metric_ascending(ablation_metric))[["feature_group", "{0}_mean".format(ablation_metric)]].rename(columns={"{0}_mean".format(ablation_metric): "score"})
            ablation_rank["feature_group"] = ablation_rank["feature_group"].map(_format_feature_group)
            st.plotly_chart(
                _build_ranked_bar(
                    ablation_rank,
                    x="score",
                    y="feature_group",
                    title="Information-Group Ranking",
                    orientation="h",
                ),
                use_container_width=True,
            )

        if not robustness_window_df.empty or not robustness_cluster_df.empty:
            if analysis_focus == "Robustness":
                st.info("Recommended reading order: use the robustness plots to show that the project conclusions do not depend on one arbitrary setting.")
            st.subheader("How Stable The Findings Are")
            robustness_cols = st.columns(2)
            with robustness_cols[0]:
                if not robustness_window_df.empty:
                    robustness_env = st.selectbox(
                        "Segment-length system",
                        options=sorted(robustness_window_df["env"].unique().tolist()),
                        format_func=lambda name: ENV_LABELS.get(name, name),
                    )
                    robustness_metric = st.selectbox(
                        "Segment-length score",
                        options=["anomaly_auc", "shift_auc", "shift_localization_error"],
                        format_func=_metric_label,
                    )
                    st.plotly_chart(
                        _build_window_robustness_figure(robustness_window_df, env_name=robustness_env, metric=robustness_metric),
                        use_container_width=True,
                    )
                    st.caption("This view shows whether the conclusions persist as the time-segment length changes.")
                    robust_slice = robustness_window_df[robustness_window_df["env"] == robustness_env].sort_values("{0}_mean".format(robustness_metric), ascending=_metric_ascending(robustness_metric))
                    robust_slice = robust_slice.copy()
                    robust_slice["policy_label"] = robust_slice["policy"].map(_format_policy_name)
                    robust_slice["setting"] = robust_slice["policy_label"] + " / L=" + robust_slice["window_length"].astype(int).astype(str)
                    st.plotly_chart(
                        _build_ranked_bar(
                            robust_slice.rename(columns={"{0}_mean".format(robustness_metric): "score"}),
                            x="score",
                            y="setting",
                            title="Segment-Length Stability Ranking",
                            color="policy_label",
                            orientation="h",
                        ),
                        use_container_width=True,
                    )
            with robustness_cols[1]:
                if not robustness_cluster_df.empty:
                    cluster_env = st.selectbox(
                        "Behavior-group system",
                        options=sorted(robustness_cluster_df["env"].unique().tolist()),
                        format_func=lambda name: ENV_LABELS.get(name, name),
                    )
                    st.plotly_chart(
                        _build_cluster_robustness_figure(robustness_cluster_df, env_name=cluster_env),
                        use_container_width=True,
                    )
                    st.caption("This view tracks whether the strongest controller difference remains stable as the behavior-group granularity changes.")
                    cluster_slice = robustness_cluster_df[robustness_cluster_df["env"] == cluster_env].sort_values("top1_js_div_mean", ascending=False)
                    cluster_slice = cluster_slice.copy()
                    cluster_slice["pair"] = cluster_slice["policy_1"].map(_format_policy_name) + " vs " + cluster_slice["policy_2"].map(_format_policy_name)
                    cluster_slice["setting"] = cluster_slice["pair"] + " / K=" + cluster_slice["cluster_count"].astype(int).astype(str)
                    st.plotly_chart(
                        _build_ranked_bar(
                            cluster_slice.rename(columns={"top1_js_div_mean": "score"}),
                            x="score",
                            y="setting",
                            title="Behavior-Group Stability Ranking",
                            color="pair",
                            orientation="h",
                        ),
                        use_container_width=True,
                    )
def _render_figures_panel() -> None:
    _render_active_dataset_banner()
    benchmark_df = cached_report_csv("benchmark_table.csv")
    if benchmark_df.empty:
        st.warning("Benchmark reports are not available yet. Run the benchmark pipeline first.")
        return

    controls = _global_controls(benchmark_df)
    coverage_df = cached_artifact_coverage()
    case_index_df = _case_study_index()
    asset_catalog = cached_asset_catalog()

    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">Project Showcase</div>
        <div class="hero-title">Figures And Examples</div>
        <p class="hero-copy">
            This workspace packages the visual evidence behind the diagnostic system so a user can quickly find the
            best figure or example for a report, review meeting, or classroom presentation.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("**Purpose.** This page is the presentation layer of the project: figures, curated examples, and source-linked evidence.")
    if not coverage_df.empty:
        st.caption("The inventory below counts traces and summaries for all five controller families, and model checkpoints for the learned controllers only (`pg`, `ppo`, `dqn`).")
        st.plotly_chart(_build_coverage_figure(coverage_df), use_container_width=True)
    if not case_index_df.empty:
        st.info("Current case-study assets are a curated subset of exported examples. Right now they cover `dqn` only, so treat them as selected qualitative evidence rather than complete controller coverage.")

    if not asset_catalog.empty:
        if "pending_asset_category" in st.session_state:
            st.session_state["asset_category_selector"] = st.session_state.pop("pending_asset_category")
        asset_controls = st.columns(3)
        with asset_controls[0]:
            asset_category = st.selectbox(
                "Figure type",
                options=["paper_figure", "case_study"],
                key="asset_category_selector",
                format_func=lambda value: "Project figure" if value == "paper_figure" else "Example case",
            )
        filtered_assets = asset_catalog[asset_catalog["category"] == asset_category].copy()
        if controls["env"] != "all":
            filtered_assets = filtered_assets[(filtered_assets["env"].isna()) | (filtered_assets["env"] == controls["env"])]
        if controls["policy"] != "all":
            filtered_assets = filtered_assets[(filtered_assets["policy"].isna()) | (filtered_assets["policy"] == controls["policy"])]
        with asset_controls[1]:
            env_options = ["all"] + sorted([value for value in filtered_assets["env"].dropna().unique().tolist()])
            selected_env = st.selectbox(
                "System",
                options=env_options,
                format_func=lambda value: "All environments" if value == "all" else ENV_LABELS.get(value, value),
            )
        if selected_env != "all":
            filtered_assets = filtered_assets[filtered_assets["env"] == selected_env]
        with asset_controls[2]:
            if asset_category == "case_study":
                policy_options = ["all"] + sorted([value for value in filtered_assets["policy"].dropna().unique().tolist()])
                selected_policy = st.selectbox(
                    "Controller",
                    options=policy_options,
                    format_func=lambda value: "All policies" if value == "all" else _format_policy_name(value),
                )
                if selected_policy != "all":
                    filtered_assets = filtered_assets[filtered_assets["policy"] == selected_policy]
            else:
                selected_policy = "all"
                st.markdown("`Controller filter` is not used for global project figures.")

        if filtered_assets.empty:
            st.warning("No figures match the current filters.")
            return

        asset_labels = filtered_assets["stem"].tolist()
        asset_state_key = "selected_asset_{0}".format(asset_category)
        if asset_state_key not in st.session_state or st.session_state[asset_state_key] not in asset_labels:
            st.session_state[asset_state_key] = asset_labels[0]
        selected_asset = str(st.session_state[asset_state_key])
        selected_row = filtered_assets[filtered_assets["stem"] == selected_asset].iloc[0]
        asset_path = Path(str(selected_row["png_path"]))
        left, right = st.columns([3, 2])
        with left:
            st.image(str(asset_path), caption=selected_row["stem"])
        with right:
            st.markdown("**Active figure.** {0}".format(_asset_label(selected_row)))
            st.markdown("**Description.** {0}".format(str(selected_row["description"]) or "Exported figure from the benchmark pipeline."))
            st.markdown("**PNG path.** `{0}`".format(asset_path.name))
            pdf_path = Path(str(selected_row["pdf_path"]))
            if pdf_path.exists():
                st.markdown("**PDF pair.** `{0}`".format(pdf_path.name))
            if str(selected_row["category"]) == "case_study" and not case_index_df.empty:
                match = case_index_df[
                    (case_index_df["env"] == selected_row["env"])
                    & (case_index_df["policy"] == selected_row["policy"])
                ]
                if not match.empty:
                    case_row = match.iloc[0]
                    note = _case_study_note(str(case_row["env"]), str(case_row["policy"]), int(case_row["seed"]))
                    st.markdown("**Why this case matters.** {0} {1}".format(note["headline"], note["why"]))
                    st.markdown("**Source trace.** `{0}`".format(Path(str(case_row["trace_path"])).name))
                    st.markdown("**Source summary.** `{0}`".format(Path(str(case_row["summary_path"])).name))

        st.subheader("Figure Gallery")
        st.caption("Select a card below to switch the main viewer.")
        thumb_assets = filtered_assets.copy().reset_index(drop=True)
        thumb_assets["thumb_label"] = thumb_assets.apply(_asset_label, axis=1)
        for start in range(0, len(thumb_assets), 4):
            row_assets = thumb_assets.iloc[start : start + 4]
            thumb_cols = st.columns(len(row_assets))
            for idx, (_, asset) in enumerate(row_assets.iterrows()):
                with thumb_cols[idx]:
                    active_class = " asset-card-active" if str(asset["stem"]) == str(st.session_state[asset_state_key]) else ""
                    st.markdown('<div class="asset-card{0}">'.format(active_class), unsafe_allow_html=True)
                    st.image(str(asset["png_path"]), caption=asset["thumb_label"])
                    if str(asset["stem"]) == str(st.session_state[asset_state_key]):
                        st.markdown("**Selected**")
                    if st.button("Focus", key="thumb_{0}_{1}".format(asset_category, asset["stem"]), type="secondary"):
                        st.session_state[asset_state_key] = str(asset["stem"])
                        _trigger_rerun()
                    if str(asset["description"]):
                        st.caption(str(asset["description"]))
                    st.markdown("</div>", unsafe_allow_html=True)
        if asset_category == "case_study" and not case_index_df.empty:
            compare_options = case_index_df.apply(
                lambda row: "{0} / {1} / seed {2}".format(
                    _format_env_name(str(row["env"])),
                    _format_policy_name(str(row["policy"])),
                    int(row["seed"]),
                ),
                axis=1,
            ).tolist()
            selected_compare = st.multiselect("Compare example notes", options=compare_options, default=compare_options[: min(2, len(compare_options))])
            for label in selected_compare:
                idx = compare_options.index(label)
                row = case_index_df.iloc[idx]
                note = _case_study_note(str(row["env"]), str(row["policy"]), int(row["seed"]))
                st.markdown("- **{0}**: {1} {2}".format(label, note["headline"], note["why"]))


def _render_course_deliverables_panel() -> None:
    _render_active_dataset_banner()
    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">Instructor Workspace</div>
        <div class="hero-title">Course Deliverables And Admin Assets</div>
        <p class="hero-copy">
            This panel collects the grading-facing artifacts: the final brief, the interactivity report,
            the submission package, and the persistent case notebook created by users.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    brief_path = BENCHMARK_REPORT_DIR / "cs526_final_project_brief.md"
    interactivity_path = BENCHMARK_REPORT_DIR / "course_interactivity_report.md"
    task_performance_path = BENCHMARK_REPORT_DIR / "course_task_performance_report.md"
    notebook_path = USER_CASE_NOTEBOOK_PATH
    submission_inventory = cached_submission_package_inventory()
    task_performance_metrics = cached_course_task_performance_metrics()
    task_performance_summary = task_performance_metrics.get("summary", {}) if task_performance_metrics else {}

    artifact_rows = [
        {"label": "Final project brief", "path": brief_path},
        {"label": "Interactivity report", "path": interactivity_path},
        {"label": "User task performance report", "path": task_performance_path},
    ]
    if notebook_path.exists():
        artifact_rows.append({"label": "User case notebook", "path": notebook_path})
    for _, row in submission_inventory.iterrows():
        artifact_rows.append({"label": str(row["artifact"]), "path": COURSE_SUBMISSION_DIR / str(row["artifact"])})

    artifact_df = pd.DataFrame(artifact_rows)
    summary_cols = st.columns(5)
    summary_cols[0].metric("Submission files", int(len(submission_inventory)))
    summary_cols[1].metric("Notebook entries", int(len(_case_notebook_frame())))
    summary_cols[2].metric("Brief present", "Yes" if brief_path.exists() else "No")
    summary_cols[3].metric("Interactivity report", "Yes" if interactivity_path.exists() else "No")
    summary_cols[4].metric("Completed user tasks", int(task_performance_summary.get("completed_tasks", 0)))

    if task_performance_summary:
        st.info(
            "User-task evaluation: median completion time `{0:.1f}s`, support-signal rate `{1:.1%}`, export usage rate `{2:.1%}`.".format(
                float(task_performance_summary.get("median_duration_seconds", 0.0)),
                float(task_performance_summary.get("support_signal_rate", 0.0)),
                float(task_performance_summary.get("export_completion_rate", 0.0)),
            )
        )

    if not submission_inventory.empty:
        st.dataframe(submission_inventory, use_container_width=True, hide_index=True)

    if artifact_df.empty:
        st.info("No deliverable artifacts are available yet.")
        return

    selected_label = st.selectbox("Artifact preview", options=artifact_df["label"].tolist())
    selected_row = artifact_df[artifact_df["label"] == selected_label].iloc[0]
    selected_path = Path(str(selected_row["path"]))
    if not selected_path.exists():
        st.warning("The selected artifact is not available on disk.")
        return
    st.caption("Previewing `{0}`".format(selected_path))
    content = selected_path.read_text(encoding="utf-8")
    st.code(content, language="markdown" if selected_path.suffix in {".md", ".jsonl"} else None)


def _render_instructor_workspace() -> None:
    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">Instructor Workspace</div>
        <div class="hero-title">Evaluation, Assets, And Deliverables</div>
        <p class="hero-copy">
            This workspace is for validation, grading, and presentation support. It includes trust metrics,
            exported figures, interactivity evidence, and course-facing submission materials.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    panel = st.radio(
        "Instructor panel",
        options=["Trust & Evaluation", "Figures & Assets", "Course Deliverables"],
        horizontal=True,
    )
    if panel == "Trust & Evaluation":
        _render_validation_panel()
    elif panel == "Figures & Assets":
        _render_figures_panel()
    else:
        _render_course_deliverables_panel()


def main() -> None:
    st.set_page_config(page_title="RLVA Results Dashboard", layout="wide")
    _inject_dashboard_css()
    st.title(PRIMARY_PRODUCT_NAME)
    st.caption("Traffic-operations review product powered by RLVA, with a separate instructor workspace for evaluation, figures, and course deliverables.")

    panel = st.radio("Workspace", options=["User Workspace", "Instructor Workspace"], horizontal=True)
    if panel == "User Workspace":
        _render_demo_panel()
    else:
        _render_instructor_workspace()


if __name__ == "__main__":
    main()
