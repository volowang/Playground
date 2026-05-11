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
    custom = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(custom, (list, tuple)) and custom:
        try:
            return int(custom[0])
        except (TypeError, ValueError):
            pass
    x_value = point.get("x") if isinstance(point, dict) else None
    try:
        return int(x_value)
    except (TypeError, ValueError):
        return None


def _selected_customdata(selection_event: Any) -> List[Any]:
    points = _extract_points(selection_event)
    if not points:
        return []
    point = points[0]
    if not isinstance(point, dict):
        return []
    custom = point.get("customdata")
    if isinstance(custom, (list, tuple)):
        return list(custom)
    if custom is None:
        return []
    return [custom]


def _selected_label(selection_event: Any) -> Optional[str]:
    points = _extract_points(selection_event)
    if not points:
        return None
    point = points[0]
    if not isinstance(point, dict):
        return None
    for key in ("label", "x", "y"):
        value = point.get(key)
        if value is not None:
            return str(value)
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


def _policy_from_display(label: str) -> Optional[str]:
    for policy in BENCHMARK_POLICIES:
        if _format_policy_name(policy) == label or policy == label:
            return policy
    return None


def _env_from_display(label: str) -> Optional[str]:
    for env_name in ENVIRONMENTS:
        if _format_env_name(env_name) == label or _scenario_profile(env_name)["short_label"] == label or env_name == label:
            return env_name
    return None


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
    "anomaly_auc": "Find unusual windows",
    "shift_auc": "Find changes",
    "reward_mean": "Reward Mean",
    "reward_jump_auc": "Find result drops",
    "shift_localization_error": "Change timing error",
}

LOWER_IS_BETTER_METRICS = {"shift_localization_error"}

PRIMARY_ANALYST_ROLE = "Operations analyst"
PRIMARY_SUPERVISOR_ROLE = "Operations supervisor"
PRIMARY_OPERATIONS_OWNER = "operations review team"
PRIMARY_SUPERVISOR_OWNER = "controller review board"
PRIMARY_PRODUCT_NAME = "Decision System Review Studio"
PRIMARY_PRODUCT_ENGINE = "RLVA"

SYSTEM_PROFILES: Dict[str, Dict[str, str]] = {
    "traffic": {
        "short_label": "Traffic",
        "user": "Traffic operations analyst",
        "owner": "traffic incident desk",
        "problem": "Find risky signal-control windows and confirm corridor regime shifts.",
        "high_action": "Escalate corridor incident and review the active signal plan.",
        "moderate_action": "Review adjacent windows before keeping the signal plan live.",
        "monitor_action": "Keep monitoring; no immediate handoff is needed.",
    },
    "inventory": {
        "short_label": "Inventory",
        "user": "Inventory planner",
        "owner": "replenishment planning desk",
        "problem": "Spot demand shocks, stock-risk windows, and unstable replenishment policies.",
        "high_action": "Escalate stock-risk window and revise the replenishment rule.",
        "moderate_action": "Review demand-shift evidence before approving the controller.",
        "monitor_action": "Monitor the policy; current evidence is low risk.",
    },
    "queue": {
        "short_label": "Queue",
        "user": "Service operations manager",
        "owner": "queue operations desk",
        "problem": "Find congestion windows and compare service-scheduling controllers.",
        "high_action": "Escalate service congestion and rebalance scheduling policy.",
        "moderate_action": "Review nearby windows before keeping the scheduler active.",
        "monitor_action": "Keep monitoring normal service behavior.",
    },
    "cartpole": {
        "short_label": "CartPole",
        "user": "Control QA engineer",
        "owner": "controller validation desk",
        "problem": "Catch unstable controller behavior before deployment.",
        "high_action": "Block rollout and inspect the unstable controller window.",
        "moderate_action": "Run controller comparison before approving the policy.",
        "monitor_action": "Controller appears stable in the current window.",
    },
    "lunarlander": {
        "short_label": "LunarLander",
        "user": "Robotics safety reviewer",
        "owner": "robotics safety desk",
        "problem": "Detect unsafe descent behavior and policy-change windows.",
        "high_action": "Pause deployment and review the risky landing behavior.",
        "moderate_action": "Review adjacent descent windows and compare controllers.",
        "monitor_action": "Continue monitoring; current landing behavior is low risk.",
    },
}

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
    "Data": "The project uses 5 environments (`queue`, `inventory`, `traffic`, `cartpole`, `lunarlander`) and 5 policy families (`pg`, `ppo`, `dqn`, `random`, `heuristic`). Each system is exposed as a user-facing scenario.",
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
    "Find anomalies": "Find high-risk time windows.",
    "Locate shift": "Find where system behavior changed.",
    "Compare policies": "Compare controllers by behavior and outcome.",
    "Tell the project story": "Review the system end-to-end.",
}

DEMO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Traffic": {"env": "traffic", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
    "Inventory": {"env": "inventory", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
    "Queue": {"env": "queue", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
    "CartPole": {"env": "cartpole", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
    "LunarLander": {"env": "lunarlander", "policy": "dqn", "seed": 23, "task": "Find anomalies", "metric": "anomaly_score"},
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
    "High-risk system behavior",
    "Operating regime change",
    "Strategy comparison evidence",
    "Follow-up required",
    "Exported to review packet",
]

ACTION_COLOR_MAP = {
    "Escalate now": "#b84a39",
    "Review next": "#f0a202",
    "Monitor": "#355c7d",
}


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f7fa;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }
        .hero-shell {
            padding: 1.35rem 1.5rem 1.15rem 1.5rem;
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(20, 54, 86, 0.96), rgba(49, 87, 122, 0.92));
            color: #f7fbff;
            box-shadow: 0 12px 32px rgba(18, 34, 51, 0.12);
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
            border-radius: 8px;
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
            border-radius: 8px;
            padding: 0.55rem 0.55rem 0.35rem 0.55rem;
            box-shadow: 0 10px 24px rgba(39, 68, 93, 0.05);
        }
        .asset-card-active {
            border: 2px solid rgba(184, 74, 57, 0.55);
            box-shadow: 0 14px 28px rgba(184, 74, 57, 0.10);
        }
        .decision-strip {
            background: #ffffff;
            border: 1px solid rgba(39, 68, 93, 0.10);
            border-left: 5px solid #b84a39;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.65rem 0 1rem 0;
            box-shadow: 0 10px 24px rgba(39, 68, 93, 0.05);
        }
        .decision-strip h3 {
            margin: 0 0 0.25rem 0;
            font-size: 1.05rem;
        }
        .decision-strip p {
            margin: 0;
            color: #2f4050;
            line-height: 1.45;
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


def _queue_multi_dashboard_update(**updates: Any) -> None:
    pending = dict(st.session_state.get("_pending_multi_dashboard_update", {}))
    pending.update(updates)
    st.session_state["_pending_multi_dashboard_update"] = pending
    _trigger_rerun()


def _apply_pending_multi_dashboard_update() -> None:
    pending = st.session_state.pop("_pending_multi_dashboard_update", {})
    if not isinstance(pending, dict):
        return
    for key, value in pending.items():
        st.session_state[key] = value


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
        "Find anomalies": "Find risky windows",
        "Locate shift": "Find behavior change",
        "Compare policies": "Compare two controllers",
        "Tell the project story": "Review the system end-to-end",
    }
    return mapping.get(task_name, task_name)


def _focus_label(focus_name: str) -> str:
    mapping = {
        "Balanced": "Balanced",
        "Highest risk": "Most urgent",
        "Change": "Changed behavior",
        "Task only": "Match goal",
    }
    return mapping.get(focus_name, focus_name)


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


def _scenario_profile(env_name: str) -> Dict[str, str]:
    return SYSTEM_PROFILES.get(
        env_name,
        {
            "short_label": _format_env_name(env_name),
            "user": "Operations analyst",
            "owner": "operations review team",
            "problem": "Find risky behavior windows and compare controllers.",
            "high_action": "Escalate the focused window for review.",
            "moderate_action": "Review nearby windows before approving the controller.",
            "monitor_action": "Continue monitoring the current behavior.",
        },
    )


def _apply_system_preset(env_name: str) -> None:
    profile = _scenario_profile(env_name)
    st.session_state["multi_env"] = env_name
    st.session_state["multi_policy"] = "dqn"
    st.session_state["multi_task"] = "Find anomalies"
    st.session_state["multi_color_by"] = "anomaly_score"
    st.session_state["multi_action_filter"] = "All actions"
    st.session_state["selected_k"] = None
    st.session_state["selected_cluster"] = None
    seeds = cached_available_seeds(env_name, "dqn")
    st.session_state["multi_seed"] = 23 if 23 in seeds else (int(seeds[0]) if seeds else None)
    _reset_task_session(
        {
            "workflow": profile["short_label"],
            "role": profile["user"],
            "task": "Find anomalies",
            "env": env_name,
            "policy": "dqn",
            "seed": st.session_state.get("multi_seed"),
        }
    )


def _system_action_summary(
    env_name: str,
    task_name: str,
    policy: str,
    selected_seed: Optional[int],
    window_row: pd.Series,
    compare_row: Optional[pd.Series],
) -> Dict[str, Any]:
    profile = _scenario_profile(env_name)
    suspicion = float(window_row["anomaly_score"])
    change_conf = float(window_row["regime_shift_score"])
    reward = float(window_row["r_bar"])
    intervention_flag = bool(window_row["gt_intervention"]) if "gt_intervention" in window_row.index else False
    if suspicion >= 0.8 or change_conf >= 0.8 or intervention_flag:
        severity = "High"
        decision = profile["high_action"]
    elif suspicion >= 0.5 or change_conf >= 0.5:
        severity = "Moderate"
        decision = profile["moderate_action"]
    else:
        severity = "Low"
        decision = profile["monitor_action"]

    if task_name == "Compare policies":
        recommendation = "Compare the active controller against the strongest alternate controller, then export the comparison brief."
    elif task_name == "Locate shift":
        recommendation = "Inspect the windows before and after the focused change point."
    else:
        recommendation = "Start with the highest-priority window, then inspect adjacent windows."

    evidence = [
        "System: {0}".format(_format_env_name(env_name)),
        "Controller: {0}".format(_format_policy_name(policy)),
        "Run: {0}".format(selected_seed if selected_seed is not None else "default"),
        "Window: k={0}, t={1}-{2}".format(int(window_row["k"]), int(window_row["t_start"]), int(window_row["t_end"])),
        "Risk: {0:.3f}".format(suspicion),
        "Change: {0:.3f}".format(change_conf),
        "Outcome: {0:.3f}".format(reward),
    ]
    if intervention_flag:
        evidence.append("Reference intervention overlaps this window.")
    if compare_row is not None:
        evidence.append(
            "Controller gap: group {0}, behavior difference {1:.3f}, outcome gap {2:.3f}.".format(
                int(compare_row["cluster"]),
                float(compare_row["js_div"]),
                float(compare_row["reward_gap"]),
            )
        )

    return {
        "decision": decision,
        "severity": severity,
        "owner": profile["owner"],
        "evidence": evidence,
        "risk": "{0}: risk {1:.3f}, change {2:.3f}.".format(severity, suspicion, change_conf),
        "recommendation": recommendation,
        "action_title": "Recommended next step",
    }


def _minmax_index(values: pd.Series, reverse: bool = False) -> pd.Series:
    numeric = values.astype(float)
    low = float(numeric.min())
    high = float(numeric.max())
    if high <= low:
        scaled = pd.Series([50.0] * len(numeric), index=numeric.index)
    else:
        scaled = (numeric - low) / (high - low) * 100.0
    return 100.0 - scaled if reverse else scaled


def _operational_focus_frame(filtered_df: pd.DataFrame, selected_k: Optional[int]) -> pd.DataFrame:
    frame = filtered_df.copy()
    frame["risk_index"] = _minmax_index(frame["anomaly_score"])
    frame["change_index"] = _minmax_index(frame["regime_shift_score"])
    frame["outcome_stress_index"] = _minmax_index(frame["r_bar"], reverse=True)
    frame["priority_index"] = (
        0.50 * frame["risk_index"] + 0.35 * frame["change_index"] + 0.15 * frame["outcome_stress_index"]
    )
    if "gt_intervention" in frame.columns:
        frame.loc[frame["gt_intervention"].astype(bool), "priority_index"] += 10.0
    frame["priority_index"] = frame["priority_index"].clip(lower=0.0, upper=100.0)
    frame["priority_size"] = frame["priority_index"].clip(lower=8.0) + 8.0
    frame["window_label"] = frame.apply(_window_label, axis=1)
    frame["is_selected"] = frame["k"].astype(int).eq(int(selected_k)) if selected_k is not None else False

    def _action(row: pd.Series) -> str:
        if bool(row.get("gt_intervention", False)) or float(row["priority_index"]) >= 72.0:
            return "Escalate now"
        if float(row["priority_index"]) >= 42.0:
            return "Review next"
        return "Monitor"

    frame["recommended_action"] = frame.apply(_action, axis=1)
    return frame


def _focused_action_row(focus_df: pd.DataFrame, selected_k: Optional[int]) -> pd.Series:
    if selected_k is not None:
        selected = focus_df[focus_df["k"].astype(int) == int(selected_k)]
        if not selected.empty:
            return selected.iloc[0]
    return focus_df.sort_values("priority_index", ascending=False).iloc[0]


def _explain_value(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return "{0:.{1}f}".format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _point_explanation_metric(label: str, value: Any) -> Dict[str, str]:
    return {"What": label, "Value": str(value)}


def _set_point_explanation(
    *,
    signature: str,
    title: str,
    chart: str,
    summary: str,
    metrics: List[Dict[str, str]],
    next_step: str,
) -> None:
    if st.session_state.get("_last_point_explanation_signature") == signature and "_point_explanation" in st.session_state:
        return
    st.session_state["_last_point_explanation_signature"] = signature
    st.session_state["_point_explanation"] = {
        "signature": signature,
        "title": title,
        "chart": chart,
        "summary": summary,
        "metrics": metrics,
        "next_step": next_step,
    }


def _render_point_explanation_body(payload: Dict[str, Any]) -> None:
    st.caption(str(payload.get("chart", "Selected chart point")))
    st.markdown(str(payload.get("summary", "")))
    metrics = payload.get("metrics", [])
    if metrics:
        st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True)
    next_step = str(payload.get("next_step", ""))
    if next_step:
        st.info(next_step)
    if st.button("Close explanation", key="close_point_explanation", use_container_width=True):
        st.session_state.pop("_point_explanation", None)
        st.session_state.pop("_last_point_explanation_signature", None)
        _trigger_rerun()


def _render_point_explanation_module() -> None:
    payload = st.session_state.get("_point_explanation")
    if not isinstance(payload, dict):
        return
    dialog = getattr(st, "dialog", None)
    if callable(dialog):
        @dialog(str(payload.get("title", "Point explanation")))
        def _point_explanation_dialog() -> None:
            _render_point_explanation_body(payload)

        _point_explanation_dialog()
        return

    with st.expander(str(payload.get("title", "Point explanation")), expanded=True):
        _render_point_explanation_body(payload)


def _window_explanation_text(policy: str, row: pd.Series, task_name: str) -> str:
    return "This point represents one time window. The dashboard is checking whether this window looks unusual, whether the system changed around it, and whether the result looks weak enough to review."


def _scenario_problem_text(env_name: str, row: pd.Series) -> str:
    risk = float(row.get("risk_index", row.get("anomaly_score", 0.0)))
    change = float(row.get("change_index", row.get("regime_shift_score", 0.0)))
    outcome = float(row.get("outcome_stress_index", 0.0))
    profile = _scenario_profile(env_name)
    if env_name == "traffic":
        base = "In this traffic scenario, the point may mean the signal plan is not matching current road conditions."
        issue = "It can point to growing congestion, a sudden traffic-pattern change, or a signal timing window that needs operator review."
    elif env_name == "inventory":
        base = "In this inventory scenario, the point may mean demand or stock behavior has moved away from normal."
        issue = "It can point to a stockout risk, demand shock, or a replenishment rule that should be checked."
    elif env_name == "queue":
        base = "In this service-queue scenario, the point may mean the line is becoming harder to serve."
        issue = "It can point to congestion, a service-rate mismatch, or a scheduling choice that needs attention."
    elif env_name == "cartpole":
        base = "In this control-safety scenario, the point may mean the controller is becoming less stable."
        issue = "It can point to a rollout window where the controller should be tested before approval."
    elif env_name == "lunarlander":
        base = "In this robotics scenario, the point may mean the lander is entering a less safe descent pattern."
        issue = "It can point to a landing behavior that should be reviewed before deployment."
    else:
        base = "In this scenario, the point may mean the system is behaving differently from normal."
        issue = profile["problem"]

    reasons: List[str] = []
    if risk >= 70:
        reasons.append("it looks unusual")
    if change >= 70:
        reasons.append("the behavior appears to be changing")
    if outcome >= 70:
        reasons.append("the result looks weak")
    if not reasons:
        reasons.append("it is still worth watching under the current filters")
    return "{0} {1} Main reason: {2}.".format(base, issue, ", ".join(reasons))


def _set_window_point_explanation(
    *,
    chart: str,
    env_name: str,
    policy: str,
    task_name: str,
    row: pd.Series,
) -> None:
    action = str(row.get("recommended_action", "Review next"))
    summary = (
        "**What this point means.** {window} is one reviewed time window.\n\n"
        "**What it says about this system.** {problem}\n\n"
        "**Why it was selected.** {generic}"
    ).format(
        window=_window_label(row),
        problem=_scenario_problem_text(env_name, row),
        generic=_window_explanation_text(policy, row, task_name),
    )
    metrics = [
        _point_explanation_metric("Scenario", _format_env_name(env_name)),
        _point_explanation_metric("Controller", _format_policy_name(policy)),
        _point_explanation_metric("Window", "k={0}".format(int(row["k"]))),
        _point_explanation_metric("Looks unusual", "{0}/100".format(_explain_value(row.get("risk_index"), 1))),
        _point_explanation_metric("Looks changed", "{0}/100".format(_explain_value(row.get("change_index"), 1))),
        _point_explanation_metric("Result concern", "{0}/100".format(_explain_value(row.get("outcome_stress_index"), 1))),
        _point_explanation_metric("Review priority", "{0}/100".format(_explain_value(row.get("priority_index"), 1))),
        _point_explanation_metric("Suggested action", action),
    ]
    _set_point_explanation(
        signature="window:{0}:{1}:{2}:{3}".format(chart, env_name, policy, int(row["k"])),
        title="What this point means",
        chart=chart,
        summary=summary,
        metrics=metrics,
        next_step="Next: check the windows before and after this one. If they show the same pattern, it is probably a real system issue, not just a one-window spike.",
    )


def _set_action_mix_explanation(action: str, focus_df: pd.DataFrame) -> None:
    count = int(focus_df["recommended_action"].eq(action).sum())
    total = max(int(len(focus_df)), 1)
    share = count / total
    _set_point_explanation(
        signature="action:{0}:{1}".format(action, total),
        title="What this point means",
        chart="Recommended actions",
        summary=(
            "**What this point means.** This bar counts how many current windows fall into `{0}`.\n\n"
            "**What it says about this system.** A large bar means many windows need the same level of attention."
        ).format(action),
        metrics=[
            _point_explanation_metric("Suggested action", action),
            _point_explanation_metric("Windows", count),
            _point_explanation_metric("Share", "{0:.1%}".format(share)),
        ],
        next_step="Next: use this filter to focus only on the windows with the same suggested action.",
    )


def _set_signal_explanation(signal_key: str, env_name: str, policy: str, row: pd.Series) -> None:
    signal_map = {
        "risk": ("Looks unusual", "This bar says whether the selected window looks different from normal behavior in this scenario."),
        "change": ("Looks changed", "This bar says whether the system seems to have changed around this window."),
        "outcome": ("Result concern", "This bar says whether the result in this window looks weaker than expected."),
        "priority": ("Review priority", "This bar combines the warning signs into one review order."),
    }
    signal_label, summary = signal_map.get(signal_key, ("Signal", "This is one diagnostic signal for the focused window."))
    value_map = {
        "risk": row.get("risk_index"),
        "change": row.get("change_index"),
        "outcome": row.get("outcome_stress_index"),
        "priority": row.get("priority_index"),
    }
    _set_point_explanation(
        signature="signal:{0}:{1}:{2}:{3}".format(env_name, policy, int(row["k"]), signal_key),
        title="What this point means",
        chart="Why this window matters",
        summary="**What this point means.** {0}\n\n**What it says about this system.** {1}".format(
            summary,
            _scenario_problem_text(env_name, row),
        ),
        metrics=[
            _point_explanation_metric("Scenario", _format_env_name(env_name)),
            _point_explanation_metric("Controller", _format_policy_name(policy)),
            _point_explanation_metric("Window", "k={0}".format(int(row["k"]))),
            _point_explanation_metric(signal_label, "{0}/100".format(_explain_value(value_map.get(signal_key), 1))),
        ],
        next_step="Next: use this bar to understand why the window was placed in the review queue.",
    )


def _set_benchmark_cell_explanation(benchmark_df: pd.DataFrame, env_name: str, policy: str, metric: str, chart: str) -> None:
    row = benchmark_df[(benchmark_df["env"] == env_name) & (benchmark_df["policy"] == policy)]
    if row.empty:
        return
    active = row.iloc[0]
    mean_col = "{0}_mean".format(metric)
    std_col = "{0}_std".format(metric)
    _set_point_explanation(
        signature="benchmark:{0}:{1}:{2}:{3}".format(chart, env_name, policy, metric),
        title="What this point means",
        chart=chart,
        summary=(
            "**What this point means.** This score shows how well `{controller}` worked on past `{scenario}` data.\n\n"
            "**What it says about this system.** Higher values mean the dashboard found the known warning windows more reliably for this controller."
        ).format(controller=_format_policy_name(policy), scenario=_format_env_name(env_name)),
        metrics=[
            _point_explanation_metric("Scenario", _format_env_name(env_name)),
            _point_explanation_metric("Controller", _format_policy_name(policy)),
            _point_explanation_metric("Score type", _metric_label(metric)),
            _point_explanation_metric("Average score", _explain_value(active.get(mean_col))),
            _point_explanation_metric("Run-to-run spread", _explain_value(active.get(std_col))),
        ],
        next_step="Next: click this score to open that scenario and controller in the detailed window charts.",
    )


def _set_cluster_point_explanation(env_name: str, p1: str, p2: str, cluster_row: pd.Series, chart: str) -> None:
    diff = float(cluster_row.get("js_div", 0.0))
    gap = float(cluster_row.get("reward_gap", 0.0))
    if diff >= 0.45:
        difference_text = "The two controllers behave very differently in this group."
    elif diff >= 0.20:
        difference_text = "The two controllers behave somewhat differently in this group."
    else:
        difference_text = "The two controllers behave similarly in this group."
    if abs(gap) >= 0.20:
        outcome_text = "The result impact is large enough to check before choosing a controller."
    else:
        outcome_text = "The result impact is smaller, so this group is mainly useful for understanding behavior."
    summary = (
        "**What this point means.** This group contains windows where `{a}` and `{b}` were compared.\n\n"
        "**What it says about this system.** {difference} {outcome}"
    ).format(a=_format_policy_name(p1), b=_format_policy_name(p2), difference=difference_text, outcome=outcome_text)
    _set_point_explanation(
        signature="cluster:{0}:{1}:{2}:{3}:{4}".format(chart, env_name, p1, p2, int(cluster_row["cluster"])),
        title="What this point means",
        chart=chart,
        summary=summary,
        metrics=[
            _point_explanation_metric("Scenario", _format_env_name(env_name)),
            _point_explanation_metric("Controller A", _format_policy_name(p1)),
            _point_explanation_metric("Controller B", _format_policy_name(p2)),
            _point_explanation_metric("Group", int(cluster_row["cluster"])),
            _point_explanation_metric("How different", _explain_value(cluster_row.get("js_div"))),
            _point_explanation_metric("Result impact", _explain_value(cluster_row.get("reward_gap"))),
            _point_explanation_metric("Windows", int(cluster_row.get("n_total", 0))),
        ],
        next_step="Next: use this group to decide whether the alternate controller should be reviewed before deployment.",
    )


def _build_risk_change_matrix(
    focus_df: pd.DataFrame,
    selected_k: Optional[int],
    anomaly_threshold: float,
    shift_threshold: float,
) -> go.Figure:
    fig = px.scatter(
        focus_df,
        x="regime_shift_score",
        y="anomaly_score",
        color="recommended_action",
        size="priority_size",
        hover_name="window_label",
        custom_data=["k", "priority_index", "r_bar", "recommended_action"],
        color_discrete_map=ACTION_COLOR_MAP,
        title="Which windows need attention?",
        labels={
            "regime_shift_score": "Looks changed",
            "anomaly_score": "Looks unusual",
            "recommended_action": "Suggested action",
            "priority_size": "Review priority",
        },
    )
    fig.update_traces(
        marker={"opacity": 0.86, "line": {"width": 0.7, "color": "#1f2a33"}},
        hovertemplate=(
            "%{hovertext}<br>unusual=%{y:.3f}<br>changed=%{x:.3f}<br>"
            "review priority=%{customdata[1]:.1f}<br>result=%{customdata[2]:.3f}"
            "<br>suggestion=%{customdata[3]}<extra></extra>"
        ),
    )
    fig.add_hline(y=float(anomaly_threshold), line_dash="dot", line_color="#b84a39", opacity=0.70)
    fig.add_vline(x=float(shift_threshold), line_dash="dot", line_color="#355c7d", opacity=0.70)
    if selected_k is not None:
        selected = focus_df[focus_df["k"].astype(int) == int(selected_k)]
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected["regime_shift_score"],
                    y=selected["anomaly_score"],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 28,
                        "line": {"width": 3, "color": "#111111"},
                    },
                    name="Selected window",
                    customdata=selected[["k", "priority_index", "r_bar", "recommended_action"]].to_numpy(),
                    hovertemplate="selected k=%{customdata[0]}<br>priority=%{customdata[1]:.1f}<extra></extra>",
                )
            )
    fig.update_layout(template="plotly_white", height=380, legend_title="Suggested action")
    return fig


def _build_decision_timeline_figure(focus_df: pd.DataFrame, selected_k: Optional[int]) -> go.Figure:
    long_df = focus_df.melt(
        id_vars=["k", "window_label", "recommended_action"],
        value_vars=["risk_index", "change_index", "priority_index"],
        var_name="signal",
        value_name="score",
    )
    long_df["signal"] = long_df["signal"].map(
        {
            "risk_index": "Looks unusual",
            "change_index": "Looks changed",
            "priority_index": "Review priority",
        }
    )
    fig = px.line(
        long_df.sort_values("k"),
        x="k",
        y="score",
        color="signal",
        markers=True,
        custom_data=["k", "window_label", "recommended_action"],
        title="How the issue changes over time",
        labels={"k": "Window", "score": "Score (0-100)", "signal": "What changed"},
        color_discrete_sequence=["#b84a39", "#355c7d", "#f0a202"],
    )
    fig.update_traces(
        hovertemplate="%{customdata[1]}<br>%{legendgroup}=%{y:.1f}<br>suggestion=%{customdata[2]}<extra></extra>"
    )
    if "gt_intervention" in focus_df.columns:
        interventions = focus_df[focus_df["gt_intervention"].astype(bool)]
        if not interventions.empty:
            fig.add_trace(
                go.Scatter(
                    x=interventions["k"],
                    y=[104.0] * len(interventions),
                    mode="markers",
                    marker={"symbol": "triangle-down", "size": 12, "color": "#111111"},
                    name="Known issue",
                    customdata=interventions[["k", "window_label", "recommended_action"]].to_numpy(),
                    hovertemplate="%{customdata[1]}<br>known issue<extra></extra>",
                )
            )
    if selected_k is not None:
        fig.add_vline(x=float(selected_k), line_dash="dash", line_color="#111111", opacity=0.75)
    fig.update_layout(template="plotly_white", height=280, yaxis_range=[0, 110], legend={"orientation": "h", "y": 1.16})
    return fig


def _build_priority_queue_figure(focus_df: pd.DataFrame, top_k: int) -> go.Figure:
    view = focus_df.sort_values("priority_index", ascending=False).head(top_k).copy()
    view = view.sort_values("priority_index", ascending=True)
    fig = px.bar(
        view,
        x="priority_index",
        y="window_label",
        color="recommended_action",
        orientation="h",
        custom_data=["k", "anomaly_score", "regime_shift_score", "r_bar", "recommended_action"],
        color_discrete_map=ACTION_COLOR_MAP,
        title="What to check first",
        labels={"priority_index": "Review priority (0-100)", "window_label": "", "recommended_action": "Suggested action"},
    )
    fig.update_traces(
        hovertemplate=(
            "%{y}<br>review priority=%{x:.1f}<br>unusual=%{customdata[1]:.3f}"
            "<br>changed=%{customdata[2]:.3f}<br>result=%{customdata[3]:.3f}"
            "<br>suggestion=%{customdata[4]}<extra></extra>"
        )
    )
    fig.update_layout(template="plotly_white", height=380, xaxis_range=[0, 105], legend_title="Suggested action")
    return fig


def _filter_focus_by_action(focus_df: pd.DataFrame, action_filter: str) -> pd.DataFrame:
    if action_filter == "All actions" or action_filter not in ACTION_COLOR_MAP:
        return focus_df
    action_view = focus_df[focus_df["recommended_action"].eq(action_filter)].copy()
    return action_view if not action_view.empty else focus_df


def _build_action_mix_figure(focus_df: pd.DataFrame) -> go.Figure:
    counts = focus_df["recommended_action"].value_counts().rename_axis("recommended_action").reset_index(name="count")
    counts = counts.sort_values("count", ascending=True)
    fig = px.bar(
        counts,
        x="count",
        y="recommended_action",
        orientation="h",
        title="Recommended actions",
        color="recommended_action",
        color_discrete_map=ACTION_COLOR_MAP,
        custom_data=["recommended_action"],
        labels={"count": "Windows", "recommended_action": ""},
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<br>windows=%{x}<extra></extra>",
    )
    fig.update_layout(template="plotly_white", height=175, legend_title="", showlegend=False, margin={"t": 48, "b": 18})
    return fig


def _build_benchmark_heatmap(benchmark_df: pd.DataFrame, metric: str) -> go.Figure:
    if benchmark_df.empty:
        return go.Figure()
    frame = benchmark_df.copy()
    frame["system"] = frame["env"].map(_format_env_name)
    frame["controller"] = frame["policy"].map(_format_policy_name)
    pivot = frame.pivot_table(index="system", columns="controller", values="{0}_mean".format(metric), aggfunc="mean")
    customdata: List[List[List[str]]] = []
    for system_label in pivot.index.tolist():
        env_key = _env_from_display(system_label) or str(system_label)
        row: List[List[str]] = []
        for controller_label in pivot.columns.tolist():
            policy_key = _policy_from_display(str(controller_label)) or str(controller_label)
            row.append([env_key, policy_key, metric])
        customdata.append(row)
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            customdata=customdata,
            colorscale="Tealgrn",
            text=[[("{0:.3f}".format(value) if pd.notna(value) else "") for value in row] for row in pivot.to_numpy()],
            texttemplate="%{text}",
            hovertemplate="scenario=%{y}<br>controller=%{x}<br>score=%{z:.3f}<extra></extra>",
            colorbar={"title": _metric_label(metric)},
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Which setup works best?",
        xaxis_title="Controller",
        yaxis_title="Scenario",
        height=330,
    )
    return fig


def _build_env_policy_metric_figure(benchmark_df: pd.DataFrame, env_name: str) -> go.Figure:
    if benchmark_df.empty:
        return go.Figure()
    frame = benchmark_df[benchmark_df["env"] == env_name].copy()
    if frame.empty:
        return go.Figure()
    frame["controller"] = frame["policy"].map(_format_policy_name)
    long_df = frame.melt(
        id_vars=["policy", "controller"],
        value_vars=["anomaly_auc_mean", "shift_auc_mean", "reward_jump_auc_mean"],
        var_name="metric_key",
        value_name="score",
    )
    metric_map = {
        "anomaly_auc_mean": "Find unusual windows",
        "shift_auc_mean": "Find changes",
        "reward_jump_auc_mean": "Find result drops",
    }
    long_df["metric"] = long_df["metric_key"].map(metric_map)
    long_df["metric_state"] = long_df["metric_key"].str.replace("_mean", "", regex=False)
    fig = px.bar(
        long_df,
        x="controller",
        y="score",
        color="metric",
        barmode="group",
        custom_data=["policy", "metric_state"],
        title="{0}: controller checkup".format(_format_env_name(env_name)),
        labels={"controller": "Controller", "score": "Score", "metric": "What it finds"},
        color_discrete_sequence=["#b84a39", "#355c7d", "#f0a202"],
    )
    fig.update_traces(hovertemplate="%{x}<br>%{legendgroup}=%{y:.3f}<extra></extra>")
    fig.update_layout(template="plotly_white", height=330, legend_title="")
    return fig


def _build_feature_signal_figure(focus_df: pd.DataFrame, selected_k: Optional[int]) -> go.Figure:
    active = _focused_action_row(focus_df, selected_k)
    metrics = pd.DataFrame(
        {
            "signal": ["Unusual", "Changed", "Poor result", "Priority"],
            "signal_key": ["risk", "change", "outcome", "priority"],
            "score": [
                float(active["risk_index"]),
                float(active["change_index"]),
                float(active["outcome_stress_index"]),
                float(active["priority_index"]),
            ],
        }
    )
    fig = px.bar(
        metrics,
        x="signal",
        y="score",
        color="signal",
        custom_data=["signal_key"],
        title="Why this window matters",
        labels={"signal": "", "score": "0-100"},
        color_discrete_sequence=["#b84a39", "#355c7d", "#6c8ebf", "#f0a202"],
    )
    fig.update_traces(hovertemplate="%{x}<br>score=%{y:.1f}<extra></extra>")
    fig.update_layout(template="plotly_white", height=205, showlegend=False, yaxis_range=[0, 105], margin={"t": 48, "b": 24})
    return fig


def _render_visual_decision_cockpit(
    filtered_df: pd.DataFrame,
    selected_k: Optional[int],
    env_name: str,
    policy: str,
    task_name: str,
    decision: Dict[str, Any],
    readiness: Dict[str, Any],
    anomaly_threshold: float,
    shift_threshold: float,
    top_k_windows: int,
    action_filter: str,
) -> Optional[int]:
    focus_df = _operational_focus_frame(filtered_df, selected_k)
    action_mix_df = focus_df.copy()
    focus_df = _filter_focus_by_action(focus_df, action_filter)
    active_row = _focused_action_row(focus_df, selected_k)
    st.markdown('<div class="section-chip">Issue Review</div>', unsafe_allow_html=True)
    st.subheader("Review the current issue")
    st.markdown(
        """
        <div class="decision-strip">
            <h3>Suggested action: {action}</h3>
            <p>{decision}</p>
        </div>
        """.format(action=str(active_row["recommended_action"]), decision=decision["decision"]),
        unsafe_allow_html=True,
    )
    cockpit_cols = st.columns(4)
    cockpit_cols[0].metric("Selected window", _window_label(active_row))
    cockpit_cols[1].metric("Review priority", "{0:.1f}/100".format(float(active_row["priority_index"])))
    cockpit_cols[2].metric("Urgency", decision["severity"])
    cockpit_cols[3].metric("Evidence", readiness["level"], "{0}/{1} checks".format(readiness["score"], readiness["total"]))

    picked_k: Optional[int] = None
    chart_cols = st.columns([1.25, 1.05, 0.9])
    with chart_cols[0]:
        matrix_selection = _plotly_chart_with_optional_selection(
            _build_risk_change_matrix(
                focus_df=focus_df,
                selected_k=selected_k,
                anomaly_threshold=anomaly_threshold,
                shift_threshold=shift_threshold,
            ),
            key="cockpit_risk_change_matrix",
            use_container_width=True,
        )
        picked_k = _selected_window(matrix_selection)
        if picked_k is not None and picked_k in focus_df["k"].astype(int).tolist():
            point_row = focus_df[focus_df["k"].astype(int) == int(picked_k)].iloc[0]
            _set_window_point_explanation(
                chart="Which windows need attention?",
                env_name=env_name,
                policy=policy,
                task_name=task_name,
                row=point_row,
            )
    with chart_cols[1]:
        queue_selection = _plotly_chart_with_optional_selection(
            _build_priority_queue_figure(focus_df, top_k=min(top_k_windows, len(focus_df))),
            key="cockpit_priority_queue",
            use_container_width=True,
        )
        queue_k = _selected_window(queue_selection)
        if queue_k is not None:
            picked_k = queue_k
            if queue_k in focus_df["k"].astype(int).tolist():
                point_row = focus_df[focus_df["k"].astype(int) == int(queue_k)].iloc[0]
                _set_window_point_explanation(
                    chart="What to check first",
                    env_name=env_name,
                    policy=policy,
                    task_name=task_name,
                    row=point_row,
                )
    with chart_cols[2]:
        action_selection = _plotly_chart_with_optional_selection(
            _build_action_mix_figure(action_mix_df),
            key="cockpit_action_mix",
            use_container_width=True,
        )
        selected_action = None
        action_custom = _selected_customdata(action_selection)
        if action_custom:
            selected_action = str(action_custom[0])
        if selected_action is None:
            selected_action = _selected_label(action_selection)
        if selected_action in ACTION_COLOR_MAP and selected_action != st.session_state.get("multi_action_filter"):
            _set_action_mix_explanation(selected_action, action_mix_df)
            _queue_multi_dashboard_update(multi_action_filter=selected_action)
        elif selected_action in ACTION_COLOR_MAP:
            _set_action_mix_explanation(selected_action, action_mix_df)
        signal_selection = _plotly_chart_with_optional_selection(
            _build_feature_signal_figure(focus_df, selected_k),
            key="cockpit_feature_signal",
            use_container_width=True,
        )
        signal_custom = _selected_customdata(signal_selection)
        if signal_custom:
            signal_key = str(signal_custom[0])
            signal_row = _focused_action_row(focus_df, selected_k)
            _set_signal_explanation(signal_key, env_name, policy, signal_row)
            updates: Dict[str, Any] = {}
            if signal_key == "risk":
                updates = {"multi_task": "Find anomalies", "multi_color_by": "anomaly_score"}
            elif signal_key == "change":
                updates = {"multi_task": "Locate shift", "multi_color_by": "regime_shift_score"}
            elif signal_key == "outcome":
                updates = {"multi_color_by": "r_bar"}
            elif signal_key == "priority":
                updates = {"multi_color_by": "anomaly_score"}
            if updates and any(st.session_state.get(key) != value for key, value in updates.items()):
                _queue_multi_dashboard_update(**updates)

    timeline_selection = _plotly_chart_with_optional_selection(
        _build_decision_timeline_figure(focus_df, selected_k=selected_k),
        key="cockpit_decision_timeline",
        use_container_width=True,
    )
    timeline_k = _selected_window(timeline_selection)
    if timeline_k is not None:
        picked_k = timeline_k
        if timeline_k in focus_df["k"].astype(int).tolist():
            point_row = focus_df[focus_df["k"].astype(int) == int(timeline_k)].iloc[0]
            _set_window_point_explanation(
                chart="How the issue changes over time",
                env_name=env_name,
                policy=policy,
                task_name=task_name,
                row=point_row,
            )

    if picked_k is not None and picked_k in focus_df["k"].astype(int).tolist():
        return int(picked_k)
    return None


def _render_multi_system_dashboard() -> None:
    _apply_pending_multi_dashboard_update()
    st.session_state.setdefault("multi_env", "traffic")
    st.session_state.setdefault("multi_policy", "dqn")
    st.session_state.setdefault("multi_task", "Find anomalies")
    st.session_state.setdefault("multi_color_by", "anomaly_score")
    st.session_state.setdefault("multi_action_filter", "All actions")
    st.session_state.setdefault("selected_k", None)
    st.session_state.setdefault("selected_cluster", None)

    st.markdown(
        """
    <div class="hero-shell">
        <div class="hero-kicker">5-System User Dashboard</div>
        <div class="hero-title">Decision System Review Studio</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    system_cols = st.columns(len(ENVIRONMENTS))
    for idx, env_option in enumerate(ENVIRONMENTS):
        profile = _scenario_profile(env_option)
        button_type = "primary" if st.session_state.get("multi_env") == env_option else "secondary"
        system_cols[idx].button(
            profile["short_label"],
            key="system_pick_{0}".format(env_option),
            type=button_type,
            use_container_width=True,
            on_click=_apply_system_preset,
            args=(env_option,),
        )

    control_cols = st.columns([1.25, 1.1, 0.9, 1.25, 1.1, 1.1])
    with control_cols[0]:
        env_name = st.selectbox(
            "Scenario",
            options=ENVIRONMENTS,
            key="multi_env",
            format_func=lambda name: _scenario_profile(name)["short_label"],
        )
    profile = _scenario_profile(env_name)
    with control_cols[1]:
        if st.session_state.get("multi_policy") not in BENCHMARK_POLICIES:
            st.session_state["multi_policy"] = "dqn"
        policy = st.selectbox("Controller", options=BENCHMARK_POLICIES, key="multi_policy", format_func=_format_policy_name)
    available_seeds = cached_available_seeds(env_name, policy)
    if available_seeds and st.session_state.get("multi_seed") not in available_seeds:
        st.session_state["multi_seed"] = 23 if 23 in available_seeds else int(available_seeds[0])
    with control_cols[2]:
        selected_seed = st.selectbox("Data run", options=available_seeds, key="multi_seed") if available_seeds else None
    with control_cols[3]:
        task_name = st.selectbox(
            "Goal",
            options=["Find anomalies", "Locate shift", "Compare policies"],
            key="multi_task",
            format_func=_user_task_label,
        )
    with control_cols[4]:
        focus_preset = st.selectbox(
            "View",
            options=["Balanced", "Highest risk", "Change", "Task only"],
            key="multi_focus_preset",
            format_func=_focus_label,
        )
    with control_cols[5]:
        score_metric = st.selectbox(
            "Score view",
            options=["anomaly_auc", "shift_auc", "reward_jump_auc"],
            key="multi_score_metric",
            format_func=_metric_label,
        )

    summary_df = cached_summary(env_name, policy, seed=selected_seed).sort_values("k").reset_index(drop=True)
    if summary_df.empty:
        st.warning("No windows are available for this system/controller/run.")
        return

    k_min, k_max = int(summary_df["k"].min()), int(summary_df["k"].max())
    anomaly_default = float(summary_df["anomaly_score"].quantile(0.45))
    shift_default = float(summary_df["regime_shift_score"].quantile(0.45))
    if focus_preset == "Highest risk":
        anomaly_default = float(summary_df["anomaly_score"].quantile(0.72))
        shift_default = float(summary_df["regime_shift_score"].quantile(0.35))
    elif focus_preset == "Change":
        anomaly_default = float(summary_df["anomaly_score"].quantile(0.25))
        shift_default = float(summary_df["regime_shift_score"].quantile(0.72))
    elif focus_preset == "Task only":
        if task_name == "Locate shift":
            anomaly_default = float(summary_df["anomaly_score"].quantile(0.20))
            shift_default = float(summary_df["regime_shift_score"].quantile(0.75))
        elif task_name == "Find anomalies":
            anomaly_default = float(summary_df["anomaly_score"].quantile(0.75))
            shift_default = float(summary_df["regime_shift_score"].quantile(0.20))

    filter_cols = st.columns([1.25, 0.9, 1, 1, 1])
    with filter_cols[0]:
        k_range = st.slider(
            "Windows",
            min_value=k_min,
            max_value=k_max,
            value=(k_min, k_max),
            key="range_{0}_{1}_{2}".format(env_name, policy, selected_seed),
        )
    with filter_cols[1]:
        top_k_windows = st.slider("Top items", min_value=3, max_value=min(20, len(summary_df)), value=min(8, len(summary_df)))
    with filter_cols[2]:
        action_filter = st.selectbox(
            "Suggested action",
            options=["All actions"] + list(ACTION_COLOR_MAP.keys()),
            key="multi_action_filter",
        )
    with filter_cols[3]:
        anomaly_threshold = st.slider(
            "Unusual threshold",
            min_value=float(summary_df["anomaly_score"].min()),
            max_value=float(summary_df["anomaly_score"].max()),
            value=anomaly_default,
            key="risk_{0}_{1}_{2}_{3}".format(env_name, policy, selected_seed, focus_preset),
        )
    with filter_cols[4]:
        shift_threshold = st.slider(
            "Change threshold",
            min_value=float(summary_df["regime_shift_score"].min()),
            max_value=float(summary_df["regime_shift_score"].max()),
            value=shift_default,
            key="change_{0}_{1}_{2}_{3}".format(env_name, policy, selected_seed, focus_preset),
        )

    filtered_df = summary_df[summary_df["k"].between(k_range[0], k_range[1])].copy()
    filtered_df = filtered_df[
        filtered_df["anomaly_score"].ge(float(anomaly_threshold)) | filtered_df["regime_shift_score"].ge(float(shift_threshold))
    ].copy()
    if filtered_df.empty:
        st.warning("No windows match the current thresholds.")
        return

    top_window = _focus_selected_window(filtered_df, task_name)
    if st.session_state.get("selected_k") not in filtered_df["k"].astype(int).tolist():
        st.session_state["selected_k"] = int(top_window["k"])
    selected_window = filtered_df[filtered_df["k"] == int(st.session_state["selected_k"])].iloc[0]
    action_focus_df = _filter_focus_by_action(_operational_focus_frame(filtered_df, int(selected_window["k"])), action_filter)
    if action_filter != "All actions" and int(selected_window["k"]) not in action_focus_df["k"].astype(int).tolist():
        action_top = action_focus_df.sort_values("priority_index", ascending=False).iloc[0]
        st.session_state["selected_k"] = int(action_top["k"])
        selected_window = filtered_df[filtered_df["k"] == int(st.session_state["selected_k"])].iloc[0]

    default_p1, default_p2 = _recommended_compare_pair(task_name, policy)
    p1 = default_p1 if default_p1 in BENCHMARK_POLICIES else policy
    compare_options = [name for name in BENCHMARK_POLICIES if name != p1]
    compare_to = default_p2 if default_p2 in compare_options else compare_options[0]
    compare_df = cached_compare(env_name=env_name, p1=p1, p2=compare_to, k_clusters=8, seed=selected_seed)
    compare_df = compare_df[compare_df["js_div"].notna()].copy()
    compare_row = _top_cluster_finding(compare_df)

    decision = _system_action_summary(
        env_name=env_name,
        task_name=task_name,
        policy=policy,
        selected_seed=selected_seed,
        window_row=selected_window,
        compare_row=compare_row,
    )
    readiness = _decision_readiness(task_name=task_name, filtered_df=filtered_df, window_row=selected_window, compare_row=compare_row)
    focus_df = _filter_focus_by_action(_operational_focus_frame(filtered_df, int(selected_window["k"])), action_filter)
    active_row = _focused_action_row(focus_df, int(selected_window["k"]))

    user_cols = st.columns([1.1, 1.5, 1.2, 1.2])
    user_cols[0].metric("Role", profile["user"])
    user_cols[1].metric("Scenario", profile["short_label"])
    user_cols[1].caption(profile["problem"])
    user_cols[2].metric("Suggested action", str(active_row["recommended_action"]))
    user_cols[3].metric("Owner", decision["owner"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Windows", int(len(filtered_df)))
    metric_cols[1].metric("Selected window", int(selected_window["k"]))
    metric_cols[2].metric("Unusual", "{0:.3f}".format(float(selected_window["anomaly_score"])))
    metric_cols[3].metric("Changed", "{0:.3f}".format(float(selected_window["regime_shift_score"])))
    metric_cols[4].metric("Priority", "{0:.1f}/100".format(float(active_row["priority_index"])))

    cockpit_k = _render_visual_decision_cockpit(
        filtered_df=filtered_df,
        selected_k=int(selected_window["k"]),
        env_name=env_name,
        policy=policy,
        task_name=task_name,
        decision=decision,
        readiness=readiness,
        anomaly_threshold=float(anomaly_threshold),
        shift_threshold=float(shift_threshold),
        top_k_windows=int(top_k_windows),
        action_filter=action_filter,
    )
    if cockpit_k is not None and cockpit_k != int(selected_window["k"]):
        st.session_state["selected_k"] = int(cockpit_k)
        selected_window = filtered_df[filtered_df["k"] == int(cockpit_k)].iloc[0]
        focus_df = _filter_focus_by_action(_operational_focus_frame(filtered_df, int(selected_window["k"])), action_filter)

    behavior_fig = build_behavior_space_figure(
        filtered_df,
        env_name=env_name,
        color_by=st.session_state.get("multi_color_by", "anomaly_score"),
    )
    behavior_fig.update_layout(
        title="Windows with similar behavior",
        height=340,
        xaxis_title="",
        yaxis_title="",
        margin={"t": 52, "b": 20},
    )
    behavior_fig.update_traces(
        hovertemplate=(
            "window=%{customdata[0]}<br>unusual=%{customdata[1]:.3f}"
            "<br>changed=%{customdata[2]:.3f}<br>result=%{customdata[4]:.3f}<extra></extra>"
        )
    )
    behavior_selection = _plotly_chart_with_optional_selection(
        behavior_fig,
        key="multi_behavior_space",
        use_container_width=True,
    )
    picked_k = _selected_window(behavior_selection)
    if picked_k is not None and picked_k in filtered_df["k"].astype(int).tolist():
        point_focus_df = _operational_focus_frame(filtered_df, int(picked_k))
        point_row = point_focus_df[point_focus_df["k"].astype(int) == int(picked_k)].iloc[0]
        _set_window_point_explanation(
            chart="Windows with similar behavior",
            env_name=env_name,
            policy=policy,
            task_name=task_name,
            row=point_row,
        )
        if int(picked_k) != int(st.session_state["selected_k"]) or st.session_state.get("multi_action_filter") != "All actions":
            _queue_multi_dashboard_update(selected_k=int(picked_k), multi_action_filter="All actions")

    benchmark_df = cached_report_csv("benchmark_table.csv")
    score_cols = st.columns(2)
    with score_cols[0]:
        heatmap_selection = _plotly_chart_with_optional_selection(
            _build_benchmark_heatmap(benchmark_df, score_metric),
            key="multi_score_heatmap",
            use_container_width=True,
        )
        heatmap_custom = _selected_customdata(heatmap_selection)
        if len(heatmap_custom) >= 2:
            selected_env = str(heatmap_custom[0])
            selected_policy = str(heatmap_custom[1])
            if selected_env in ENVIRONMENTS and selected_policy in BENCHMARK_POLICIES:
                _set_benchmark_cell_explanation(
                    benchmark_df=benchmark_df,
                    env_name=selected_env,
                    policy=selected_policy,
                    metric=score_metric,
                    chart="5-System Score Map",
                )
                heatmap_changed = (
                    st.session_state.get("multi_env") != selected_env
                    or st.session_state.get("multi_policy") != selected_policy
                )
                if heatmap_changed:
                    seeds = cached_available_seeds(selected_env, selected_policy)
                    _queue_multi_dashboard_update(
                        multi_env=selected_env,
                        multi_policy=selected_policy,
                        multi_action_filter="All actions",
                        selected_k=None,
                        multi_seed=23 if 23 in seeds else (int(seeds[0]) if seeds else None),
                    )
    with score_cols[1]:
        scoreboard_selection = _plotly_chart_with_optional_selection(
            _build_env_policy_metric_figure(benchmark_df, env_name),
            key="multi_controller_scoreboard",
            use_container_width=True,
        )
        scoreboard_custom = _selected_customdata(scoreboard_selection)
        if len(scoreboard_custom) >= 2:
            selected_policy = str(scoreboard_custom[0])
            selected_metric = str(scoreboard_custom[1])
            if selected_policy in BENCHMARK_POLICIES:
                if selected_metric in {"anomaly_auc", "shift_auc", "reward_jump_auc"}:
                    _set_benchmark_cell_explanation(
                        benchmark_df=benchmark_df,
                        env_name=env_name,
                        policy=selected_policy,
                        metric=selected_metric,
                        chart="{0}: Controller Scoreboard".format(_format_env_name(env_name)),
                    )
                scoreboard_changed = st.session_state.get("multi_policy") != selected_policy
                metric_changed = selected_metric in {"anomaly_auc", "shift_auc", "reward_jump_auc"} and (
                    st.session_state.get("multi_score_metric") != selected_metric
                )
                if scoreboard_changed or metric_changed:
                    updates = {
                        "multi_policy": selected_policy,
                        "multi_action_filter": "All actions",
                    }
                    if selected_metric in {"anomaly_auc", "shift_auc", "reward_jump_auc"}:
                        updates["multi_score_metric"] = selected_metric
                    if scoreboard_changed:
                        seeds = cached_available_seeds(env_name, selected_policy)
                        updates["selected_k"] = None
                        updates["multi_seed"] = 23 if 23 in seeds else (int(seeds[0]) if seeds else None)
                    _queue_multi_dashboard_update(**updates)

    st.markdown('<div class="section-chip">Controller Comparison</div>', unsafe_allow_html=True)
    cmp_cols = st.columns([1, 1, 1, 1])
    with cmp_cols[0]:
        p1 = st.selectbox("Controller A", options=BENCHMARK_POLICIES, index=_control_default(BENCHMARK_POLICIES, policy), key="multi_cmp_p1", format_func=_format_policy_name)
    p2_options = [name for name in BENCHMARK_POLICIES if name != p1]
    with cmp_cols[1]:
        p2 = st.selectbox("Controller B", options=p2_options, index=_control_default(p2_options, compare_to), key="multi_cmp_p2", format_func=_format_policy_name)
    with cmp_cols[2]:
        k_clusters = st.slider("Groups", min_value=3, max_value=20, value=8, key="multi_cmp_clusters")
    with cmp_cols[3]:
        compare_top_k = st.slider("Show", min_value=3, max_value=12, value=8, key="multi_cmp_top_k")

    compare_df = cached_compare(env_name=env_name, p1=p1, p2=p2, k_clusters=k_clusters, seed=selected_seed)
    compare_df = compare_df[compare_df["js_div"].notna()].copy()
    if not compare_df.empty:
        top_df = compare_df.sort_values("js_div", ascending=False).head(compare_top_k).copy()
        top_df["behavior_group"] = top_df["cluster"].astype(int).astype(str)
        selected_compare_cluster: Optional[int] = None
        selected_compare_chart = "Behavior Difference"
        compare_cols = st.columns([1.15, 1])
        with compare_cols[0]:
            compare_bar = px.bar(
                top_df,
                x="behavior_group",
                y="js_div",
                color="reward_gap",
                custom_data=["cluster", "reward_gap", "n_total"],
                title="Where controllers differ",
                labels={"behavior_group": "Group", "js_div": "How different", "reward_gap": "Result impact"},
                color_continuous_scale="Tealgrn",
            )
            compare_bar.update_traces(
                hovertemplate="group=%{x}<br>difference=%{y:.3f}<br>result impact=%{customdata[1]:.3f}<br>windows=%{customdata[2]}<extra></extra>"
            )
            compare_bar.update_layout(template="plotly_white", height=330)
            compare_selection = _plotly_chart_with_optional_selection(
                compare_bar,
                key="multi_comparison_bar",
                use_container_width=True,
            )
            selected_compare_cluster = _selected_cluster(compare_selection)
        with compare_cols[1]:
            frontier = px.scatter(
                top_df,
                x="reward_gap",
                y="js_div",
                size="n_total",
                color="cluster",
                custom_data=["cluster", "reward_gap", "js_div", "n_total"],
                title="Difference vs result impact",
                labels={"reward_gap": "Result impact", "js_div": "How different"},
            )
            frontier.update_traces(
                hovertemplate="group=%{customdata[0]}<br>result impact=%{customdata[1]:.3f}<br>difference=%{customdata[2]:.3f}<br>windows=%{customdata[3]}<extra></extra>"
            )
            frontier.update_layout(template="plotly_white", height=330, legend_title="Group")
            frontier_selection = _plotly_chart_with_optional_selection(
                frontier,
                key="multi_comparison_frontier",
                use_container_width=True,
            )
            frontier_custom = _selected_customdata(frontier_selection)
            if frontier_custom:
                try:
                    selected_compare_cluster = int(frontier_custom[0])
                    selected_compare_chart = "Outcome Gap vs Behavior Difference"
                except (TypeError, ValueError):
                    pass
        if selected_compare_cluster is not None and selected_compare_cluster in compare_df["cluster"].astype(int).tolist():
            selected_cluster_row = compare_df[compare_df["cluster"].astype(int) == int(selected_compare_cluster)].iloc[0]
            _set_cluster_point_explanation(env_name, p1, p2, selected_cluster_row, selected_compare_chart)
            st.session_state["selected_cluster"] = int(selected_compare_cluster)
        if st.session_state.get("selected_cluster") in compare_df["cluster"].astype(int).tolist():
            cluster_row = compare_df[compare_df["cluster"].astype(int) == int(st.session_state["selected_cluster"])].iloc[0]
            group_cols = st.columns(4)
            group_cols[0].metric("Selected group", int(cluster_row["cluster"]))
            group_cols[1].metric("How different", "{0:.3f}".format(float(cluster_row["js_div"])))
            group_cols[2].metric("Result impact", "{0:.3f}".format(float(cluster_row["reward_gap"])))
            group_cols[3].metric("Windows", int(cluster_row["n_total"]))
    else:
        st.info("No comparison groups are available for this pair.")

    st.markdown('<div class="section-chip">Save / Export</div>', unsafe_allow_html=True)
    export_cols = st.columns([1.2, 1.2, 1])
    with export_cols[0]:
        st.session_state.setdefault("case_note_title", "")
        note_title = st.text_input("Case title", key="case_note_title", placeholder="Short case name")
    with export_cols[1]:
        st.session_state.setdefault("case_note_status", "Needs review")
        note_status = st.selectbox("Status", options=USER_CASE_STATUS_OPTIONS, key="case_note_status")
    with export_cols[2]:
        note_owner = st.text_input("Owner", value=decision["owner"], key="case_note_owner_multi")
    note_body = st.text_area("Note", height=90, placeholder="What should the user do next?", key="case_note_body_multi")
    save_cols = st.columns(3)
    if save_cols[0].button("Save case", use_container_width=True):
        question_frame = _task_question_rows(
            task_name=task_name,
            role_name=profile["user"],
            env_name=env_name,
            policy=policy,
            filtered_df=filtered_df,
            window_row=selected_window,
            compare_row=compare_row,
        )
        entry = _compose_case_note_entry(
            workflow_name=profile["short_label"],
            role_name=profile["user"],
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
            note_tags=["High-risk system behavior"] if decision["severity"] == "High" else ["Follow-up required"],
            compare_row=compare_row,
            question_frame=question_frame,
        )
        _save_case_notebook_entry(entry)
        st.success("Saved.")
    memo = _build_review_memo_markdown(
        role_name=profile["user"],
        workflow_name=profile["short_label"],
        env_name=env_name,
        policy=policy,
        selected_seed=selected_seed,
        task_name=task_name,
        window_row=selected_window,
        decision=decision,
        note_title=note_title,
        note_body=note_body,
        question_frame=_task_question_rows(
            task_name=task_name,
            role_name=profile["user"],
            env_name=env_name,
            policy=policy,
            filtered_df=filtered_df,
            window_row=selected_window,
            compare_row=compare_row,
        ),
    )
    save_cols[1].download_button(
        "Download memo",
        data=memo.encode("utf-8"),
        file_name="{0}_{1}_memo.md".format(env_name, policy),
        mime="text/markdown",
        use_container_width=True,
    )
    save_cols[2].download_button(
        "Download windows CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="{0}_{1}_windows.csv".format(env_name, policy),
        mime="text/csv",
        use_container_width=True,
    )
    _render_point_explanation_module()


def main() -> None:
    st.set_page_config(page_title=PRIMARY_PRODUCT_NAME, layout="wide")
    _inject_dashboard_css()
    _render_multi_system_dashboard()


if __name__ == "__main__":
    main()
