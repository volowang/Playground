from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"
TRACE_DIR = OUTPUTS_DIR / "traces"
SUMMARY_DIR = OUTPUTS_DIR / "summaries"
TABLE_DIR = OUTPUTS_DIR / "tables"
FIG_DIR = OUTPUTS_DIR / "figs"
REPORT_ASSET_DIR = OUTPUTS_DIR / "report_assets"
MODEL_DIR = OUTPUTS_DIR / "models"
BENCHMARK_DIR = OUTPUTS_DIR / "benchmark"
BENCHMARK_TRACE_DIR = BENCHMARK_DIR / "traces"
BENCHMARK_SUMMARY_DIR = BENCHMARK_DIR / "summaries"
BENCHMARK_REPORT_DIR = BENCHMARK_DIR / "reports"
BENCHMARK_MODEL_DIR = BENCHMARK_DIR / "models"
COURSE_DATASET_DIR = OUTPUTS_DIR / "course_dataset"
COURSE_DATASET_REPORT_DIR = COURSE_DATASET_DIR / "reports"
COURSE_DATASET_FILENAME = "rlva_trace_corpus_close_to_1gb.csv"
COURSE_DATASET_PATH = COURSE_DATASET_DIR / COURSE_DATASET_FILENAME
COURSE_DATASET_ARCHIVE_PATH = COURSE_DATASET_DIR / "{0}.gz".format(COURSE_DATASET_FILENAME)
COURSE_DATASET_ARCHIVE_PART_PREFIX = "{0}.gz.part-".format(COURSE_DATASET_FILENAME)
COURSE_DATASET_METADATA_PATH = COURSE_DATASET_REPORT_DIR / "large_dataset_metadata.json"
COURSE_INTERACTIVITY_METRICS_PATH = BENCHMARK_REPORT_DIR / "course_interactivity_metrics.json"
COURSE_INTERACTIVITY_REPORT_PATH = BENCHMARK_REPORT_DIR / "course_interactivity_report.md"
COURSE_TASK_PERFORMANCE_METRICS_PATH = BENCHMARK_REPORT_DIR / "course_task_performance_metrics.json"
COURSE_TASK_PERFORMANCE_REPORT_PATH = BENCHMARK_REPORT_DIR / "course_task_performance_report.md"
COURSE_SUBMISSION_DIR = BENCHMARK_REPORT_DIR / "cs526_submission_package"
USER_CASE_NOTEBOOK_PATH = BENCHMARK_REPORT_DIR / "user_case_notebook.jsonl"
USER_TASK_LOG_PATH = BENCHMARK_REPORT_DIR / "user_task_log.jsonl"

ENVIRONMENTS: List[str] = ["queue", "inventory", "traffic", "cartpole", "lunarlander"]
POLICIES: List[str] = ["pg", "random", "heuristic"]
BENCHMARK_POLICIES: List[str] = ["pg", "ppo", "dqn", "random", "heuristic"]
LEARNED_POLICIES: List[str] = ["pg", "ppo", "dqn"]
PAPER_ENVIRONMENTS: List[str] = ["queue", "inventory", "traffic", "cartpole", "lunarlander"]
PAPER_POLICY_FAMILIES: List[str] = ["pg", "ppo", "dqn", "random", "heuristic"]
HEURISTIC_POLICY = "heuristic"

SUMMARY_FEATURE_COLUMNS: List[str] = [
    "q_bar",
    "mu_bar",
    "sigma_bar",
    "rho_bar",
    "freq0",
    "freq1",
    "freq2",
    "r_bar",
    "H_bar",
    "switch_rate",
    "state_delta_bar",
    "state_delta_std",
    "r_std",
    "reward_trend",
    "dominance_gap",
]

ANALYSIS_FEATURE_COLUMNS: List[str] = SUMMARY_FEATURE_COLUMNS + [
    "anomaly_score",
    "regime_shift_score",
]

ENV_LABELS: Dict[str, str] = {
    "queue": "Queue Scheduling",
    "inventory": "Inventory Control",
    "traffic": "Traffic Signal Control",
    "cartpole": "CartPole Control",
    "lunarlander": "LunarLander Control",
}


def ensure_outputs() -> None:
    for path in [
        OUTPUTS_DIR,
        TRACE_DIR,
        SUMMARY_DIR,
        TABLE_DIR,
        FIG_DIR,
        REPORT_ASSET_DIR,
        MODEL_DIR,
        BENCHMARK_DIR,
        BENCHMARK_MODEL_DIR,
        BENCHMARK_TRACE_DIR,
        BENCHMARK_SUMMARY_DIR,
        BENCHMARK_REPORT_DIR,
        COURSE_DATASET_DIR,
        COURSE_DATASET_REPORT_DIR,
        COURSE_SUBMISSION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def course_dataset_archive_parts() -> List[Path]:
    return sorted(COURSE_DATASET_DIR.glob("{0}*".format(COURSE_DATASET_ARCHIVE_PART_PREFIX)))


def ensure_course_dataset_archive() -> Path:
    if COURSE_DATASET_ARCHIVE_PATH.exists():
        return COURSE_DATASET_ARCHIVE_PATH

    parts = course_dataset_archive_parts()
    if not parts:
        return COURSE_DATASET_PATH

    COURSE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with COURSE_DATASET_ARCHIVE_PATH.open("wb") as handle:
        for part_path in parts:
            handle.write(part_path.read_bytes())
    return COURSE_DATASET_ARCHIVE_PATH


def resolve_course_dataset_path() -> Path:
    if COURSE_DATASET_PATH.exists():
        return COURSE_DATASET_PATH
    if COURSE_DATASET_ARCHIVE_PATH.exists():
        return COURSE_DATASET_ARCHIVE_PATH
    return ensure_course_dataset_archive()


def trace_filename(env_name: str, policy: str) -> str:
    return "{0}__{1}.csv".format(env_name, policy)


def summary_filename(env_name: str, policy: str) -> str:
    return "{0}__{1}_summary.csv".format(env_name, policy)


def model_filename(env_name: str) -> str:
    return "{0}__pg_linear.pt".format(env_name)


def benchmark_model_filename(env_name: str, policy: str, seed: int) -> str:
    return "{0}__{1}__seed_{2}.pt".format(env_name, policy, seed)


def comparison_stem(env_name: str, p1: str, p2: str) -> str:
    return "{0}__{1}_vs_{2}".format(env_name, p1, p2)


def resolve_trace_path(env_name: str, policy: str, trace_dir: Path = TRACE_DIR) -> Path:
    return trace_dir / trace_filename(env_name=env_name, policy=policy)


def resolve_summary_path(env_name: str, policy: str, summary_dir: Path = SUMMARY_DIR) -> Path:
    return summary_dir / summary_filename(env_name=env_name, policy=policy)


def resolve_model_path(env_name: str, model_dir: Path = MODEL_DIR) -> Path:
    return model_dir / model_filename(env_name=env_name)


def resolve_benchmark_model_path(env_name: str, policy: str, seed: int, model_dir: Path = BENCHMARK_MODEL_DIR) -> Path:
    return model_dir / env_name / policy / benchmark_model_filename(env_name=env_name, policy=policy, seed=seed)
