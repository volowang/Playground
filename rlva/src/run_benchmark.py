from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from rlva.src.behavior import summarize_trace
from rlva.src.config import (
    BENCHMARK_MODEL_DIR,
    BENCHMARK_POLICIES,
    BENCHMARK_SUMMARY_DIR,
    BENCHMARK_TRACE_DIR,
    PAPER_ENVIRONMENTS,
    LEARNED_POLICIES,
    resolve_benchmark_model_path,
)
from rlva.src.env import is_env_available, make_env
from rlva.src.learned_policy import LearnedPolicy
from rlva.src.policies import HeuristicPolicy, RandomPolicy
from rlva.src.trace import run_and_record


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _heuristic_mode_for_env(env_name: str) -> str:
    if env_name == "traffic":
        return "a2"
    if env_name == "lunarlander":
        return "a1"
    return "a0"


def _build_policy(policy_name: str, env_name: str, seed: int, model_dir: Path):
    if policy_name == "random":
        return RandomPolicy(seed=seed + 101)
    if policy_name == "heuristic":
        return HeuristicPolicy(mode=_heuristic_mode_for_env(env_name))
    if policy_name in LEARNED_POLICIES:
        model_path = resolve_benchmark_model_path(env_name=env_name, policy=policy_name, seed=seed, model_dir=model_dir)
        if not model_path.exists():
            raise FileNotFoundError("Missing trained model for env={0} policy={1} seed={2}: {3}".format(env_name, policy_name, seed, model_path))
        return LearnedPolicy(model_path=model_path, seed=seed + 202)
    raise ValueError("Unknown policy: {0}".format(policy_name))


def _trace_out_path(root: Path, env_name: str, policy_name: str, seed: int) -> Path:
    return root / env_name / policy_name / "seed_{0}.csv".format(seed)


def _summary_out_path(root: Path, env_name: str, policy_name: str, seed: int) -> Path:
    return root / env_name / policy_name / "seed_{0}.csv".format(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed RLVA benchmark experiments with intervention labels")
    parser.add_argument("--envs", type=str, default=",".join(PAPER_ENVIRONMENTS))
    parser.add_argument("--policies", type=str, default="pg,ppo,dqn,random,heuristic")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23")
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--L", type=int, default=50)
    parser.add_argument("--trace-root", type=Path, default=BENCHMARK_TRACE_DIR)
    parser.add_argument("--summary-root", type=Path, default=BENCHMARK_SUMMARY_DIR)
    parser.add_argument("--model-dir", type=Path, default=BENCHMARK_MODEL_DIR)
    parser.add_argument("--intervention-profile", type=str, default="standard", choices=["mild", "standard", "severe"])
    parser.add_argument("--disable-interventions", action="store_true")
    return parser.parse_args()


def _iter_runs(envs: Iterable[str], policies: Iterable[str], seeds: Iterable[int]):
    for env_name in envs:
        for policy_name in policies:
            for seed in seeds:
                yield env_name, policy_name, seed


def main() -> None:
    args = parse_args()
    envs = _parse_csv_list(args.envs)
    policies = _parse_csv_list(args.policies)
    seeds = _parse_seeds(args.seeds)

    unknown_envs = [name for name in envs if name not in PAPER_ENVIRONMENTS]
    if unknown_envs:
        raise ValueError("Unknown environments: {0}".format(unknown_envs))
    unknown_policies = [name for name in policies if name not in BENCHMARK_POLICIES]
    if unknown_policies:
        raise ValueError("Unknown policies: {0}".format(unknown_policies))

    for env_name, policy_name, seed in _iter_runs(envs=envs, policies=policies, seeds=seeds):
        if not is_env_available(env_name):
            print("[skip] env={0} is unavailable in this environment".format(env_name))
            continue
        env = make_env(
            env_name=env_name,
            seed=seed,
            with_interventions=not args.disable_interventions,
            horizon=args.T,
            intervention_profile=args.intervention_profile,
        )
        policy = _build_policy(policy_name=policy_name, env_name=env_name, seed=seed, model_dir=args.model_dir)

        trace_df = run_and_record(env=env, policy=policy, T=args.T, seed=seed, policy_name=policy_name)
        summary_df = summarize_trace(trace=trace_df, L=args.L)

        trace_out = _trace_out_path(root=args.trace_root, env_name=env_name, policy_name=policy_name, seed=seed)
        summary_out = _summary_out_path(root=args.summary_root, env_name=env_name, policy_name=policy_name, seed=seed)
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(trace_out, index=False)
        summary_df.to_csv(summary_out, index=False)

        print(
            "[env={0} policy={1} seed={2}] trace_rows={3} summary_rows={4}".format(
                env_name,
                policy_name,
                seed,
                len(trace_df),
                len(summary_df),
            )
        )
        print("  trace={0}".format(trace_out))
        print("  summary={0}".format(summary_out))


if __name__ == "__main__":
    main()
