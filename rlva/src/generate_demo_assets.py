from __future__ import annotations

from rlva.src.compare import compare_summary_clusters, load_summary
from rlva.src.config import ENVIRONMENTS, FIG_DIR, TABLE_DIR, comparison_stem, ensure_outputs, resolve_model_path, resolve_summary_path, resolve_trace_path
from rlva.src.env import is_env_available, make_env
from rlva.src.export_findings import main as export_findings_main
from rlva.src.train_pg import train
from rlva.src.policies import HeuristicPolicy, RandomPolicy
from rlva.src.trace import run_and_record
from rlva.src.viz import build_behavior_space_figure, build_comparison_figure
from rlva.src.collect_traces import LearnedPGPolicy
from rlva.src.behavior import summarize_trace


def main() -> None:
    ensure_outputs()
    seed = 42
    for env_name in ENVIRONMENTS:
        if not is_env_available(env_name):
            print("Skipping {0}: optional backend unavailable".format(env_name))
            continue
        model_path = resolve_model_path(env_name=env_name)
        train(env_name=env_name, episodes=40, steps_per_episode=240, gamma=0.99, lr=5e-3, log_every=10, seed=seed, out_path=model_path)

        pg_policy = LearnedPGPolicy(model_path=model_path, seed=seed + 101)
        random_policy = RandomPolicy(seed=seed + 202)
        heuristic_policy = HeuristicPolicy(mode="a0")
        for policy_name, policy in [("pg", pg_policy), ("random", random_policy), ("heuristic", heuristic_policy)]:
            trace_df = run_and_record(env=make_env(env_name=env_name, seed=seed), policy=policy, T=600, seed=seed)
            trace_path = resolve_trace_path(env_name=env_name, policy=policy_name)
            trace_df.to_csv(trace_path, index=False)
            summary_df = summarize_trace(trace_df, L=50)
            summary_path = resolve_summary_path(env_name=env_name, policy=policy_name)
            summary_df.to_csv(summary_path, index=False)
            build_behavior_space_figure(summary_df, env_name=env_name, color_by="anomaly_score").write_html(
                str(FIG_DIR / "{0}__{1}_behavior_space.html".format(env_name, policy_name)),
                include_plotlyjs="cdn",
                full_html=True,
            )

        for p1, p2 in [("pg", "random"), ("pg", "heuristic"), ("random", "heuristic")]:
            table = compare_summary_clusters(load_summary(env_name, p1), load_summary(env_name, p2), n_clusters=8)
            stem = comparison_stem(env_name=env_name, p1=p1, p2=p2)
            table.to_csv(TABLE_DIR / "{0}.csv".format(stem), index=False)
            build_comparison_figure(table, env_name=env_name, p1=p1, p2=p2).write_html(
                str(FIG_DIR / "{0}.html".format(stem)),
                include_plotlyjs="cdn",
                full_html=True,
            )
    export_findings_main()
    print("Demo assets regenerated for environments: {0}".format(", ".join(ENVIRONMENTS)))


if __name__ == "__main__":
    main()
