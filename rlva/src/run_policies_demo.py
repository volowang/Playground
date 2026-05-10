from __future__ import annotations

from rlva.src.config import ENVIRONMENTS
from rlva.src.env import is_env_available, make_env
from rlva.src.policies import HeuristicPolicy, RandomPolicy, SoftmaxLinearPolicy


def run_policy(env_name, policy, seed, num_steps=100):
    env = make_env(env_name=env_name, seed=seed)
    state = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(num_steps):
        action_id, _probs = policy.act(state, getattr(env, "queue", []))
        state, reward, _info = env.step(action_id)
        total_reward += reward
    return total_reward / float(num_steps)


def main() -> None:
    for env_name in ENVIRONMENTS:
        if not is_env_available(env_name):
            print("\n[{0}] skipped (optional backend unavailable)".format(env_name))
            continue
        print("\n[{0}]".format(env_name))
        for idx, (name, policy) in enumerate(
            [
                ("random", RandomPolicy(seed=42)),
                ("heuristic_a0", HeuristicPolicy(mode="a0")),
                ("rl_stub", SoftmaxLinearPolicy(state_dim=4, seed=42)),
            ]
        ):
            avg_reward = run_policy(env_name=env_name, policy=policy, seed=42 + idx)
            print("{0:14s} avg_reward={1:.4f}".format(name, avg_reward))


if __name__ == "__main__":
    main()
