from __future__ import annotations

from rlva.src.config import ENVIRONMENTS
from rlva.src.env import is_env_available, make_env


def main() -> None:
    for env_name in ENVIRONMENTS:
        if not is_env_available(env_name):
            print("{0:10s} skipped (optional backend unavailable)".format(env_name))
            continue
        env = make_env(env_name=env_name, seed=42)
        state = env.reset()
        for _ in range(20):
            state, reward, info = env.step(env.sample_action())
        print("{0:10s} final_state={1} reward={2:.4f} info={3}".format(env_name, [round(v, 3) for v in state], float(reward), info))


if __name__ == "__main__":
    main()
