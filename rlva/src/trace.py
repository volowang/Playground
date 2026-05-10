from __future__ import annotations

import random
from typing import Any, Dict, List

import pandas as pd

from rlva.src.env import BaseDecisionEnv


TRACE_COLUMNS = [
    "env",
    "policy",
    "seed",
    "t",
    "q",
    "mu",
    "sigma",
    "rho",
    "a",
    "action_label",
    "p0",
    "p1",
    "p2",
    "r",
    "state_delta",
    "intervention_active",
    "intervention_start",
    "intervention_name",
]


def run_and_record(env: BaseDecisionEnv, policy: Any, T: int, seed: int, policy_name: str = "unknown") -> pd.DataFrame:
    if T <= 0:
        raise ValueError("T must be positive")

    random.seed(seed)
    state = env.reset(seed=seed)
    rows: List[Dict[str, float]] = []

    for t in range(T):
        action_id, probs = policy.act(state, getattr(env, "queue", []))
        if len(probs) != 3:
            raise ValueError("Policy must output 3 action probs, got {0}".format(len(probs)))

        next_state, reward, info = env.step(int(action_id))
        state_delta = sum((float(next_state[i]) - float(state[i])) ** 2 for i in range(len(state))) ** 0.5

        rows.append(
            {
                "env": env.ENV_NAME,
                "policy": str(policy_name),
                "seed": int(seed),
                "t": int(t),
                "q": float(state[0]),
                "mu": float(state[1]),
                "sigma": float(state[2]),
                "rho": float(state[3]),
                "a": int(action_id),
                "action_label": str(info.get("action_label", env.action_label(int(action_id)))),
                "p0": float(probs[0]),
                "p1": float(probs[1]),
                "p2": float(probs[2]),
                "r": float(reward),
                "state_delta": float(state_delta),
                "intervention_active": float(info.get("intervention_active", 0.0)),
                "intervention_start": float(info.get("intervention_start", 0.0)),
                "intervention_name": str(info.get("intervention_name", "none")),
            }
        )
        state = next_state

    return pd.DataFrame(rows, columns=TRACE_COLUMNS)
