from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List, Sequence

import torch
from torch import nn

from rlva.src.config import ENVIRONMENTS, MODEL_DIR, TRACE_DIR, resolve_model_path, trace_filename
from rlva.src.env import make_env
from rlva.src.policies import HeuristicPolicy, RandomPolicy
from rlva.src.trace import run_and_record


class TorchSoftmaxLinearPolicy(nn.Module):
    def __init__(self, state_dim: int, num_actions: int) -> None:
        super(TorchSoftmaxLinearPolicy, self).__init__()
        self.linear = nn.Linear(state_dim, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


class LearnedPGPolicy(object):
    def __init__(self, model_path: Path, seed: int) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = TorchSoftmaxLinearPolicy(state_dim=int(checkpoint["state_dim"]), num_actions=int(checkpoint["num_actions"]))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.rng = random.Random(seed)

    def _softmax(self, logits: Sequence[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(value - m) for value in logits]
        denom = sum(exps)
        return [value / denom for value in exps]

    def act(self, state: Sequence[float], queue: Sequence[object]) -> tuple:
        del queue
        with torch.no_grad():
            logits = self.model(torch.tensor(state, dtype=torch.float32)).tolist()
        probs = self._softmax(logits)
        r = self.rng.random()
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if r <= acc:
                return idx, probs
        return len(probs) - 1, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect comparable decision traces for one environment")
    parser.add_argument("--env", type=str, required=True, choices=ENVIRONMENTS)
    parser.add_argument("--T", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--heuristic-mode", type=str, default="a0", choices=["a0", "a1", "a2"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model if args.model is not None else resolve_model_path(env_name=args.env, model_dir=MODEL_DIR)

    pg_policy = LearnedPGPolicy(model_path=model_path, seed=args.seed + 101)
    random_policy = RandomPolicy(seed=args.seed + 202)
    heuristic_policy = HeuristicPolicy(mode=args.heuristic_mode)

    for policy_name, policy in [("pg", pg_policy), ("random", random_policy), ("heuristic", heuristic_policy)]:
        env = make_env(env_name=args.env, seed=args.seed)
        df = run_and_record(env=env, policy=policy, T=args.T, seed=args.seed, policy_name=policy_name)
        out_path = args.out_dir / trace_filename(env_name=args.env, policy=policy_name)
        df.to_csv(out_path, index=False)
        print("Saved: {0} ({1} rows)".format(out_path, len(df)))


if __name__ == "__main__":
    main()
