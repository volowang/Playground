from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch import nn


class MLPPolicyNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super(MLPPolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MLPQNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super(MLPQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LearnedPolicy(object):
    def __init__(self, model_path: Path, seed: int) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        self.algorithm = str(checkpoint["algorithm"])
        self.state_dim = int(checkpoint["state_dim"])
        self.num_actions = int(checkpoint["num_actions"])
        self.hidden_dim = int(checkpoint.get("hidden_dim", 64))
        self.rng = random.Random(seed)

        if self.algorithm == "dqn":
            self.model = MLPQNet(state_dim=self.state_dim, num_actions=self.num_actions, hidden_dim=self.hidden_dim)
        else:
            self.model = MLPPolicyNet(state_dim=self.state_dim, num_actions=self.num_actions, hidden_dim=self.hidden_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _softmax(self, logits: Sequence[float]) -> List[float]:
        m = max(logits)
        exps = [math.exp(value - m) for value in logits]
        denom = sum(exps)
        return [value / denom for value in exps]

    def act(self, state: Sequence[float], queue: Sequence[object]) -> Tuple[int, List[float]]:
        del queue
        with torch.no_grad():
            values = self.model(torch.tensor(state, dtype=torch.float32)).tolist()
        if self.algorithm == "dqn":
            best = int(max(range(len(values)), key=lambda idx: values[idx]))
            probs = self._softmax(values)
            return best, probs
        probs = self._softmax(values)
        r = self.rng.random()
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if r <= acc:
                return idx, probs
        return len(probs) - 1, probs
