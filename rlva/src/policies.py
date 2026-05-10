from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple


ActionProbs = List[float]


class BasePolicy:
    """Unified policy interface."""

    def act(self, state: Sequence[float], queue: Sequence[object]) -> Tuple[int, ActionProbs]:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Uniformly sample action from {0, 1, 2}."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def act(self, state: Sequence[float], queue: Sequence[object]) -> Tuple[int, ActionProbs]:
        probs = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        action_id = self._sample_from_probs(probs)
        return action_id, probs

    def _sample_from_probs(self, probs: Sequence[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1


class HeuristicPolicy(BasePolicy):
    """Fixed-rule policy: always pick one action id."""

    MODES = ("a0", "a1", "a2", "shortest", "longest", "fcfs")

    def __init__(self, mode: str = "a0") -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}. Expected one of {self.MODES}.")
        self.mode = mode

    def act(self, state: Sequence[float], queue: Sequence[object]) -> Tuple[int, ActionProbs]:
        del state
        del queue
        if self.mode in ("a0", "shortest"):
            return 0, [1.0, 0.0, 0.0]
        if self.mode in ("a1", "longest"):
            return 1, [0.0, 1.0, 0.0]
        return 2, [0.0, 0.0, 1.0]


class SoftmaxLinearPolicy(BasePolicy):
    """RL policy stub: probs = softmax(W*s + b), then sample action."""

    def __init__(
        self,
        state_dim: int = 4,
        num_actions: int = 3,
        seed: Optional[int] = None,
        init_scale: float = 0.1,
    ) -> None:
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_actions != 3:
            raise ValueError("num_actions must be 3 for this project")

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.rng = random.Random(seed)

        self.W: List[List[float]] = [
            [self.rng.uniform(-init_scale, init_scale) for _ in range(state_dim)]
            for _ in range(num_actions)
        ]
        self.b: List[float] = [0.0 for _ in range(num_actions)]

    def act(self, state: Sequence[float], queue: Sequence[object]) -> Tuple[int, ActionProbs]:
        if len(state) != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {len(state)}.")

        logits = []
        for a in range(self.num_actions):
            v = self.b[a]
            for i in range(self.state_dim):
                v += self.W[a][i] * float(state[i])
            logits.append(v)

        probs = self._softmax(logits)
        action_id = self._sample_from_probs(probs)
        return action_id, probs

    def _softmax(self, logits: Sequence[float]) -> ActionProbs:
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        denom = sum(exps)
        return [x / denom for x in exps]

    def _sample_from_probs(self, probs: Sequence[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1
