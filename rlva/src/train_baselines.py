from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from rlva.src.config import BENCHMARK_MODEL_DIR, LEARNED_POLICIES, PAPER_ENVIRONMENTS, resolve_benchmark_model_path
from rlva.src.env import is_env_available, make_env
from rlva.src.learned_policy import MLPPolicyNet, MLPQNet


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seeds(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def discounted_returns(rewards: Sequence[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train_pg(env_name: str, seed: int, episodes: int, steps_per_episode: int, gamma: float, lr: float, hidden_dim: int) -> nn.Module:
    torch.manual_seed(seed)
    env = make_env(env_name=env_name, seed=seed)
    policy = MLPPolicyNet(state_dim=env.state_dim, num_actions=3, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        for _ in range(steps_per_episode):
            logits = policy(torch.tensor(state, dtype=torch.float32))
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
            log_probs.append(dist.log_prob(torch.tensor(action)))
            state, reward, _info = env.step(action)
            rewards.append(float(reward))
        returns = discounted_returns(rewards, gamma)
        advantages = (returns - returns.mean()) / max(1e-6, float(returns.std(unbiased=False)))
        loss = -torch.sum(torch.stack(log_probs) * advantages)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy


def train_ppo(env_name: str, seed: int, episodes: int, steps_per_episode: int, gamma: float, lr: float, hidden_dim: int, clip_eps: float) -> nn.Module:
    torch.manual_seed(seed)
    env = make_env(env_name=env_name, seed=seed)
    policy = MLPPolicyNet(state_dim=env.state_dim, num_actions=3, hidden_dim=hidden_dim)
    value_net = ValueNet(state_dim=env.state_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=lr)

    for episode in range(episodes):
        state = env.reset(seed=seed + episode)
        states: List[torch.Tensor] = []
        actions: List[int] = []
        old_log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        for _ in range(steps_per_episode):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = policy(state_tensor)
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())
            next_state, reward, _info = env.step(action)
            states.append(state_tensor)
            actions.append(action)
            old_log_probs.append(dist.log_prob(torch.tensor(action)).detach())
            rewards.append(float(reward))
            state = next_state

        returns = discounted_returns(rewards, gamma)
        state_batch = torch.stack(states)
        action_batch = torch.tensor(actions, dtype=torch.int64)
        old_log_prob_batch = torch.stack(old_log_probs)
        value_pred = value_net(state_batch)
        advantages = returns - value_pred.detach()
        advantages = (advantages - advantages.mean()) / max(1e-6, float(advantages.std(unbiased=False)))

        for _ in range(4):
            logits = policy(state_batch)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(action_batch)
            ratio = torch.exp(new_log_probs - old_log_prob_batch)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = torch.mean((value_net(state_batch) - returns) ** 2)
            entropy_bonus = torch.mean(dist.entropy())
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return policy


def train_dqn(
    env_name: str,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    gamma: float,
    lr: float,
    hidden_dim: int,
    batch_size: int = 64,
) -> nn.Module:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = make_env(env_name=env_name, seed=seed)
    q_net = MLPQNet(state_dim=env.state_dim, num_actions=3, hidden_dim=hidden_dim)
    target_net = MLPQNet(state_dim=env.state_dim, num_actions=3, hidden_dim=hidden_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay: Deque[Tuple[List[float], int, float, List[float]]] = deque(maxlen=5000)

    epsilon_start = 0.35
    epsilon_end = 0.05
    total_updates = 0
    for episode in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, (episodes - episode) / float(episodes))
        state = env.reset(seed=seed + episode)
        for _ in range(steps_per_episode):
            if rng.random() < epsilon:
                action = rng.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(state, dtype=torch.float32))
                    action = int(torch.argmax(q_values).item())
            next_state, reward, _info = env.step(action)
            replay.append((list(state), action, float(reward), list(next_state)))
            state = next_state

            if len(replay) < batch_size:
                continue
            batch = rng.sample(list(replay), batch_size)
            state_batch = torch.tensor([item[0] for item in batch], dtype=torch.float32)
            action_batch = torch.tensor([item[1] for item in batch], dtype=torch.int64)
            reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            next_state_batch = torch.tensor([item[3] for item in batch], dtype=torch.float32)

            q_values = q_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q = reward_batch + gamma * torch.max(target_net(next_state_batch), dim=1)[0]
            loss = torch.mean((q_values - target_q) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_updates += 1
            if total_updates % 40 == 0:
                target_net.load_state_dict(q_net.state_dict())
    target_net.load_state_dict(q_net.state_dict())
    return q_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stronger RLVA baselines for benchmark experiments")
    parser.add_argument("--envs", type=str, default=",".join(PAPER_ENVIRONMENTS))
    parser.add_argument("--policies", type=str, default="pg,ppo,dqn")
    parser.add_argument("--seeds", type=str, default="11,13,17,19,23")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--steps-per-episode", type=int, default=180)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--model-dir", type=Path, default=BENCHMARK_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envs = _parse_csv_list(args.envs)
    policies = _parse_csv_list(args.policies)
    seeds = _parse_seeds(args.seeds)

    for env_name in envs:
        if env_name not in PAPER_ENVIRONMENTS:
            raise ValueError("Unknown environment: {0}".format(env_name))
    for policy_name in policies:
        if policy_name not in LEARNED_POLICIES:
            raise ValueError("Unsupported learned policy: {0}".format(policy_name))

    for env_name in envs:
        if not is_env_available(env_name):
            print("[skip] env={0} is unavailable in this environment".format(env_name))
            continue
        for policy_name in policies:
            for seed in seeds:
                if policy_name == "pg":
                    model = train_pg(
                        env_name=env_name,
                        seed=seed,
                        episodes=args.episodes,
                        steps_per_episode=args.steps_per_episode,
                        gamma=args.gamma,
                        lr=args.lr,
                        hidden_dim=args.hidden_dim,
                    )
                elif policy_name == "ppo":
                    model = train_ppo(
                        env_name=env_name,
                        seed=seed,
                        episodes=args.episodes,
                        steps_per_episode=args.steps_per_episode,
                        gamma=args.gamma,
                        lr=args.lr,
                        hidden_dim=args.hidden_dim,
                        clip_eps=args.clip_eps,
                    )
                else:
                    model = train_dqn(
                        env_name=env_name,
                        seed=seed,
                        episodes=args.episodes,
                        steps_per_episode=args.steps_per_episode,
                        gamma=args.gamma,
                        lr=args.lr,
                        hidden_dim=args.hidden_dim,
                    )

                out_path = resolve_benchmark_model_path(env_name=env_name, policy=policy_name, seed=seed, model_dir=args.model_dir)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "algorithm": policy_name,
                        "env_name": env_name,
                        "state_dim": 4,
                        "num_actions": 3,
                        "seed": seed,
                        "hidden_dim": args.hidden_dim,
                        "model_state_dict": model.state_dict(),
                    },
                    out_path,
                )
                print("Saved {0} model: {1}".format(policy_name, out_path))


if __name__ == "__main__":
    main()
