from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.distributions import Categorical

from rlva.src.config import ENVIRONMENTS, MODEL_DIR, resolve_model_path
from rlva.src.env import make_env
from rlva.src.policies import SoftmaxLinearPolicy


class TorchSoftmaxLinearPolicy(nn.Module):
    def __init__(self, base_policy: SoftmaxLinearPolicy) -> None:
        super(TorchSoftmaxLinearPolicy, self).__init__()
        self.linear = nn.Linear(base_policy.state_dim, base_policy.num_actions)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(base_policy.W, dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor(base_policy.b, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state)


def discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train(env_name: str, episodes: int, steps_per_episode: int, gamma: float, lr: float, log_every: int, seed: int, out_path: Path) -> None:
    torch.manual_seed(seed)
    env = make_env(env_name=env_name, seed=seed)
    base_policy = SoftmaxLinearPolicy(state_dim=env.state_dim, num_actions=3, seed=seed)
    policy = TorchSoftmaxLinearPolicy(base_policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    running_baseline = 0.0
    for episode in range(1, episodes + 1):
        state = env.reset(seed=seed + episode)
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []

        for _ in range(steps_per_episode):
            logits = policy(torch.tensor(state, dtype=torch.float32))
            dist = Categorical(logits=logits)
            action_id = int(dist.sample().item())
            log_probs.append(dist.log_prob(torch.tensor(action_id)))
            state, reward, _info = env.step(action_id)
            rewards.append(float(reward))

        returns = discounted_returns(rewards, gamma)
        episode_return = float(sum(rewards))
        running_baseline = episode_return if episode == 1 else running_baseline + (episode_return - running_baseline) / float(episode)
        advantages = returns - torch.full_like(returns, running_baseline)
        loss = -torch.sum(advantages.detach() * torch.stack(log_probs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % log_every == 0 or episode == 1 or episode == episodes:
            print(
                "[{0}] episode={1:4d}/{2} avg_reward={3:8.3f} baseline={4:9.3f}".format(
                    env_name,
                    episode,
                    episodes,
                    episode_return / float(steps_per_episode),
                    running_baseline,
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "env_name": env_name,
            "state_dim": env.state_dim,
            "num_actions": 3,
            "gamma": gamma,
            "seed": seed,
            "model_state_dict": policy.state_dict(),
        },
        out_path,
    )
    print("Saved model to: {0}".format(out_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE policy on one of the demo environments")
    parser.add_argument("--env", type=str, required=True, choices=ENVIRONMENTS)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--steps-per-episode", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out if args.out is not None else resolve_model_path(env_name=args.env, model_dir=MODEL_DIR)
    train(
        env_name=args.env,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        gamma=args.gamma,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
