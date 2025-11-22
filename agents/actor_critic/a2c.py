"""Synchronous Advantage Actor-Critic (A2C) implementation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from utils.networks import MLP, GaussianPolicy


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


class A2CAgent(Agent):
    """Simple actor-critic for both discrete and continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_space,
        hidden_sizes=(128, 128),
        gamma: float = 0.99,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.is_discrete = hasattr(action_space, "n")
        self.action_space = action_space

        if self.is_discrete:
            self.policy = MLP(obs_dim, action_space.n, hidden_sizes)
        else:
            self.policy = GaussianPolicy(obs_dim, action_space.shape[0], hidden_sizes)
        self.value_fn = MLP(obs_dim, 1, hidden_sizes)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_fn.parameters()), lr=lr)

        self.last_obs = None
        self.last_action = None

    def _sample_action(self, obs_tensor: torch.Tensor):
        if self.is_discrete:
            logits = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean, std = self.policy.forward(obs_tensor)
            dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if not self.is_discrete:
            log_prob = log_prob.sum(-1)
        return action, log_prob, dist.entropy()

    def begin_episode(self, observation):
        obs_tensor = to_tensor(observation)
        action, log_prob, entropy = self._sample_action(obs_tensor)
        self.last_obs = obs_tensor
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_entropy = entropy
        return self._to_env_action(action)

    def _to_env_action(self, action_tensor: torch.Tensor):
        action = action_tensor.detach().cpu().numpy()
        if self.is_discrete:
            return int(action)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def step(self, observation, reward: float, terminated: bool, truncated: bool):
        done = terminated or truncated
        obs_tensor = to_tensor(observation)
        value = self.value_fn(self.last_obs)
        next_value = self.value_fn(obs_tensor).detach()
        # Advantage = (r + gamma * V(s') - V(s)). Using TD(0) target.
        advantage = reward + self.gamma * (1 - float(done)) * next_value - value

        if self.is_discrete:
            logits = self.policy(self.last_obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(self.last_action)
            entropy = dist.entropy()
        else:
            mean, std = self.policy(self.last_obs)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(self.last_action).sum(-1)
            entropy = dist.entropy().sum(-1)

        actor_loss = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy_bonus = -self.entropy_coef * entropy
        loss = actor_loss + critic_loss + entropy_bonus

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.last_obs = obs_tensor
        action, log_prob, entropy = self._sample_action(obs_tensor)
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_entropy = entropy
        return self._to_env_action(action)

    def act(self, observation):
        obs_tensor = to_tensor(observation)
        if self.is_discrete:
            logits = self.policy(obs_tensor)
            return int(torch.argmax(logits).item())
        else:
            mean, _ = self.policy(obs_tensor)
            action = mean.detach().cpu().numpy()
            return np.clip(action, self.action_space.low, self.action_space.high)

    def end_episode(self) -> None:
        return
