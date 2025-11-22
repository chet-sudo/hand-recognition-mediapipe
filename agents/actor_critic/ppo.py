"""Proximal Policy Optimization (PPO) with clipped surrogate loss.

The clipping term prevents the new policy from moving too far away from
its behavior policy in a single update, which stabilizes training.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from utils.networks import MLP, GaussianPolicy


def to_tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def compute_gae(rewards, values, masks, gamma: float, lam: float):
    """Generalized Advantage Estimation helper."""
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return torch.stack(advantages), torch.stack(returns)


class PPOAgent(Agent):
    def __init__(
        self,
        obs_dim: int,
        action_space,
        hidden_sizes=(64, 64),
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        train_iters: int = 5,
        batch_size: int = 2048,
        entropy_coef: float = 0.0,
    ):
        self.is_discrete = hasattr(action_space, "n")
        self.action_space = action_space
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        if self.is_discrete:
            self.policy = MLP(obs_dim, action_space.n, hidden_sizes)
        else:
            self.policy = GaussianPolicy(obs_dim, action_space.shape[0], hidden_sizes)
        self.value_fn = MLP(obs_dim, 1, hidden_sizes)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_fn.parameters()), lr=lr)

        # Storage for trajectory data collected within an episode
        self.reset_storage()

    def reset_storage(self):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.mask_buf = []
        self.val_buf = []

    def _distribution(self, obs_tensor: torch.Tensor):
        if self.is_discrete:
            logits = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mean, std = self.policy(obs_tensor)
            dist = torch.distributions.Normal(mean, std)
        return dist

    def begin_episode(self, observation):
        obs_tensor = to_tensor(observation)
        action, log_prob = self._sample_action(obs_tensor)
        value = self.value_fn(obs_tensor)
        self._store(obs_tensor, action, log_prob, 0.0, 1.0, value)
        return self._to_env_action(action)

    def _sample_action(self, obs_tensor):
        dist = self._distribution(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if not self.is_discrete:
            log_prob = log_prob.sum(-1)
        return action, log_prob

    def _to_env_action(self, action_tensor: torch.Tensor):
        action = action_tensor.detach().cpu().numpy()
        if self.is_discrete:
            return int(action)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def step(self, observation, reward: float, terminated: bool, truncated: bool):
        done = terminated or truncated
        obs_tensor = to_tensor(observation)
        action, log_prob = self._sample_action(obs_tensor)
        value = self.value_fn(obs_tensor)
        mask = 0.0 if done else 1.0
        self._store(obs_tensor, action, log_prob, reward, mask, value)

        if done or len(self.rew_buf) >= self.batch_size:
            # When an episode ends or the batch is full, update networks.
            self._update(done_value=0.0 if done else value.detach())
            self.reset_storage()
        return self._to_env_action(action)

    def _store(self, obs, action, log_prob, reward, mask, value):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.logp_buf.append(log_prob)
        self.rew_buf.append(torch.tensor(reward, dtype=torch.float32))
        self.mask_buf.append(torch.tensor(mask, dtype=torch.float32))
        self.val_buf.append(value.detach())

    def _update(self, done_value: float):
        # Append value for final state to compute bootstrap targets
        values = torch.stack(self.val_buf + [torch.as_tensor(done_value)])
        rewards = torch.stack(self.rew_buf)
        masks = torch.stack(self.mask_buf)

        advantages, returns = compute_gae(rewards, values, masks, self.gamma, self.lam)
        obs_tensor = torch.stack(self.obs_buf)
        act_tensor = torch.stack(self.act_buf)
        old_logp = torch.stack(self.logp_buf)

        for _ in range(self.train_iters):
            dist = self._distribution(obs_tensor)
            new_logp = dist.log_prob(act_tensor)
            if not self.is_discrete:
                new_logp = new_logp.sum(-1)
            ratio = torch.exp(new_logp - old_logp)
            # PPO objective clips the policy ratio to prevent large updates
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            entropy = dist.entropy()
            if not self.is_discrete:
                entropy = entropy.sum(-1)

            value_pred = self.value_fn(obs_tensor).squeeze(-1)
            value_loss = (returns - value_pred).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def act(self, observation):
        obs_tensor = to_tensor(observation)
        if self.is_discrete:
            logits = self.policy(obs_tensor)
            return int(torch.argmax(logits).item())
        mean, _ = self.policy(obs_tensor)
        action = mean.detach().cpu().numpy()
        return np.clip(action, self.action_space.low, self.action_space.high)

    def end_episode(self) -> None:
        return
