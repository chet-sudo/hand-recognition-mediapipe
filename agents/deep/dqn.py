"""Minimal DQN implementation with replay buffer and target network."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from utils.replay_buffer import ReplayBuffer
from utils.networks import MLP


def to_tensor(array) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


class DQNAgent(Agent):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes=(128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 50000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        target_update: int = 200,
    ):
        self.q_net = MLP(obs_dim, n_actions, hidden_sizes)
        self.target_net = MLP(obs_dim, n_actions, hidden_sizes)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps = 0

        self.last_obs = None
        self.last_action = None

    def _epsilon(self) -> float:
        # Linearly decay epsilon until it reaches epsilon_end
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.steps / self.epsilon_decay)

    def _choose_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self._epsilon():
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.q_net(to_tensor(obs)).cpu().numpy()
        return int(np.argmax(q_values))

    def begin_episode(self, observation):
        action = self._choose_action(observation)
        self.last_obs = observation
        self.last_action = action
        return action

    def step(self, observation, reward: float, terminated: bool, truncated: bool):
        done = float(terminated)
        # Store transition for replay. Using ``done`` as a float lets us mask
        # the bootstrap term later.
        self.buffer.push(self.last_obs, self.last_action, reward, observation, done)
        self.steps += 1

        self._maybe_update_networks()

        self.last_obs = observation
        self.last_action = self._choose_action(observation)
        return self.last_action

    def _maybe_update_networks(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states = to_tensor(batch.state)
        actions = torch.as_tensor(batch.action, dtype=torch.long)
        rewards = to_tensor(batch.reward).unsqueeze(1)
        next_states = to_tensor(batch.next_state)
        dones = to_tensor(batch.done).unsqueeze(1)

        # Q-learning target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * max_next_q

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update the target network to provide a stable bootstrap
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, observation):
        with torch.no_grad():
            q_values = self.q_net(to_tensor(observation)).cpu().numpy()
        return int(np.argmax(q_values))

    def end_episode(self) -> None:
        return
