"""Replay buffer used by off-policy algorithms such as DQN.

Storing transitions and sampling them uniformly breaks the temporal
correlations present in online experience streams. That helps stabilize
neural network training and lets us reuse past experience more than once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random
import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    """Simple cyclic replay buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.position = 0

    def push(self, *transition: Tuple[np.ndarray, int, float, np.ndarray, float]) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # type: ignore
        self.buffer[self.position] = Transition(*transition)  # type: ignore
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return Transition(states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        return len(self.buffer)
