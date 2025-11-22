"""On-policy SARSA implementation for discrete environments."""
from __future__ import annotations

import numpy as np
from agents.base import Agent


class SarsaAgent(Agent):
    """SARSA uses the action actually taken in the next state for its target.

    This makes it on-policy: the behavior policy (exploratory epsilon-greedy)
    is also the policy being evaluated and improved.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None
        self.n_actions = n_actions

    def _choose_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def begin_episode(self, observation: int) -> int:
        self.last_state = observation
        action = self._choose_action(observation)
        self.last_action = action
        return action

    def step(self, observation: int, reward: float, terminated: bool, truncated: bool) -> int:
        # On-policy TD target uses the value of the *next exploratory action*.
        next_action = self._choose_action(observation)
        td_target = reward + self.gamma * self.q_table[observation, next_action] * (1 - float(terminated))
        td_error = td_target - self.q_table[self.last_state, self.last_action]
        self.q_table[self.last_state, self.last_action] += self.alpha * td_error

        self.last_state = observation
        self.last_action = next_action
        return next_action

    def act(self, observation: int) -> int:
        return int(np.argmax(self.q_table[observation]))

    def end_episode(self) -> None:
        return
