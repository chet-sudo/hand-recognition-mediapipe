"""Base agent interface shared by all algorithms in the RL lab.

The goal of this interface is to keep the training loop agnostic to the
specific algorithm. Every agent receives environment observations and
returns an action to take. During training, the agent also receives the
reward and termination flag so it can update its internal parameters.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional


class Agent(abc.ABC):
    """Abstract agent used by the training loop.

    The interface is intentionally lightweight so that tabular and neural
    network-based agents can both implement it without heavy boilerplate.
    """

    @abc.abstractmethod
    def begin_episode(self, observation: Any) -> Any:
        """Called at the start of each episode.

        Args:
            observation: Initial observation returned by ``env.reset``.

        Returns:
            The first action to take in the environment.
        """

    @abc.abstractmethod
    def step(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Any:
        """Update the agent after a transition and return the next action.

        Args:
            observation: Observation after taking the previous action.
            reward: Reward from the previous action.
            terminated: Whether the episode ended because a terminal state
                was reached.
            truncated: Whether the episode ended because of a time limit.

        Returns:
            The next action to take.
        """

    @abc.abstractmethod
    def act(self, observation: Any) -> Any:
        """Choose an action for evaluation (no exploration side effects)."""

    @abc.abstractmethod
    def end_episode(self) -> None:
        """Hook called when an episode finishes."""

    def state_dict(self) -> Dict[str, Any]:
        """Optional serialization hook used by some agents."""
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Optional deserialization hook used by some agents."""
        return

    def on_train_start(self, seed: Optional[int] = None) -> None:
        """Hook executed once before training begins."""
        return
