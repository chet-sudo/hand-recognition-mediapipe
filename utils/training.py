"""Generic training and evaluation loops shared by all algorithms."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import numpy as np
import gymnasium as gym

from agents.base import Agent


EpisodeStats = Dict[str, float]


def train_agent(
    env_fn: Callable[[], gym.Env],
    agent: Agent,
    episodes: int = 200,
    eval_every: int = 0,
    seed: int | None = None,
) -> Tuple[List[float], List[EpisodeStats]]:
    """Train an agent for a fixed number of episodes.

    The loop is intentionally straightforward and readable. Each step asks the
    agent for an action, applies it to the environment, and then lets the
    agent update itself with the resulting transition.
    """
    returns: List[float] = []
    stats: List[EpisodeStats] = []

    env = env_fn()
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    agent.on_train_start(seed=seed)

    for episode in range(episodes):
        obs, _ = env.reset()
        action = agent.begin_episode(obs)
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
            # Let the agent update its parameters and request the next action
            action = agent.step(next_obs, reward, terminated, truncated)
        agent.end_episode()
        returns.append(total_reward)
        stats.append({"return": total_reward, "length": length})

    env.close()
    return returns, stats


def evaluate_agent(env_fn: Callable[[], gym.Env], agent: Agent, episodes: int = 5) -> float:
    """Evaluate an agent without exploration noise."""
    env = env_fn()
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        scores.append(total)
    env.close()
    return float(np.mean(scores))
