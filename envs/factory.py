"""Environment creation utilities using Gymnasium.

This module keeps environment setup in one place so experiments can
request environments by ID with consistent options such as seeding or
wrapping.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from typing import Optional, Tuple


def make_env(env_id: str, seed: Optional[int] = None, flatten: bool = False) -> gym.Env:
    """Create a Gymnasium environment with minimal, educational defaults.

    Args:
        env_id: Official Gymnasium environment ID.
        seed: Optional random seed for deterministic resets.
        flatten: If True, wrap the environment so complex observations (e.g.,
            Dict or Tuple) are flattened into a single vector.

    Returns:
        A configured Gymnasium environment instance.
    """
    env = gym.make(env_id)
    if seed is not None:
        # Seed both environment dynamics and action space sampling
        env.reset(seed=seed)
        env.action_space.seed(seed)
    if flatten:
        env = FlattenObservation(env)
    return env


def eval_episode(env: gym.Env, agent, render: bool = False) -> Tuple[float, int]:
    """Run a single evaluation episode using ``agent.act``.

    Args:
        env: Environment created via ``make_env``.
        agent: Agent implementing an ``act`` method.
        render: Whether to render frames during evaluation.

    Returns:
        Tuple of (episode_return, length).
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        if render:
            env.render()
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    return total_reward, steps
