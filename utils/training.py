"""Generic training and evaluation loops shared by all algorithms."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np
import gymnasium as gym

from agents.base import Agent


EpisodeStats = Dict[str, float]


def train_agent(
    env_fn: Callable[..., gym.Env],
    agent: Agent,
    episodes: int = 200,
    eval_every: int = 0,
    seed: int | None = None,
    render_every: int | None = None,
    render_mode: str = "human",
    log_every: int | None = 20,
) -> Tuple[List[float], List[EpisodeStats]]:
    """Train an agent for a fixed number of episodes.

    The loop is intentionally straightforward and readable. Each step asks the
    agent for an action, applies it to the environment, and then lets the
    agent update itself with the resulting transition.
    """
    returns: List[float] = []
    stats: List[EpisodeStats] = []

    env = _build_env(env_fn, render_mode if render_every else None)
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
        should_render = render_every is not None and (episode % render_every == 0)
        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
            # Let the agent update its parameters and request the next action
            action = agent.step(next_obs, reward, terminated, truncated)
            if should_render:
                env.render()
        agent.end_episode()
        returns.append(total_reward)
        stats.append({"return": total_reward, "length": length})

        if log_every and (episode + 1) % log_every == 0:
            window = returns[-log_every:]
            mean_return = float(np.mean(window))
            print(
                f"[Episode {episode + 1}/{episodes}] "
                f"last return={total_reward:.2f}, length={length}, "
                f"{log_every}-episode avg return={mean_return:.2f}"
            )

    env.close()
    return returns, stats


def evaluate_agent(env_fn: Callable[..., gym.Env], agent: Agent, episodes: int = 5) -> float:
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


def _build_env(env_fn: Callable[..., gym.Env], render_mode: str | None) -> gym.Env:
    """Create an environment, passing through render mode when supported."""

    try:
        if render_mode is not None:
            return env_fn(render_mode=render_mode)
    except TypeError:
        # ``render_mode`` is optional; fall back if the factory does not accept it.
        pass
    return env_fn()


def record_agent_video(
    env_fn: Callable[..., gym.Env],
    agent: Agent,
    video_dir: str = "videos",
    episodes: int = 1,
    prefix: str | None = None,
) -> Path:
    """Roll out an agent and save a replay video.

    Environments are constructed with ``render_mode="rgb_array"`` when supported so
    the resulting frames can be written by :class:`gymnasium.wrappers.RecordVideo`.
    """

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = prefix or "agent"
    output_dir = Path(video_dir) / f"{folder_name}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(env_fn, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(output_dir), episode_trigger=lambda i: True)

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    return output_dir
