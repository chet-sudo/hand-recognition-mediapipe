"""Run tabular methods on FrozenLake-v1 for side-by-side comparison."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from agents.tabular.q_learning import QLearningAgent
from agents.tabular.sarsa import SarsaAgent
from utils.training import train_agent
from utils.plotting import plot_learning_curves


def run(episodes: int = 500):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_agent = QLearningAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)
    sarsa_agent = SarsaAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)

    def env_fn():
        return gym.make("FrozenLake-v1", is_slippery=False)

    q_returns, _ = train_agent(env_fn, q_agent, episodes=episodes)
    sarsa_returns, _ = train_agent(env_fn, sarsa_agent, episodes=episodes)

    plot_learning_curves({"Q-learning": q_returns, "SARSA": sarsa_returns}, window=10, title="FrozenLake")


if __name__ == "__main__":
    run()
