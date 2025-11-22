"""PPO experiment for the MuJoCo Pusher-v4 environment."""
from __future__ import annotations

import gymnasium as gym

from agents.actor_critic.ppo import PPOAgent
from utils.training import train_agent
from utils.plotting import plot_learning_curves


def run(episodes: int = 300):
    env_id = "Pusher-v4"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]

    ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=2048)

    def env_fn():
        return gym.make(env_id)

    ppo_returns, _ = train_agent(env_fn, ppo_agent, episodes=episodes)
    plot_learning_curves({"PPO": ppo_returns}, window=5, title=env_id)


if __name__ == "__main__":
    run()
