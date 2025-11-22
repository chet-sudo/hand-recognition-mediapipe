"""PPO experiment for the MuJoCo Pusher-v4 environment."""
from __future__ import annotations

import gymnasium as gym

from agents.actor_critic.ppo import PPOAgent
from utils.training import record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run(episodes: int = 300, render_every: int | None = 75):
    env_id = "Pusher-v4"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]

    ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=2048)

    def env_fn(render_mode=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        return gym.make(env_id, **kwargs)

    ppo_returns, _ = train_agent(
        env_fn, ppo_agent, episodes=episodes, render_every=render_every, log_every=20
    )
    plot_learning_curves({"PPO": ppo_returns}, window=5, title=env_id)

    video_path = record_agent_video(env_fn, ppo_agent, video_dir="videos/pusher", prefix="ppo")
    print("Saved Pusher replay:")
    print(f"- PPO: {video_path}")


if __name__ == "__main__":
    run()
