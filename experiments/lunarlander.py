"""Experiments for LunarLander variants using DQN/A2C/PPO."""
from __future__ import annotations

import gymnasium as gym

from agents.deep.dqn import DQNAgent
from agents.actor_critic.a2c import A2CAgent
from agents.actor_critic.ppo import PPOAgent
from utils.training import record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run_discrete(episodes: int = 500, render_every: int | None = 100):
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    dqn_agent = DQNAgent(obs_dim, n_actions)
    a2c_agent = A2CAgent(obs_dim, env.action_space)
    ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=1024)

    def env_fn(render_mode=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        return gym.make("LunarLander-v2", **kwargs)

    dqn_returns, _ = train_agent(
        env_fn, dqn_agent, episodes=episodes, render_every=render_every, log_every=25
    )
    a2c_returns, _ = train_agent(
        env_fn, a2c_agent, episodes=episodes, render_every=render_every, log_every=25
    )
    ppo_returns, _ = train_agent(
        env_fn, ppo_agent, episodes=episodes, render_every=render_every, log_every=25
    )

    plot_learning_curves(
        {"DQN": dqn_returns, "A2C": a2c_returns, "PPO": ppo_returns},
        window=10,
        title="LunarLander-v2",
    )

    base_dir = "videos/lunarlander-discrete"
    dqn_video = record_agent_video(env_fn, dqn_agent, video_dir=base_dir, prefix="dqn")
    a2c_video = record_agent_video(env_fn, a2c_agent, video_dir=base_dir, prefix="a2c")
    ppo_video = record_agent_video(env_fn, ppo_agent, video_dir=base_dir, prefix="ppo")

    print("Saved LunarLander-v2 replays:")
    print(f"- DQN: {dqn_video}")
    print(f"- A2C: {a2c_video}")
    print(f"- PPO: {ppo_video}")


def run_continuous(episodes: int = 500, render_every: int | None = 100):
    env_id = "LunarLanderContinuous-v2"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]

    ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=1024)
    a2c_agent = A2CAgent(obs_dim, env.action_space)

    def env_fn(render_mode=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        return gym.make(env_id, **kwargs)

    ppo_returns, _ = train_agent(
        env_fn, ppo_agent, episodes=episodes, render_every=render_every, log_every=25
    )
    a2c_returns, _ = train_agent(
        env_fn, a2c_agent, episodes=episodes, render_every=render_every, log_every=25
    )

    plot_learning_curves(
        {"PPO": ppo_returns, "A2C": a2c_returns},
        window=10,
        title=env_id,
    )

    base_dir = "videos/lunarlander-continuous"
    ppo_video = record_agent_video(env_fn, ppo_agent, video_dir=base_dir, prefix="ppo")
    a2c_video = record_agent_video(env_fn, a2c_agent, video_dir=base_dir, prefix="a2c")

    print("Saved LunarLanderContinuous replays:")
    print(f"- PPO: {ppo_video}")
    print(f"- A2C: {a2c_video}")


if __name__ == "__main__":
    run_discrete()
