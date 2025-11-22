"""Compare DQN, A2C, and PPO on CartPole-v1."""
from __future__ import annotations

import gymnasium as gym

from agents.deep.dqn import DQNAgent
from agents.actor_critic.a2c import A2CAgent
from agents.actor_critic.ppo import PPOAgent
from utils.training import record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run(episodes: int = 300, render_every: int | None = 100):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    dqn_agent = DQNAgent(obs_dim, n_actions)
    a2c_agent = A2CAgent(obs_dim, env.action_space)
    ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=512)

    def env_fn(render_mode=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        return gym.make("CartPole-v1", **kwargs)

    dqn_returns, _ = train_agent(
        env_fn, dqn_agent, episodes=episodes, render_every=render_every, log_every=20
    )
    a2c_returns, _ = train_agent(
        env_fn, a2c_agent, episodes=episodes, render_every=render_every, log_every=20
    )
    ppo_returns, _ = train_agent(
        env_fn, ppo_agent, episodes=episodes, render_every=render_every, log_every=20
    )

    plot_learning_curves(
        {"DQN": dqn_returns, "A2C": a2c_returns, "PPO": ppo_returns},
        window=10,
        title="CartPole-v1",
    )

    dqn_video = record_agent_video(env_fn, dqn_agent, video_dir="videos/cartpole", prefix="dqn")
    a2c_video = record_agent_video(env_fn, a2c_agent, video_dir="videos/cartpole", prefix="a2c")
    ppo_video = record_agent_video(env_fn, ppo_agent, video_dir="videos/cartpole", prefix="ppo")

    print("Saved CartPole replays:")
    print(f"- DQN: {dqn_video}")
    print(f"- A2C: {a2c_video}")
    print(f"- PPO: {ppo_video}")


if __name__ == "__main__":
    run()
