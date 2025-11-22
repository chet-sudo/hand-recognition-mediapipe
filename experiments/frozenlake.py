"""Run tabular methods on FrozenLake-v1 for side-by-side comparison."""
from __future__ import annotations

import gymnasium as gym

from agents.tabular.q_learning import QLearningAgent
from agents.tabular.sarsa import SarsaAgent
from utils.training import record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run(episodes: int = 500, render_every: int | None = 100):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_agent = QLearningAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)
    sarsa_agent = SarsaAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)

    def env_fn(render_mode=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        return gym.make("FrozenLake-v1", is_slippery=False, **kwargs)

    q_returns, _ = train_agent(
        env_fn, q_agent, episodes=episodes, render_every=render_every, log_every=20
    )
    sarsa_returns, _ = train_agent(
        env_fn, sarsa_agent, episodes=episodes, render_every=render_every, log_every=20
    )

    plot_learning_curves(
        {"Q-learning": q_returns, "SARSA": sarsa_returns}, window=10, title="FrozenLake"
    )

    spec = gym.spec("FrozenLake-v1")
    supports_rgb = bool(getattr(spec, "render_modes", None)) and "rgb_array" in spec.render_modes

    if supports_rgb:
        q_video = record_agent_video(
            env_fn, q_agent, video_dir="videos/frozenlake", prefix="q-learning"
        )
        sarsa_video = record_agent_video(
            env_fn, sarsa_agent, video_dir="videos/frozenlake", prefix="sarsa"
        )

        print("Saved FrozenLake replays:")
        print(f"- Q-learning: {q_video}")
        print(f"- SARSA: {sarsa_video}")
    else:
        print(
            "Skipping FrozenLake replay capture: gymnasium 0.29.1 only supports "
            "'human'/'ansi' rendering for this environment."
        )


if __name__ == "__main__":
    run()
