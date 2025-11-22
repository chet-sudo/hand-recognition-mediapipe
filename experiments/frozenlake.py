"""Run tabular methods on FrozenLake-v1 for side-by-side comparison."""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from agents.tabular.q_learning import QLearningAgent
from agents.tabular.sarsa import SarsaAgent
from experiments import aggregate_returns, resolve_seeds, write_experiment_summary
from utils.logger import MetricsLogger
from utils.training import evaluate_agent, record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run(
    episodes: int = 500,
    render_every: int | None = 100,
    seeds: list[int] | int | None = None,
    seed_mode: str = "list",
    experiment_label: str = "frozenlake",
    eval_episodes: int = 20,
):
    env_id = "FrozenLake-v1"
    seed_values = resolve_seeds(seeds, seed_mode=seed_mode)
    aggregate_training: dict[str, list[list[float]]] = {"Q-learning": [], "SARSA": []}
    eval_scores: dict[str, list[float]] = {"Q-learning": [], "SARSA": []}

    def env_fn(render_mode=None, seed=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        env_instance = gym.make(env_id, is_slippery=False, **kwargs)
        if seed is not None:
            env_instance.reset(seed=seed)
            env_instance.action_space.seed(seed)
        return env_instance

    for seed in seed_values:
        env = gym.make(env_id, is_slippery=False)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        env.close()

        q_agent = QLearningAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)
        sarsa_agent = SarsaAgent(n_states, n_actions, alpha=0.8, gamma=0.99, epsilon=0.1)

        q_hparams = {"alpha": q_agent.alpha, "gamma": q_agent.gamma, "epsilon": q_agent.epsilon}
        sarsa_hparams = {
            "alpha": sarsa_agent.alpha,
            "gamma": sarsa_agent.gamma,
            "epsilon": sarsa_agent.epsilon,
        }

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="Q-learning",
            environment=env_id,
            seed=seed,
            hyperparameters=q_hparams,
            run_label=experiment_label,
        ) as q_logger:
            q_returns, _ = train_agent(
                env_fn,
                q_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=20,
                seed=seed,
                logger=q_logger,
            )
            q_eval = evaluate_agent(env_fn, q_agent, episodes=eval_episodes, seed=seed)
            q_logger.log_summary(evaluation_return=q_eval, training_episodes=episodes)

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="SARSA",
            environment=env_id,
            seed=seed,
            hyperparameters=sarsa_hparams,
            run_label=experiment_label,
        ) as sarsa_logger:
            sarsa_returns, _ = train_agent(
                env_fn,
                sarsa_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=20,
                seed=seed,
                logger=sarsa_logger,
            )
            sarsa_eval = evaluate_agent(env_fn, sarsa_agent, episodes=eval_episodes, seed=seed)
            sarsa_logger.log_summary(evaluation_return=sarsa_eval, training_episodes=episodes)

        aggregate_training["Q-learning"].append(q_returns)
        aggregate_training["SARSA"].append(sarsa_returns)
        eval_scores["Q-learning"].append(q_eval)
        eval_scores["SARSA"].append(sarsa_eval)

        spec = gym.spec(env_id)
        supports_rgb = bool(getattr(spec, "render_modes", None)) and "rgb_array" in spec.render_modes

        if supports_rgb:
            q_video = record_agent_video(
                env_fn, q_agent, video_dir="videos/frozenlake", prefix=f"q-learning-seed-{seed}"
            )
            sarsa_video = record_agent_video(
                env_fn, sarsa_agent, video_dir="videos/frozenlake", prefix=f"sarsa-seed-{seed}"
            )

            print("Saved FrozenLake replays:")
            print(f"- Q-learning (seed {seed}): {q_video}")
            print(f"- SARSA (seed {seed}): {sarsa_video}")
        else:
            print(
                "Skipping FrozenLake replay capture: gymnasium 0.29.1 only supports "
                "'human'/'ansi' rendering for this environment."
            )

    mean_returns = {}
    std_returns = {}
    eval_mean = {k: float(sum(v) / len(v)) for k, v in eval_scores.items()}
    eval_std = {k: float(np.std(v)) for k, v in eval_scores.items()}

    for algo, runs in aggregate_training.items():
        mean, std = aggregate_returns(runs)
        mean_returns[algo] = mean
        std_returns[algo] = std

    plot_path = f"results/{experiment_label}/frozenlake.png"
    plot_learning_curves(
        mean_returns,
        std_results=std_returns,
        evaluation_scores=eval_mean,
        window=10,
        title="FrozenLake",
        save_path=plot_path,
    )
    write_experiment_summary(experiment_label, mean_returns, std_returns, eval_mean, eval_std)


if __name__ == "__main__":
    run()
