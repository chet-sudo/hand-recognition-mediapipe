"""PPO experiment for the MuJoCo Pusher-v4 environment."""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from agents.actor_critic.ppo import PPOAgent
from experiments import aggregate_returns, resolve_seeds, write_experiment_summary
from utils.logger import MetricsLogger
from utils.training import evaluate_agent, record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run(
    episodes: int = 300,
    render_every: int | None = 75,
    seeds: list[int] | int | None = None,
    seed_mode: str = "list",
    experiment_label: str = "pusher",
    eval_episodes: int = 5,
):
    env_id = "Pusher-v4"
    seed_values = resolve_seeds(seeds, seed_mode=seed_mode)
    aggregate_training: dict[str, list[list[float]]] = {"PPO": []}
    eval_scores: dict[str, list[float]] = {"PPO": []}

    def env_fn(render_mode=None, seed=None):
        kwargs = {"render_mode": render_mode} if render_mode is not None else {}
        env_instance = gym.make(env_id, **kwargs)
        if seed is not None:
            env_instance.reset(seed=seed)
            env_instance.action_space.seed(seed)
        return env_instance

    for seed in seed_values:
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        env.close()

        ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=2048)
        ppo_hparams = {
            "gamma": ppo_agent.gamma,
            "lr": ppo_agent.lr,
            "batch_size": ppo_agent.batch_size,
            "clip_ratio": ppo_agent.clip_ratio,
        }

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="PPO",
            environment=env_id,
            seed=seed,
            hyperparameters=ppo_hparams,
            run_label=experiment_label,
        ) as ppo_logger:
            ppo_returns, _ = train_agent(
                env_fn,
                ppo_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=20,
                seed=seed,
                logger=ppo_logger,
            )
            ppo_eval = evaluate_agent(env_fn, ppo_agent, episodes=eval_episodes, seed=seed)
            ppo_logger.log_summary(evaluation_return=ppo_eval, training_episodes=episodes)

        aggregate_training["PPO"].append(ppo_returns)
        eval_scores["PPO"].append(ppo_eval)

        video_path = record_agent_video(env_fn, ppo_agent, video_dir="videos/pusher", prefix=f"ppo-seed-{seed}")
        print("Saved Pusher replay:")
        print(f"- PPO (seed {seed}): {video_path}")

    mean_returns = {}
    std_returns = {}
    eval_mean = {k: float(sum(v) / len(v)) for k, v in eval_scores.items()}
    eval_std = {k: float(np.std(v)) for k, v in eval_scores.items()}

    for algo, runs in aggregate_training.items():
        mean, std = aggregate_returns(runs)
        mean_returns[algo] = mean
        std_returns[algo] = std

    plot_path = f"results/{experiment_label}/pusher.png"
    plot_learning_curves(
        mean_returns,
        std_results=std_returns,
        evaluation_scores=eval_mean,
        window=5,
        title=env_id,
        save_path=plot_path,
    )
    write_experiment_summary(experiment_label, mean_returns, std_returns, eval_mean, eval_std)


if __name__ == "__main__":
    run()
