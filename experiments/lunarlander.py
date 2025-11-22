"""Experiments for LunarLander variants using DQN/A2C/PPO."""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from agents.deep.dqn import DQNAgent
from agents.actor_critic.a2c import A2CAgent
from agents.actor_critic.ppo import PPOAgent
from experiments import aggregate_returns, resolve_seeds, write_experiment_summary
from utils.logger import MetricsLogger
from utils.training import evaluate_agent, record_agent_video, train_agent
from utils.plotting import plot_learning_curves


def run_discrete(
    episodes: int = 500,
    render_every: int | None = 100,
    seeds: list[int] | int | None = None,
    seed_mode: str = "list",
    experiment_label: str = "lunarlander-discrete",
    eval_episodes: int = 5,
):
    env_id = "LunarLander-v2"
    seed_values = resolve_seeds(seeds, seed_mode=seed_mode)
    aggregate_training: dict[str, list[list[float]]] = {"DQN": [], "A2C": [], "PPO": []}
    eval_scores: dict[str, list[float]] = {"DQN": [], "A2C": [], "PPO": []}

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
        n_actions = env.action_space.n
        env.close()

        dqn_agent = DQNAgent(obs_dim, n_actions)
        a2c_agent = A2CAgent(obs_dim, env.action_space)
        ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=1024)

        dqn_hparams = {
            "gamma": dqn_agent.gamma,
            "lr": dqn_agent.optimizer.param_groups[0]["lr"],
            "batch_size": dqn_agent.batch_size,
            "buffer_size": dqn_agent.buffer.capacity,
            "epsilon_start": dqn_agent.epsilon_start,
            "epsilon_end": dqn_agent.epsilon_end,
            "epsilon_decay": dqn_agent.epsilon_decay,
            "target_update": dqn_agent.target_update,
        }
        a2c_hparams = {"gamma": a2c_agent.gamma, "lr": a2c_agent.lr}
        ppo_hparams = {
            "gamma": ppo_agent.gamma,
            "lr": ppo_agent.lr,
            "batch_size": ppo_agent.batch_size,
            "clip_ratio": ppo_agent.clip_ratio,
        }

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="DQN",
            environment=env_id,
            seed=seed,
            hyperparameters=dqn_hparams,
            run_label=experiment_label,
        ) as dqn_logger:
            dqn_returns, _ = train_agent(
                env_fn,
                dqn_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=25,
                seed=seed,
                logger=dqn_logger,
            )
            dqn_eval = evaluate_agent(env_fn, dqn_agent, episodes=eval_episodes, seed=seed)
            dqn_logger.log_summary(evaluation_return=dqn_eval, training_episodes=episodes)

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="A2C",
            environment=env_id,
            seed=seed,
            hyperparameters=a2c_hparams,
            run_label=experiment_label,
        ) as a2c_logger:
            a2c_returns, _ = train_agent(
                env_fn,
                a2c_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=25,
                seed=seed,
                logger=a2c_logger,
            )
            a2c_eval = evaluate_agent(env_fn, a2c_agent, episodes=eval_episodes, seed=seed)
            a2c_logger.log_summary(evaluation_return=a2c_eval, training_episodes=episodes)

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
                log_every=25,
                seed=seed,
                logger=ppo_logger,
            )
            ppo_eval = evaluate_agent(env_fn, ppo_agent, episodes=eval_episodes, seed=seed)
            ppo_logger.log_summary(evaluation_return=ppo_eval, training_episodes=episodes)

        aggregate_training["DQN"].append(dqn_returns)
        aggregate_training["A2C"].append(a2c_returns)
        aggregate_training["PPO"].append(ppo_returns)
        eval_scores["DQN"].append(dqn_eval)
        eval_scores["A2C"].append(a2c_eval)
        eval_scores["PPO"].append(ppo_eval)

        base_dir = "videos/lunarlander-discrete"
        dqn_video = record_agent_video(env_fn, dqn_agent, video_dir=base_dir, prefix=f"dqn-seed-{seed}")
        a2c_video = record_agent_video(env_fn, a2c_agent, video_dir=base_dir, prefix=f"a2c-seed-{seed}")
        ppo_video = record_agent_video(env_fn, ppo_agent, video_dir=base_dir, prefix=f"ppo-seed-{seed}")

        print("Saved LunarLander-v2 replays:")
        print(f"- DQN (seed {seed}): {dqn_video}")
        print(f"- A2C (seed {seed}): {a2c_video}")
        print(f"- PPO (seed {seed}): {ppo_video}")

    mean_returns = {}
    std_returns = {}
    eval_mean = {k: float(sum(v) / len(v)) for k, v in eval_scores.items()}
    eval_std = {k: float(np.std(v)) for k, v in eval_scores.items()}

    for algo, runs in aggregate_training.items():
        mean, std = aggregate_returns(runs)
        mean_returns[algo] = mean
        std_returns[algo] = std

    plot_path = f"results/{experiment_label}/lunarlander_discrete.png"
    plot_learning_curves(
        mean_returns,
        std_results=std_returns,
        evaluation_scores=eval_mean,
        window=10,
        title="LunarLander-v2",
        save_path=plot_path,
    )
    write_experiment_summary(experiment_label, mean_returns, std_returns, eval_mean, eval_std)


def run_continuous(
    episodes: int = 500,
    render_every: int | None = 100,
    seeds: list[int] | int | None = None,
    seed_mode: str = "list",
    experiment_label: str = "lunarlander-continuous",
    eval_episodes: int = 5,
):
    env_id = "LunarLanderContinuous-v2"
    seed_values = resolve_seeds(seeds, seed_mode=seed_mode)
    aggregate_training: dict[str, list[list[float]]] = {"PPO": [], "A2C": []}
    eval_scores: dict[str, list[float]] = {"PPO": [], "A2C": []}

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

        ppo_agent = PPOAgent(obs_dim, env.action_space, batch_size=1024)
        a2c_agent = A2CAgent(obs_dim, env.action_space)

        ppo_hparams = {
            "gamma": ppo_agent.gamma,
            "lr": ppo_agent.lr,
            "batch_size": ppo_agent.batch_size,
            "clip_ratio": ppo_agent.clip_ratio,
        }
        a2c_hparams = {"gamma": a2c_agent.gamma, "lr": a2c_agent.lr}

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
                log_every=25,
                seed=seed,
                logger=ppo_logger,
            )
            ppo_eval = evaluate_agent(env_fn, ppo_agent, episodes=eval_episodes, seed=seed)
            ppo_logger.log_summary(evaluation_return=ppo_eval, training_episodes=episodes)

        with MetricsLogger(
            experiment=experiment_label,
            algorithm="A2C",
            environment=env_id,
            seed=seed,
            hyperparameters=a2c_hparams,
            run_label=experiment_label,
        ) as a2c_logger:
            a2c_returns, _ = train_agent(
                env_fn,
                a2c_agent,
                episodes=episodes,
                render_every=render_every,
                log_every=25,
                seed=seed,
                logger=a2c_logger,
            )
            a2c_eval = evaluate_agent(env_fn, a2c_agent, episodes=eval_episodes, seed=seed)
            a2c_logger.log_summary(evaluation_return=a2c_eval, training_episodes=episodes)

        aggregate_training["PPO"].append(ppo_returns)
        aggregate_training["A2C"].append(a2c_returns)
        eval_scores["PPO"].append(ppo_eval)
        eval_scores["A2C"].append(a2c_eval)

        base_dir = "videos/lunarlander-continuous"
        ppo_video = record_agent_video(env_fn, ppo_agent, video_dir=base_dir, prefix=f"ppo-seed-{seed}")
        a2c_video = record_agent_video(env_fn, a2c_agent, video_dir=base_dir, prefix=f"a2c-seed-{seed}")

        print("Saved LunarLanderContinuous replays:")
        print(f"- PPO (seed {seed}): {ppo_video}")
        print(f"- A2C (seed {seed}): {a2c_video}")

    mean_returns = {}
    std_returns = {}
    eval_mean = {k: float(sum(v) / len(v)) for k, v in eval_scores.items()}
    eval_std = {k: float(np.std(v)) for k, v in eval_scores.items()}

    for algo, runs in aggregate_training.items():
        mean, std = aggregate_returns(runs)
        mean_returns[algo] = mean
        std_returns[algo] = std

    plot_path = f"results/{experiment_label}/lunarlander_continuous.png"
    plot_learning_curves(
        mean_returns,
        std_results=std_returns,
        evaluation_scores=eval_mean,
        window=10,
        title=env_id,
        save_path=plot_path,
    )
    write_experiment_summary(experiment_label, mean_returns, std_returns, eval_mean, eval_std)


if __name__ == "__main__":
    run_discrete()
