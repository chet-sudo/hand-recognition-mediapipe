# Reinforcement Learning Lab

An educational collection of tabular and neural RL agents with small
experiments for Gymnasium environments. The code prioritizes clarity and
inline commentary over raw performance so you can study how each
algorithm works.

## Project layout

- `envs/`: Environment factory helpers.
- `agents/`: Algorithm implementations with a shared `Agent` interface.
  - `tabular/`: Q-learning and SARSA for discrete state spaces.
  - `deep/`: DQN with replay buffer and target network.
  - `actor_critic/`: A2C and PPO for discrete and continuous control.
- `utils/`: Shared helpers (replay buffer, neural networks, plotting, training loops).
- `experiments/`: Ready-to-run experiment scripts for the supported environments.
- `main.py`: CLI entry point for launching experiments.

## Running experiments

1. Install dependencies (Box2D and MuJoCo extras are included):

```bash
pip install -r requirements.txt
```

2. Launch an experiment, e.g.:

```bash
python main.py frozenlake
python main.py cartpole --episodes 400
python main.py lunarlander-discrete
python main.py lunarlander-continuous
python main.py pusher
```

Plots are saved under `plots/` (and optionally displayed interactively if
you set `show=True` in the plotting helper) so runs do not hang in headless
environments.

## Algorithms

- **Tabular Q-learning** (off-policy) and **SARSA** (on-policy) for `FrozenLake-v1`.
- **DQN** with replay buffer, target network, and epsilon-greedy decay for `CartPole-v1` and `LunarLander-v2`.
- **A2C** and **PPO** with Gaussian policies for continuous control on `LunarLanderContinuous-v2` and `Pusher-v4`.

Each implementation contains docstrings and comments describing the
update rules, policy/value roles, and stabilization tricks (replay
buffers, target networks, clipping, entropy bonuses).
