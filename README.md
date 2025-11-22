**Prompt for Codex:**

I want you to design and implement a small but clean “Reinforcement Learning lab” project in Python so that I can *learn* from the implementation.

### Overall goals

* Implement a set of **clean, well-commented RL algorithms** that are easy to study.
* Use **Gymnasium built-in environments** only (no custom envs).
* Support training algorithms on:

  * `FrozenLake-v1`
  * `CartPole-v1`
  * `LunarLander-v2`
  * `LunarLanderContinuous-v2`
  * `Pusher-v4` (MuJoCo)
* Provide a way to **run multiple algorithms on the same environment** and **compare their performance side by side** (learning curves, metrics).
* Prioritize **clarity, structure, and comments** over extreme performance. This is for learning and studying how each algorithm works.

You are free to refine and improve the design if you see better approaches, as long as it stays educational and clean.

---

### Tech stack and dependencies

Use Python and the standard RL ecosystem:

* Gymnasium for environments (with the appropriate extras for classic control, Box2D, and MuJoCo).
* NumPy for tabular value-based methods.
* PyTorch for function approximation and deep RL (DQN, A2C, PPO).
* A simple plotting library (e.g., Matplotlib) for learning curves.
* Optional: basic experiment logging / tracking (e.g., TensorBoard or Weights & Biases) if it can be integrated cleanly.

You can add small utility dependencies if they really help with clarity, but keep the stack reasonably minimal.

---

### Project structure (you can adjust if you have a better idea)

Organize the project into clear modules; for example:

* An `envs` module:

  * A function that creates Gymnasium environments by ID.
  * Optional wrappers (state normalization, seeding, etc.).
* An `agents` package:

  * A base agent interface that all algorithms implement.
  * Tabular agents (e.g., Q-learning, SARSA).
  * Deep RL agents (e.g., DQN, A2C, PPO).
* A `utils` package:

  * Replay buffer implementation (for DQN / off-policy methods).
  * Neural network definitions for policies and value functions (PyTorch).
  * Plotting utilities to compare algorithms (e.g., learning curves).
  * Simple metrics and logging helpers.
* An `experiments` package:

  * Scripts or functions that set up a specific environment and run one or more agents on it.
  * For example:

    * FrozenLake: tabular algorithms comparison.
    * CartPole: DQN vs A2C vs PPO.
    * LunarLander: similar comparisons.
    * Pusher: continuous-control PPO experiment.
* Configuration files (YAML/JSON) for hyperparameters and experiment settings.
* A simple `main` entry point or CLI for running experiments with different envs and agents.

You can adapt this structure if you know a cleaner or more idiomatic layout, but keep things modular and easy to navigate.

---

### Agent interface and training logic

Define a **common agent interface** so that training code is algorithm-agnostic.

Conceptually, each agent should support something like:

* Starting a new episode given the initial observation and returning the first action.
* Updating its internal state on each step given reward, next observation, and done flag, and returning the next action.
* Doing any cleanup at the end of an episode.
* A separate “act” method for evaluation (e.g., no exploration).

The training loop should:

* Work with any Gymnasium environment and any agent that implements the interface.
* Run for a configurable number of episodes.
* Log per-episode returns and any other useful stats.
* Optionally support multiple random seeds to compute averages and variances.

Also provide an evaluation loop that runs an agent without exploration, over several episodes, to estimate its final performance.

---

### Algorithms and which environments they should run on

Implement clean, well-documented versions of at least the following:

1. **Tabular methods** (NumPy only):

   * Q-Learning (off-policy, value-based)
   * SARSA (on-policy, value-based)
   * These should be used on `FrozenLake-v1` (discrete states and actions).
   * Make sure to explain in comments how the TD target differs between Q-learning and SARSA and what “on-policy” vs “off-policy” means.

2. **Deep value-based method**:

   * DQN (Deep Q-Network), suitable for:

     * `CartPole-v1`
     * `LunarLander-v2` (discrete)
   * Include:

     * Replay buffer
     * Target network
     * Epsilon-greedy exploration with decay
   * Comments should clearly describe:

     * Why we use a replay buffer
     * Why we use a target network
     * How the Q-learning TD target is computed with a neural network.

3. **Actor-critic / policy-gradient methods**:

   * A2C (Advantage Actor-Critic) or similar synchronous actor-critic.
   * PPO (Proximal Policy Optimization) with a clipped surrogate objective.
   * Use these on:

     * `CartPole-v1` (discrete actions)
     * `LunarLander-v2`
     * `LunarLanderContinuous-v2` and `Pusher-v4` (continuous actions) using an appropriate Gaussian policy.
   * Clearly explain via comments:

     * The role of the policy (actor) and value function (critic).
     * How advantages are computed.
     * The PPO clipping idea and why it stabilizes training.
     * For continuous action spaces, explain how actions are sampled from a Gaussian and mapped to the environment’s action bounds.

You can add more algorithms if you think they are useful, but the above are the core set I want.

---

### Comparison and visualization

The main outcome I care about:

* Being able to run **multiple algorithms on the same environment** and see their performance **side by side**.

Please:

* Implement a simple way to:

  * Train one or more agents on a given environment.
  * Save episode returns and other relevant metrics.
  * Plot learning curves for different agents on the same figure.
* For example:

  * Q-learning vs SARSA on FrozenLake.
  * DQN vs A2C vs PPO on CartPole.
  * PPO on LunarLander and Pusher with different hyperparameter settings.
* Optionally, support multiple seeds and show mean ± standard deviation bands.

The plotting and logging code should be separated from the core agent logic so the RL algorithms remain clean and focused.

---

### Documentation, comments, and readability

This project is primarily for **learning and studying** RL, so:

* Write **clear docstrings and inline comments** explaining:

  * The update rules (with plain language, not just formulas).
  * Why each algorithm is on-policy or off-policy.
  * Why certain tricks are used (replay buffer, target networks, clipping, entropy bonus, etc.).
* Favor **simple, readable implementations** over highly optimized or ultra-generic frameworks.
* It is fine to duplicate some code across agents if it makes each algorithm easier to understand in isolation.

---

### Freedom to improve

You are allowed and encouraged to:

* Improve the project structure if you know of more idiomatic or educational layouts.
* Add lightweight abstractions if they improve clarity, not just generality.
* Include extra tools (like simple config management or logging helpers) if they make it easier to run and compare experiments.

Just keep the focus on:

* Clear implementations.
* Good separation of concerns.
* Easy side-by-side comparison of algorithms on Gymnasium environments.

At the end, I want to be able to:

* Open each agent implementation and understand how it works.
* Run experiments that compare algorithms on the same task.
* Look at the results and develop an intuition for how and why different RL algorithms behave differently.

Please implement the full project accordingly.
