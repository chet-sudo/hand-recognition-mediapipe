"""Command-line entry point for running RL lab experiments."""
from __future__ import annotations

import argparse

from experiments import frozenlake, cartpole, lunarlander, pusher


EXPERIMENTS = {
    "frozenlake": frozenlake.run,
    "cartpole": cartpole.run,
    "lunarlander-discrete": lunarlander.run_discrete,
    "lunarlander-continuous": lunarlander.run_continuous,
    "pusher": pusher.run,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Educational RL lab")
    parser.add_argument("experiment", choices=EXPERIMENTS.keys(), help="Which preset experiment to run")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fn = EXPERIMENTS[args.experiment]
    kwargs = {}
    if args.episodes is not None:
        kwargs["episodes"] = args.episodes
    run_fn(**kwargs)


if __name__ == "__main__":
    main()
