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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "List of random seeds. Provide a single value to reuse it or combine with "
            "--seed-mode=count to interpret it as the number of seeds to run."
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=["list", "count"],
        default="list",
        help="Interpretation for a single --seeds value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_fn = EXPERIMENTS[args.experiment]
    kwargs = {}
    if args.episodes is not None:
        kwargs["episodes"] = args.episodes
    if args.seeds is not None:
        kwargs["seeds"] = args.seeds
        kwargs["seed_mode"] = args.seed_mode
    run_fn(**kwargs)


if __name__ == "__main__":
    main()
