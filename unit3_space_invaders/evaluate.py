"""Evaluate a trained Zoo agent (scores only, no window / no mp4).

Usage:
    python evaluate.py
    python evaluate.py --exp-id 1 -n 5000
    python evaluate.py --env BeamRiderNoFrameskip-v4 -f rl_trained/
"""

from __future__ import annotations

import argparse

import config
from zoo import run_zoo_checked


def evaluate(
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.DEFAULT_ENV,
    folder: str = config.LOG_FOLDER,
    n_timesteps: int = config.ENJOY_TIMESTEPS,
    exp_id: int = 0,
    load_best: bool = False,
    deterministic: bool = False,
) -> None:
    """Run `rl_zoo3.enjoy --no-render` against a local experiment folder."""
    args = [
        "--algo",
        algo,
        "--env",
        env_id,
        "-f",
        folder,
        "-n",
        str(n_timesteps),
        "--no-render",
        "--exp-id",
        str(exp_id),
    ]
    if load_best:
        args.append("--load-best")
    if deterministic:
        args.append("--deterministic")

    run_zoo_checked("rl_zoo3.enjoy", args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default=config.DEFAULT_ALGO)
    parser.add_argument("--env", default=config.DEFAULT_ENV)
    parser.add_argument("-f", "--folder", default=config.LOG_FOLDER)
    parser.add_argument("-n", "--n-timesteps", type=int, default=config.ENJOY_TIMESTEPS)
    parser.add_argument("--exp-id", type=int, default=0, help="0 = latest experiment")
    parser.add_argument("--load-best", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    evaluate(
        algo=args.algo,
        env_id=args.env,
        folder=args.folder,
        n_timesteps=args.n_timesteps,
        exp_id=args.exp_id,
        load_best=args.load_best,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
