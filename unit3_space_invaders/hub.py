"""Push a trained Zoo agent to the Hugging Face Hub, or pull one down.

Usage:
    python hub.py push --repo-name dqn-SpaceInvadersNoFrameskip-v4 --orga YOUR_USER
    python hub.py pull --env BeamRiderNoFrameskip-v4 --orga sb3 -f rl_trained/
"""

from __future__ import annotations

import argparse

import config
from zoo import run_zoo_checked


def push_to_hub(
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.DEFAULT_ENV,
    folder: str = config.LOG_FOLDER,
    organization: str = "",
    repo_name: str | None = None,
    exp_id: int = 0,
    load_best: bool = False,
    n_timesteps: int = config.ENJOY_TIMESTEPS,
) -> None:
    """Evaluate, record a Hub replay, and upload via `rl_zoo3.push_to_hub`."""
    if not organization:
        raise ValueError("organization / Hugging Face username is required (--orga)")

    args = [
        "--algo",
        algo,
        "--env",
        env_id,
        "-f",
        folder,
        "-orga",
        organization,
        "--exp-id",
        str(exp_id),
        "-n",
        str(n_timesteps),
    ]
    if repo_name:
        args.extend(["-name", repo_name])
    if load_best:
        args.append("--load-best")

    run_zoo_checked("rl_zoo3.push_to_hub", args)


def pull_from_hub(
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.BEAMRIDER_ENV,
    folder: str = config.TRAINED_FOLDER,
    organization: str = config.BEAMRIDER_ORGA,
    repo_name: str | None = None,
    force: bool = False,
) -> None:
    """Download a Hub-trained agent into `folder` via `rl_zoo3.load_from_hub`."""
    args = [
        "--algo",
        algo,
        "--env",
        env_id,
        "-f",
        folder,
        "-orga",
        organization,
    ]
    if repo_name:
        args.extend(["-name", repo_name])
    if force:
        args.append("--force")

    run_zoo_checked("rl_zoo3.load_from_hub", args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="Upload a local Zoo run to the Hub")
    p_push.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_push.add_argument("--env", default=config.DEFAULT_ENV)
    p_push.add_argument("-f", "--folder", default=config.LOG_FOLDER)
    p_push.add_argument("-orga", "--organization", required=True)
    p_push.add_argument("-name", "--repo-name", default=None)
    p_push.add_argument("--exp-id", type=int, default=0)
    p_push.add_argument("--load-best", action="store_true")
    p_push.add_argument("-n", "--n-timesteps", type=int, default=config.ENJOY_TIMESTEPS)

    p_pull = sub.add_parser("pull", help="Download a Hub model into a local folder")
    p_pull.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_pull.add_argument("--env", default=config.BEAMRIDER_ENV)
    p_pull.add_argument("-f", "--folder", default=config.TRAINED_FOLDER)
    p_pull.add_argument("-orga", "--organization", default=config.BEAMRIDER_ORGA)
    p_pull.add_argument("-name", "--repo-name", default=None)
    p_pull.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "push":
        push_to_hub(
            algo=args.algo,
            env_id=args.env,
            folder=args.folder,
            organization=args.organization,
            repo_name=args.repo_name,
            exp_id=args.exp_id,
            load_best=args.load_best,
            n_timesteps=args.n_timesteps,
        )
    elif args.command == "pull":
        pull_from_hub(
            algo=args.algo,
            env_id=args.env,
            folder=args.folder,
            organization=args.organization,
            repo_name=args.repo_name,
            force=args.force,
        )


if __name__ == "__main__":
    main()
