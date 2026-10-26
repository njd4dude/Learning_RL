"""CLI entrypoint tying the Unit 3 DQN / RL Zoo pipeline together.

Usage:
    python main.py train [--env space-invaders|beam-rider] [-n N]
    python main.py evaluate [--exp-id 1]
    python main.py record [--exp-id 1] [--episodes 2]
    python main.py push --orga YOUR_USER [--repo-name NAME]
    python main.py pull [--env beam-rider] [--orga sb3]
"""

from __future__ import annotations

import argparse

import config
import evaluate as evaluate_mod
import hub
import record_video
import train as train_mod


def _resolve_env(name: str) -> str:
    if name in config.ENVIRONMENTS:
        return config.ENVIRONMENTS[name]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a DQN agent with RL Zoo")
    p_train.add_argument(
        "--env",
        default=config.DEFAULT_ENVIRONMENT,
        help="Shortcut (space-invaders|beam-rider) or a full Gymnasium env id",
    )
    p_train.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_train.add_argument("-f", "--log-folder", default=config.LOG_FOLDER)
    p_train.add_argument("-c", "--conf", default=config.DEFAULT_CONFIG)
    p_train.add_argument(
        "-n",
        "--n-timesteps",
        type=int,
        default=None,
        help="Override n_timesteps from dqn.yml",
    )
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--no-progress", action="store_true")
    p_train.add_argument(
        "-tb",
        "--tensorboard-log",
        default=config.TENSORBOARD_LOG_FOLDER,
        help="TensorBoard log dir (default: tb/)",
    )
    p_train.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging and auto-start",
    )
    p_train.add_argument(
        "--no-auto-tensorboard",
        action="store_true",
        help="Write TB logs but do not auto-launch TensorBoard",
    )
    p_train.add_argument(
        "--device",
        default=config.DEVICE,
        help='Torch device (default: auto → CUDA when available)',
    )
    p_train.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default=config.VEC_ENV,
        help="VecEnv type (default: subproc, for n_envs > 1 in dqn.yml)",
    )
    p_train.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Evaluate every N timesteps (Zoo default 25000; -1 disables)",
    )
    p_train.add_argument(
        "-i",
        "--trained-agent",
        default=None,
        help=(
            "Resume from a .zip in-place (same run folder, continuing "
            "timestep counter; -n is additional steps)"
        ),
    )

    p_eval = sub.add_parser("evaluate", help="Score a saved model (no video)")
    p_eval.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_eval.add_argument("--env", default=config.DEFAULT_ENV)
    p_eval.add_argument("-f", "--folder", default=config.LOG_FOLDER)
    p_eval.add_argument("-n", "--n-timesteps", type=int, default=config.ENJOY_TIMESTEPS)
    p_eval.add_argument("--exp-id", type=int, default=0)
    p_eval.add_argument("--load-best", action="store_true")
    p_eval.add_argument("--deterministic", action="store_true")

    p_rec = sub.add_parser("record", help="Save mp4 replays under videos/")
    p_rec.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_rec.add_argument("--env", default=config.DEFAULT_ENV)
    p_rec.add_argument("-f", "--folder", default=config.LOG_FOLDER)
    p_rec.add_argument("--exp-id", type=int, default=0)
    p_rec.add_argument("-n", "--n-timesteps", type=int, default=config.ENJOY_TIMESTEPS)
    p_rec.add_argument("--episodes", type=int, default=config.VIDEO_EPISODES)
    p_rec.add_argument("--out-dir", default=config.VIDEO_FOLDER)
    p_rec.add_argument("--fps", type=int, default=config.VIDEO_FPS)
    p_rec.add_argument("--load-best", action="store_true")
    p_rec.add_argument("--deterministic", action="store_true")

    p_push = sub.add_parser("push", help="Upload a trained agent to the Hub")
    p_push.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_push.add_argument("--env", default=config.DEFAULT_ENV)
    p_push.add_argument("-f", "--folder", default=config.LOG_FOLDER)
    p_push.add_argument("-orga", "--organization", required=True)
    p_push.add_argument("-name", "--repo-name", default=None)
    p_push.add_argument("--exp-id", type=int, default=0)
    p_push.add_argument("--load-best", action="store_true")

    p_pull = sub.add_parser("pull", help="Download a Hub model locally")
    p_pull.add_argument("--algo", default=config.DEFAULT_ALGO)
    p_pull.add_argument("--env", default=config.BEAMRIDER_ENV)
    p_pull.add_argument("-f", "--folder", default=config.TRAINED_FOLDER)
    p_pull.add_argument("-orga", "--organization", default=config.BEAMRIDER_ORGA)
    p_pull.add_argument("-name", "--repo-name", default=None)
    p_pull.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        tb = None if args.no_tensorboard else (args.tensorboard_log or None)
        train_mod.train(
            algo=args.algo,
            env_id=_resolve_env(args.env),
            log_folder=args.log_folder,
            conf_file=args.conf,
            n_timesteps=args.n_timesteps,
            progress=not args.no_progress,
            seed=args.seed,
            tensorboard_log=tb,
            auto_tensorboard=(
                False if args.no_auto_tensorboard or args.no_tensorboard else None
            ),
            device=args.device,
            vec_env=args.vec_env,
            eval_freq=args.eval_freq,
            trained_agent=args.trained_agent,
        )
    elif args.command == "evaluate":
        evaluate_mod.evaluate(
            algo=args.algo,
            env_id=args.env,
            folder=args.folder,
            n_timesteps=args.n_timesteps,
            exp_id=args.exp_id,
            load_best=args.load_best,
            deterministic=args.deterministic,
        )
    elif args.command == "record":
        episodes = None if args.episodes == 0 else args.episodes
        record_video.record(
            folder=args.folder,
            algo=args.algo,
            env_id=args.env,
            exp_id=args.exp_id,
            n_timesteps=args.n_timesteps,
            episodes=episodes,
            out_dir=args.out_dir,
            fps=args.fps,
            load_best=args.load_best,
            deterministic=args.deterministic,
        )
    elif args.command == "push":
        hub.push_to_hub(
            algo=args.algo,
            env_id=args.env,
            folder=args.folder,
            organization=args.organization,
            repo_name=args.repo_name,
            exp_id=args.exp_id,
            load_best=args.load_best,
        )
    elif args.command == "pull":
        hub.pull_from_hub(
            algo=args.algo,
            env_id=args.env,
            folder=args.folder,
            organization=args.organization,
            repo_name=args.repo_name,
            force=args.force,
        )


if __name__ == "__main__":
    main()
