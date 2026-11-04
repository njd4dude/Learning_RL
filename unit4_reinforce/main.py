"""CLI entrypoint tying the REINFORCE pipeline together.

Usage:
    python main.py train [--env cartpole|pixelcopter|pixelcopter-pixels]
    python main.py train --resume <model_path> [--episodes N] [--lr RATE]
    python main.py evaluate [model_path]
    python main.py record [model_path] [out_path] [--episodes N]
    python main.py visualize [model_path] [out_path]
    python main.py visualize --untrained [out_path]
    python main.py push <model_path> [repo_id]
    python main.py pull <repo_id>

model_path defaults to the newest runs/*/model.pt. repo_id defaults to
<config.USERNAME>/<the environment's model_name>.
"""

import argparse

import torch

import config
import evaluate as evaluate_mod
import hub
import record_video
import train as train_mod
import visualize as visualize_mod


def _default_repo_id(model_path: str) -> str:
    env_name = torch.load(model_path, weights_only=True)["env_name"]
    return f"{config.USERNAME}/{config.ENVIRONMENTS[env_name]['model_name']}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a REINFORCE agent")
    p_train.add_argument(
        "--env",
        choices=list(config.ENVIRONMENTS),
        default=config.DEFAULT_ENVIRONMENT,
        help="Which environment to train on. Ignored with --resume.",
    )
    p_train.add_argument("--resume", metavar="MODEL_PATH", help="Continue from a saved model.pt or checkpoint.pt.")
    p_train.add_argument("--episodes", type=int, help="Episodes to train this session.")
    p_train.add_argument("--lr", type=float, help="Learning rate, replacing the saved or configured one.")

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("model_path", nargs="?", default=None)

    p_rec = sub.add_parser("record", help="Record replay video(s) of a saved model")
    p_rec.add_argument("model_path", nargs="?", default=None)
    p_rec.add_argument("out_path", nargs="?", default=None)
    p_rec.add_argument(
        "--episodes",
        type=int,
        default=config.VIDEO_EPISODES,
        help="How many episodes to record (default: config.VIDEO_EPISODES).",
    )

    p_vis = sub.add_parser("visualize", help="Video of what a CNN model sees and detects")
    p_vis.add_argument("model_path", nargs="?", default=None)
    p_vis.add_argument("out_path", nargs="?", default=None)
    p_vis.add_argument("--untrained", action="store_true", help="Use a fresh random network instead of a saved model.")

    p_push = sub.add_parser("push", help="Upload a saved model to the Hugging Face Hub")
    p_push.add_argument("model_path")
    p_push.add_argument("repo_id", nargs="?", default=None)

    p_pull = sub.add_parser("pull", help="Download + evaluate a model from the Hub")
    p_pull.add_argument("repo_id")

    args = parser.parse_args()

    if args.command == "train":
        train_mod.train(args.env, resume_from=args.resume, episodes=args.episodes, lr=args.lr)
    elif args.command == "evaluate":
        evaluate_mod.evaluate(args.model_path)
    elif args.command == "record":
        record_video.record(args.model_path, args.out_path, episodes=args.episodes)
    elif args.command == "visualize":
        if args.untrained and args.model_path and not args.out_path:
            # With --untrained there's no model, so a lone path is the output.
            args.model_path, args.out_path = None, args.model_path
        visualize_mod.visualize(args.model_path, args.out_path, untrained=args.untrained)
    elif args.command == "push":
        hub.push_to_hub(args.model_path, args.repo_id or _default_repo_id(args.model_path))
    elif args.command == "pull":
        hub.pull_from_hub(args.repo_id)


if __name__ == "__main__":
    main()
