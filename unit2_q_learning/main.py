"""CLI entrypoint tying the Q-Learning pipeline together.

Usage:
    python main.py train [--env frozenlake|frozenlake-slippery|taxi]
    python main.py evaluate [model_path]
    python main.py record [model_path] [out_path]
    python main.py push <model_path> <repo_id>
    python main.py pull <repo_id> <filename>
"""

import argparse

import config
import evaluate as evaluate_mod
import hub
import record_video
import train as train_mod


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a Q-Learning agent and pickle its Q-table")
    p_train.add_argument(
        "--env",
        choices=list(config.ENVIRONMENTS),
        default=config.DEFAULT_ENVIRONMENT,
        help="Which environment to train on.",
    )

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model")
    p_eval.add_argument("model_path", nargs="?", default=config.FROZENLAKE_MODEL_PATH)

    p_rec = sub.add_parser("record", help="Record replay videos of a saved model")
    p_rec.add_argument("model_path", nargs="?", default=config.FROZENLAKE_MODEL_PATH)
    p_rec.add_argument("out_path", nargs="?", default=None)

    p_push = sub.add_parser("push", help="Upload a saved model to the Hugging Face Hub")
    p_push.add_argument("model_path")
    p_push.add_argument("repo_id")

    p_pull = sub.add_parser("pull", help="Download + evaluate a model from the Hub")
    p_pull.add_argument("repo_id")
    p_pull.add_argument("filename")

    args = parser.parse_args()

    if args.command == "train":
        train_mod.train(args.env)
    elif args.command == "evaluate":
        evaluate_mod.evaluate(args.model_path)
    elif args.command == "record":
        record_video.record(args.model_path, args.out_path)
    elif args.command == "push":
        hub.push_to_hub(args.model_path, args.repo_id)
    elif args.command == "pull":
        hub.pull_from_hub(args.repo_id, args.filename)


if __name__ == "__main__":
    main()
