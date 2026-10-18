"""CLI entrypoint tying the LunarLander pipeline together.

Usage:
    python main.py train
    python main.py evaluate [model_path]
    python main.py record [model_path]
    python main.py push
    python main.py pull <repo_id> <filename>
"""

import argparse

import config
import evaluate as evaluate_mod
import hub
import record_video
import train as train_mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train a new PPO agent and save it to disk")

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved agent")
    p_eval.add_argument("model_path", nargs="?", default=config.MODEL_NAME)

    p_rec = sub.add_parser("record", help="Record replay videos of a saved agent")
    p_rec.add_argument("model_path", nargs="?", default=config.MODEL_NAME)

    sub.add_parser("push", help="Upload the saved agent to the Hugging Face Hub")

    p_pull = sub.add_parser("pull", help="Download + evaluate a model from the Hub")
    p_pull.add_argument("repo_id")
    p_pull.add_argument("filename")

    args = parser.parse_args()

    if args.command == "train":
        train_mod.train()
    elif args.command == "evaluate":
        evaluate_mod.evaluate(args.model_path)
    elif args.command == "record":
        record_video.record(args.model_path)
    elif args.command == "push":
        hub.push_to_hub()
    elif args.command == "pull":
        hub.pull_from_hub(args.repo_id, args.filename)


if __name__ == "__main__":
    main()
