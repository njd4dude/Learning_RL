"""Train a Q-Learning agent on FrozenLake or Taxi and save its Q-table.

Usage:
    python train.py                   # train FrozenLake (the default)
    python train.py --env taxi        # train Taxi-v3 instead
"""

import argparse
import pickle

import gymnasium as gym

import config
from q_learning import initialize_q_table, train as run_training


def build_env(env_id: str, env_kwargs: dict):
    return gym.make(env_id, render_mode="rgb_array", **env_kwargs)


def train(env_name: str = config.DEFAULT_ENVIRONMENT, save_path: str | None = None):
    """Train a fresh Q-table for `env_name` ("frozenlake" or "taxi") and pickle it.

    The saved file is a dict of hyperparameters + the Q-table, matching the
    course's `model` dictionary shape, so `evaluate.py`, `record_video.py`,
    and `hub.py` can all work from it without retraining.
    """
    settings = config.ENVIRONMENTS[env_name]
    env = build_env(settings["env_id"], settings["env_kwargs"])

    state_space = env.observation_space.n
    action_space = env.action_space.n
    print(f"{env_name}: {state_space} states, {action_space} actions")

    qtable = initialize_q_table(state_space, action_space)
    train_kwargs = settings["train_kwargs"]
    qtable = run_training(
        env,
        qtable,
        n_training_episodes=train_kwargs["n_training_episodes"],
        learning_rate=train_kwargs["learning_rate"],
        gamma=train_kwargs["gamma"],
        max_steps=train_kwargs["max_steps"],
        max_epsilon=train_kwargs["max_epsilon"],
        min_epsilon=train_kwargs["min_epsilon"],
        decay_rate=train_kwargs["decay_rate"],
    )

    model = {
        "env_id": settings["env_id"],
        "env_kwargs": settings["env_kwargs"],
        **train_kwargs,
        "qtable": qtable,
    }

    path = save_path or settings["model_path"]
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=list(config.ENVIRONMENTS),
        default=config.DEFAULT_ENVIRONMENT,
        help="Which environment to train on.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.env)
