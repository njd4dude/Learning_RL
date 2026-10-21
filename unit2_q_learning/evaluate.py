"""Evaluate a saved Q-Learning model (FrozenLake or Taxi).

Usage:
    python evaluate.py [model_path]
"""

import pickle
import sys

import gymnasium as gym

import config
from q_learning import evaluate_agent


def load_saved_model(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate(model_path: str = config.FROZENLAKE_MODEL_PATH):
    """Load a pickled model dict and report its mean/std reward.

    Note: evaluation uses a fresh environment instance, never the one used
    for training.
    """
    model = load_saved_model(model_path)
    env = gym.make(model["env_id"], **model.get("env_kwargs", {}))

    mean_reward, std_reward = evaluate_agent(
        env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"]
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.FROZENLAKE_MODEL_PATH
    evaluate(path)
