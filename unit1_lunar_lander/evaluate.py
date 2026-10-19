"""Evaluate a trained PPO or DQN agent on LunarLander.

Usage:
    python evaluate.py [model_path]
"""

import sys

import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file

import config


def load_saved_model(model_path: str):
    """Load a PPO or DQN zip.

    The file does not store the algorithm name. DQN saves its exploration
    settings, and PPO does not, so that field picks the class.
    """
    data, _, _ = load_from_zip_file(model_path)
    if "exploration_initial_eps" in data:
        return DQN.load(model_path)
    return PPO.load(model_path)


def evaluate(model_path: str = config.MODEL_NAME, n_episodes: int = 20):
    """Load a saved model and report its mean/std reward.

    Note: evaluation uses a fresh environment instance, never the one used
    for training.
    """
    model = load_saved_model(model_path)
    eval_env = Monitor(gym.make(config.ENV_ID, render_mode="rgb_array"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_NAME
    evaluate(path)
