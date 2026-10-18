"""Evaluate a trained PPO agent on LunarLander.

Usage:
    python evaluate.py [model_path]
"""

import sys

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import config


def evaluate(model_path: str = config.MODEL_NAME, n_episodes: int = 10):
    """Load a saved model and report its mean/std reward.

    Note: evaluation uses a fresh environment instance, never the one used
    for training.
    """
    model = PPO.load(model_path)
    eval_env = Monitor(gym.make(config.ENV_ID, render_mode="rgb_array"))

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_NAME
    evaluate(path)
