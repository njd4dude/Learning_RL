"""Evaluate a saved REINFORCE model (CartPole or Pixelcopter).

Usage:
    python evaluate.py [model_path]    # defaults to the newest runs/*/model.pt
"""

import sys
from pathlib import Path

import torch

import config
from envs import make_env
from policy import build_policy
from reinforce import evaluate_agent


def latest_model_path(env_name: str | None = None) -> Path:
    """The model.pt from the most recent training run, optionally of one environment."""
    pattern = f"*_{env_name}/model.pt" if env_name else "*/model.pt"
    candidates = sorted(Path(config.RUNS_FOLDER).glob(pattern))
    if not candidates:
        which = f"{env_name} runs" if env_name else "runs"
        raise FileNotFoundError(f"No {which} found under {config.RUNS_FOLDER}/ — train one first.")
    return candidates[-1]


def load_saved_model(model_path: str | Path | None = None):
    """Rebuild the policy saved by train.py. Returns (policy, hyperparameters)."""
    model_path = model_path or latest_model_path()
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    hyperparameters = checkpoint["hyperparameters"]
    policy = build_policy(hyperparameters, config.DEVICE)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy, hyperparameters


def n_evaluation_episodes(env_id: str) -> int:
    """How many episodes to score a model on, read from config.py at eval time.

    It never affects training, so editing config.py applies to already-trained
    models too instead of using the value saved in model.pt.
    """
    for settings in config.ENVIRONMENTS.values():
        if settings["env_id"] == env_id:
            return settings["hyperparameters"]["n_evaluation_episodes"]
    raise KeyError(f"No config.ENVIRONMENTS entry for {env_id}")


def evaluate(model_path: str | Path | None = None):
    """Load a saved model and report its mean/std reward on a fresh env."""
    model_path = model_path or latest_model_path()
    print(f"Evaluating {model_path}")
    policy, hyperparameters = load_saved_model(model_path)
    env = make_env(hyperparameters["env_id"])
    mean_reward, std_reward = evaluate_agent(
        env, hyperparameters["max_t"], n_evaluation_episodes(hyperparameters["env_id"]), policy
    )
    env.close()
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else None)
