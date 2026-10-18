"""Push a trained model to the Hugging Face Hub, or pull one down.

Usage:
    python hub.py push               # upload the model in config.py to the Hub
    python hub.py pull <repo_id> <filename>   # download + evaluate a model
"""

import sys

import gymnasium as gym
from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config


def push_to_hub(
    model_path: str = config.MODEL_NAME,
    repo_id: str = config.REPO_ID,
    env_id: str = config.ENV_ID,
    model_architecture: str = config.MODEL_ARCHITECTURE,
    commit_message: str = config.COMMIT_MESSAGE,
):
    """Evaluate, record a replay, generate a model card, and upload to the Hub.

    Requires a Hugging Face account + write token. Log in once beforehand with
    `huggingface-cli login`, or call `huggingface_hub.notebook_login()` in a
    notebook.
    """
    model = PPO.load(model_path)
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

    package_to_hub(
        model=model,
        model_name=config.MODEL_NAME,
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message=commit_message,
    )


def pull_from_hub(repo_id: str, filename: str, n_episodes: int = 10):
    """Download a model from the Hub and report its evaluation score.

    Some older models on the Hub were trained with the original `gym` library
    rather than `gymnasium`; if loading fails, `pip install shimmy` to bridge
    the two APIs.
    """
    # Neutralizes a learning-rate/clip-range schedule mismatch that can occur
    # when the pickle was produced by a different library version.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    checkpoint = load_from_hub(repo_id, filename)
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)

    eval_env = Monitor(gym.make(config.ENV_ID))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return model


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("push", "pull"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "push":
        push_to_hub()
    else:
        if len(sys.argv) != 4:
            print("Usage: python hub.py pull <repo_id> <filename>")
            sys.exit(1)
        pull_from_hub(sys.argv[2], sys.argv[3])
