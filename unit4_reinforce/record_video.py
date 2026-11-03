"""Record a replay video of a trained REINFORCE agent.

Usage:
    python record_video.py [model_path] [out_path]

With no model_path, uses the newest runs/*/model.pt. With no out_path, the
video lands in `videos/<run-folder>.mp4`.
"""

import sys
from pathlib import Path

import imageio
import numpy as np
import torch

import config
from envs import make_env
from evaluate import latest_model_path, load_saved_model


def record_episode(env, policy, out_path: str | Path, fps: int = config.VIDEO_FPS) -> float:
    """Play one episode with `policy` on an rgb_array `env` and save it as an mp4."""
    images = []
    state, _info = env.reset()
    images.append(env.render())

    total_reward = 0.0
    terminated = truncated = False
    with torch.no_grad():
        while not (terminated or truncated):
            action, _ = policy.act(state)
            state, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            images.append(env.render())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, [np.array(img) for img in images], fps=fps)
    return total_reward


def record(model_path: str | Path | None = None, out_path: str | Path | None = None):
    model_path = Path(model_path or latest_model_path())
    policy, hyperparameters = load_saved_model(model_path)
    if out_path is None:
        out_path = Path(config.VIDEO_FOLDER) / f"{model_path.parent.name}.mp4"

    env = make_env(hyperparameters["env_id"], render_mode="rgb_array")
    try:
        total_reward = record_episode(env, policy, out_path)
    finally:
        env.close()

    print(f"Total reward = {total_reward:.1f}")
    print(f"Saved video to {out_path}")
    return out_path


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    record(path, out)
