"""Record replay videos of a trained Q-Learning agent.

Usage:
    python record_video.py [model_path] [out_path]

Each episode is saved as its own mp4. With no out_path, files land in
`videos/<model>-episode-<n>.mp4`. An out_path is used as the base name, so
`replay.mp4` becomes `replay-episode-0.mp4`, `replay-episode-1.mp4`, and so on.
"""

import random
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np

import config
from evaluate import load_saved_model
from q_learning import greedy_policy


def _episode_path(model_path: str, out_path: str | None, episode: int) -> Path:
    """Where episode `episode` is written.

    No out_path: `videos/<model>-episode-<n>.mp4`.
    A directory: `<dir>/<model>-episode-<n>.mp4`.
    A file path: `<stem>-episode-<n><suffix>` next to that file.
    """
    if out_path is None:
        directory = Path(config.VIDEO_FOLDER)
        stem = Path(model_path).stem
        return directory / f"{stem}-episode-{episode}.mp4"

    path = Path(out_path)
    if path.suffix == "":
        stem = Path(model_path).stem
        return path / f"{stem}-episode-{episode}.mp4"
    return path.with_name(f"{path.stem}-episode-{episode}{path.suffix}")


def record(
    model_path: str = config.FROZENLAKE_MODEL_PATH,
    out_path: str | None = None,
    fps: int = config.VIDEO_FPS,
    episodes: int = config.VIDEO_EPISODES,
):
    """Play `episodes` greedy-policy episodes and save each one as an mp4.

    Each episode is reset with its own seed, so the three replays start from
    different states instead of repeating the same trajectory.
    """
    model = load_saved_model(model_path)
    env = gym.make(model["env_id"], render_mode="rgb_array", **model.get("env_kwargs", {}))
    qtable = model["qtable"]
    saved = []

    try:
        for episode in range(episodes):
            images = []
            state, _info = env.reset(seed=random.randint(0, 500))
            images.append(env.render())

            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = greedy_policy(qtable, state)
                state, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                images.append(env.render())

            out = _episode_path(model_path, out_path, episode)
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(out, [np.array(img) for img in images], fps=fps)
            print(f"Episode {episode}: total reward = {total_reward:.1f}")
            print(f"Saved video to {out}")
            saved.append(out)
    finally:
        env.close()

    return saved


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.FROZENLAKE_MODEL_PATH
    out = sys.argv[2] if len(sys.argv) > 2 else None
    record(path, out)
