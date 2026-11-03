"""Record replay videos of a trained REINFORCE agent.

Usage:
    python record_video.py [model_path] [out_path] [--episodes N]

With no model_path, uses the newest runs/*/model.pt. With no out_path, videos
land in `videos/<run-folder>.mp4` (one episode) or
`videos/<run-folder>-episode-01.mp4`, `-02.mp4`, ... (several).
"""

import argparse
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


def _episode_path(base: Path, episode: int, episodes: int) -> Path:
    if episodes == 1:
        return base
    return base.with_name(f"{base.stem}-episode-{episode:02d}{base.suffix}")


def record(
    model_path: str | Path | None = None,
    out_path: str | Path | None = None,
    episodes: int | None = None,
):
    """Play `episodes` stochastic rollouts and save each as an mp4.

    Episodes run one after another in the same env (PLE/pygame is not safe to
    spawn in parallel). Each file is a different sample from the policy.
    """
    model_path = Path(model_path or latest_model_path())
    policy, hyperparameters = load_saved_model(model_path)
    episodes = config.VIDEO_EPISODES if episodes is None else episodes
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if out_path is None:
        out_path = Path(config.VIDEO_FOLDER) / f"{model_path.parent.name}.mp4"
    else:
        out_path = Path(out_path)

    env = make_env(hyperparameters["env_id"], render_mode="rgb_array")
    saved = []
    try:
        for episode in range(1, episodes + 1):
            path = _episode_path(out_path, episode, episodes)
            total_reward = record_episode(env, policy, path)
            saved.append((path, total_reward))
            print(f"Episode {episode}/{episodes}: total reward = {total_reward:.1f}  -> {path}")
    finally:
        env.close()

    if episodes > 1:
        mean_reward = float(np.mean([reward for _path, reward in saved]))
        print(f"mean_reward={mean_reward:.2f} over {episodes} videos")
    return [path for path, _reward in saved]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("out_path", nargs="?", default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.VIDEO_EPISODES,
        help="How many episodes to record (default: config.VIDEO_EPISODES).",
    )
    args = parser.parse_args()
    record(args.model_path, args.out_path, episodes=args.episodes)
