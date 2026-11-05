"""Record replay videos of a trained agent, or watch it live in a window.

Usage:
    python record_video.py                  # newest run
    python record_video.py runs/<run>       # a specific run folder
    python record_video.py --episodes 5
    python record_video.py --watch          # live window, no video saved

Videos land in `videos/<run-folder>/episode-01.mp4`, `episode-02.mp4`, ...
"""

import argparse
from pathlib import Path

import gymnasium as gym
import highway_env
import imageio

import config
from evaluate import latest_run, load_best

gym.register_envs(highway_env)


class FrameCollector:
    """Keeps every simulated frame, not just the one after each decision.

    highway-env calls `_capture_frame` on this for the frames between decisions.
    """

    def __init__(self, env):
        self.env = env
        self.frames = []
        env.unwrapped.set_record_video_wrapper(self)

    def _capture_frame(self) -> None:
        self.frames.append(self.env.render())


def play_episode(env, model, collector: FrameCollector | None = None) -> tuple[float, int, bool]:
    """Play one episode; return its reward, decisions made, and whether it crashed."""
    obs, _info = env.reset()
    if collector:
        collector._capture_frame()

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        if collector:
            collector._capture_frame()
    return total_reward, steps, info["crashed"]


def record(run_dir: Path, episodes: int) -> None:
    model = load_best(run_dir)
    env = gym.make(config.ENV_ID, render_mode="rgb_array", config=config.ENV_CONFIG)
    collector = FrameCollector(env)

    for episode in range(1, episodes + 1):
        collector.frames = []
        reward, steps, crashed = play_episode(env, model, collector)
        path = Path(config.VIDEO_FOLDER) / run_dir.name / f"episode-{episode:02d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, collector.frames, fps=config.VIDEO_FPS, macro_block_size=1)
        print(f"{path}: reward {reward:.2f}, steps {steps}, crashed {crashed}")
    env.close()


def watch(run_dir: Path, episodes: int) -> None:
    model = load_best(run_dir)
    env = gym.make(config.ENV_ID, render_mode="human", config={**config.ENV_CONFIG, "real_time_rendering": True})
    for episode in range(1, episodes + 1):
        reward, steps, crashed = play_episode(env, model)
        print(f"episode {episode}: reward {reward:.2f}, steps {steps}, crashed {crashed}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Run folder. Defaults to the newest.")
    parser.add_argument("--episodes", type=int, default=config.VIDEO_EPISODES)
    parser.add_argument("--watch", action="store_true", help="Show a live window instead of saving videos.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    (watch if args.watch else record)(run_dir, args.episodes)
