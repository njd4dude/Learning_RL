"""Play highway-v0 with random actions, to check the install and see the task.

Usage:
    python try_env.py                # 3 episodes, printing what the agent gets
    python try_env.py --episodes 1
    python try_env.py --watch        # open a window instead of saving a video

Without --watch, the first episode is saved to `videos/random.mp4`.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import highway_env
import imageio

gym.register_envs(highway_env)

ENV_ID = "highway-v0"
VIDEO_PATH = Path("videos/random.mp4")
VIDEO_FPS = 2  # one frame per decision, and the agent decides once a second


def run(episodes: int, watch: bool) -> None:
    env = gym.make(ENV_ID, render_mode="human" if watch else "rgb_array")
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(f"actions: {env.unwrapped.action_type.actions}")

    frames = []
    for episode in range(1, episodes + 1):
        env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            steps += 1
            if not watch and episode == 1:
                frames.append(env.render())
        print(f"episode {episode}: reward {total_reward:.2f}, steps {steps}, crashed {info['crashed']}")
    env.close()

    if frames:
        VIDEO_PATH.parent.mkdir(exist_ok=True)
        imageio.mimsave(VIDEO_PATH, frames, fps=VIDEO_FPS)
        print(f"saved {VIDEO_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    run(args.episodes, args.watch)
