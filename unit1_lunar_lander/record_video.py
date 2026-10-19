"""Record replay videos of a trained agent playing LunarLander.

Usage:
    python record_video.py [model_path]

On a headless Linux box (e.g. a server or Colab) you'll need a virtual
display first, since LunarLander renders with OpenGL:

    sudo apt-get install -y xvfb python3-opengl ffmpeg
    pip install pyvirtualdisplay

    from pyvirtualdisplay import Display
    Display(visible=0, size=(1400, 900)).start()

On Windows/macOS with a normal desktop session this isn't needed.
"""

import sys

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import config
from evaluate import load_saved_model


def record(
    model_path: str = config.MODEL_NAME,
    episodes: int = config.VIDEO_EPISODES,
    video_folder: str = config.VIDEO_FOLDER,
):
    """Play `episodes` episodes with a trained model and save them as videos."""
    model = load_saved_model(model_path)
    env = RecordVideo(
        gym.make(config.ENV_ID, render_mode="rgb_array"),
        video_folder=video_folder,
        episode_trigger=lambda e: True,
    )

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode}: total reward = {total_reward:.1f}")

    env.close()
    print(f"Videos saved to ./{video_folder}/")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_NAME
    record(path)
