"""Record replay videos of a trained agent, or watch it live in a window.

Usage:
    python record_video.py                  # newest run
    python record_video.py runs/<run>       # a specific run folder
    python record_video.py --episodes 5
    python record_video.py --watch          # live window, no video saved

Videos land in `<run folder>/videos/episode-01.mp4`, `episode-02.mp4`, ... with a
caption of the latest decision: its action and reward, and the running total.
Below the road is a panel of what the model saw for the decision being carried
out. For numbers, that's the exact values it was given, what they mean, and a map
drawn only from them. For images, it's the frames themselves.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import highway_env
import imageio
import numpy as np
from highway_env.road.lane import AbstractLane
from PIL import Image, ImageDraw, ImageFont

import config
from evaluate import latest_run, load_best, trained_reward

gym.register_envs(highway_env)

CAPTION_FONT = ImageFont.load_default(size=14)
PANEL_FONT = ImageFont.load_default(size=12)
PANEL_BACKGROUND = (20, 20, 24)
PANEL_TEXT = (230, 230, 230)
PANEL_FADED = (120, 120, 130)
YOUR_CAR = (80, 220, 120)
OTHER_CAR = (90, 170, 255)
TITLE_HEIGHT = 22
ROW_HEIGHT = 16
MAP_HEIGHT = 64
MAP_METRES = (-20, 200)  # the most it can see: a little behind, up to 200 m ahead


class FrameCollector:
    """Keeps every simulated frame, not just the one after each decision.

    highway-env calls `_capture_frame` on this for the frames between decisions.
    """

    def __init__(self, env):
        self.env = env
        self.frames = []
        self.caption = ""
        self.panel = None
        env.unwrapped.set_record_video_wrapper(self)

    def _capture_frame(self) -> None:
        frame = Image.fromarray(self.env.render())
        draw = ImageDraw.Draw(frame)
        draw.text((8, 6), self.caption, font=CAPTION_FONT, fill="white", stroke_width=2, stroke_fill="black")
        self.frames.append(np.vstack([np.asarray(frame), self.panel]))


def _car_meaning(car: dict) -> str:
    """Put one decoded row of the observation into words."""
    side = round(car["y"] / AbstractLane.DEFAULT_WIDTH)
    lane = "same lane"
    if side:
        lane = f"{abs(side)} lane{'s' if abs(side) > 1 else ''} {'right' if side > 0 else 'left'}"
    where = "ahead" if car["x"] >= 0 else "behind"
    return f"{abs(car['x']):.0f} m {where}, {lane}, {car['vx']:+.0f} m/s vs you"


def _draw_map(draw: ImageDraw.ImageDraw, top: int, width: int, cars: list[dict], lanes: int) -> None:
    """Draw the road using only the observation, so unseen cars are simply missing."""
    lane_height = (MAP_HEIGHT - ROW_HEIGHT) / lanes
    low, high = MAP_METRES

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        along = 24 + (x - low) / (high - low) * (width - 48)
        return along, top + (y / AbstractLane.DEFAULT_WIDTH + 0.5) * lane_height

    road_bottom = top + lanes * lane_height
    for lane in range(lanes + 1):
        y = top + lane * lane_height
        draw.line((24, y, width - 24, y), fill=PANEL_FADED)
    for metres in range(0, high + 1, 50):
        x, _ = to_pixel(metres, 0)
        draw.line((x, top, x, road_bottom), fill=(50, 50, 58))
        draw.text((x, road_bottom + 1), f"{metres} m", font=PANEL_FONT, fill=PANEL_FADED, anchor="ma")

    you = cars[0]
    for number, car in enumerate(cars):
        if not car["presence"]:
            continue
        x, y = to_pixel(0, you["y"]) if number == 0 else to_pixel(car["x"], you["y"] + car["y"])
        draw.rectangle((x - 7, y - 4, x + 7, y + 4), fill=YOUR_CAR if number == 0 else OTHER_CAR)
        if number:
            draw.text((x, y), str(number), font=PANEL_FONT, fill="black", anchor="mm")


def _table_panel(env, obs: np.ndarray, width: int, title: str) -> Image.Image:
    """Show a table of numbers: one row per car, with one column per feature."""
    observation_type = env.unwrapped.observation_type
    features = observation_type.features
    # Only these settings can be turned back into metres and speeds.
    decodable = (
        observation_type.normalize
        and not observation_type.absolute
        and {"presence", "x", "y", "vx"} <= set(features)
    )
    ranges = observation_type.features_range if decodable else {}
    cars = [
        {
            name: np.interp(value, (-1, 1), ranges[name]) if name in ranges else value
            for name, value in zip(features, row)
        }
        for row in obs
    ]
    lanes = env.unwrapped.config["lanes_count"]

    table_bottom = TITLE_HEIGHT + ROW_HEIGHT * (len(obs) + 1)
    height = table_bottom + (MAP_HEIGHT + 6 if decodable else 0)
    panel = Image.new("RGB", (width, height + height % 2), PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), title, font=PANEL_FONT, fill=PANEL_TEXT)

    columns = [52 + index * 54 for index in range(len(features))]
    meaning_column = columns[-1] + 56
    draw.text((8, TITLE_HEIGHT), "row", font=PANEL_FONT, fill=PANEL_FADED)
    for x, name in zip(columns, features):
        draw.text((x, TITLE_HEIGHT), name, font=PANEL_FONT, fill=PANEL_FADED)
    if decodable:
        draw.text((meaning_column, TITLE_HEIGHT), "what that means", font=PANEL_FONT, fill=PANEL_FADED)

    for number, (row, car) in enumerate(zip(obs, cars)):
        y = TITLE_HEIGHT + ROW_HEIGHT * (number + 1)
        seen = row[0] != 0
        colour = YOUR_CAR if number == 0 else PANEL_TEXT if seen else PANEL_FADED
        draw.text((8, y), "you" if number == 0 else f"car {number}", font=PANEL_FONT, fill=colour)
        for x, value in zip(columns, row):
            draw.text((x, y), f"{value:+.2f}", font=PANEL_FONT, fill=colour)
        if decodable:
            if number == 0:
                lane = round(car["y"] / AbstractLane.DEFAULT_WIDTH) + 1
                meaning = f"lane {lane} of {lanes}, {car['vx']:.0f} m/s"
            else:
                meaning = _car_meaning(car) if seen else "empty slot"
            draw.text((meaning_column, y), meaning, font=PANEL_FONT, fill=colour)

    if decodable:
        _draw_map(draw, table_bottom + 4, width, cars, lanes)
    return panel


def _image_panel(obs: np.ndarray, width: int, title: str) -> Image.Image:
    """Show image layers side by side. highway-env stores them as (stack, width, height)."""
    tile_width = width // len(obs)
    tiles = []
    for layer in obs:
        low, high = layer.min(), layer.max()
        pixels = layer if obs.dtype == np.uint8 else (layer - low) / max(high - low, 1e-6) * 255
        tile = Image.fromarray(pixels.T.astype(np.uint8))
        tile_height = max(tile.height * tile_width // tile.width, 1)
        tiles.append(tile.resize((tile_width, tile_height), Image.Resampling.NEAREST))

    height = TITLE_HEIGHT + tiles[0].height
    panel = Image.new("RGB", (width, height + height % 2), PANEL_BACKGROUND)
    ImageDraw.Draw(panel).text((8, 4), title, font=PANEL_FONT, fill=PANEL_TEXT)
    for index, tile in enumerate(tiles):
        panel.paste(tile.convert("RGB"), (index * tile_width, TITLE_HEIGHT))
    return panel


def _observation_panel(env, obs: np.ndarray, action: int, step: int) -> np.ndarray:
    """Draw what the model was given for one decision, and what it chose."""
    name = env.unwrapped.action_type.actions[action]
    title = f"What the model saw at decision {step}, and chose {name}"
    width = env.unwrapped.config["screen_width"]
    if hasattr(env.unwrapped.observation_type, "features") and obs.ndim == 2:
        panel = _table_panel(env, obs, width, title)
    else:
        panel = _image_panel(obs, width, title)
    return np.asarray(panel)


def _caption(env, action: int, step: int, reward: float, total: float, info: dict) -> str:
    name = env.unwrapped.action_type.actions[action]
    crashed = "  CRASHED" if info["crashed"] else ""
    return (
        f"step {step}/{config.MAX_EPISODE_STEPS}  {info['speed']:.0f} m/s  {name}  "
        f"reward {reward:+.2f}  total {total:.2f}{crashed}"
    )


def play_episode(env, model, collector: FrameCollector | None = None) -> tuple[float, int, bool]:
    """Play one episode; return its reward, decisions made, and whether it crashed."""
    obs, _info = env.reset()

    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        if collector:
            collector.panel = _observation_panel(env, obs, int(action), steps + 1)
            if steps == 0:
                collector.caption = "Start"
                collector._capture_frame()
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        if collector:
            collector.caption = _caption(env, int(action), steps, reward, total_reward, info)
            collector._capture_frame()
    return total_reward, steps, info["crashed"]


def record(run_dir: Path, episodes: int) -> None:
    model = load_best(run_dir)
    env = gym.make(config.ENV_ID, render_mode="rgb_array", config=config.ENV_CONFIG)
    collector = FrameCollector(env)

    for episode in range(1, episodes + 1):
        collector.frames = []
        reward, steps, crashed = play_episode(env, model, collector)
        path = run_dir / "videos" / f"episode-{episode:02d}.mp4"
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
    config.ENV_CONFIG = trained_reward(run_dir)
    (watch if args.watch else record)(run_dir, args.episodes)
