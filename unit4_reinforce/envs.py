"""Environment construction, including a Gymnasium port of Pixelcopter.

The course notebook gets Pixelcopter from `gym_pygame`, which only supports
the legacy `gym` API. This module wraps the same PyGame Learning Environment
(PLE) game the way `gym_pygame` does, but speaks the Gymnasium API used by
every other unit, and registers it under the same `Pixelcopter-PLE-v0` id.

`Pixelcopter-PLE-pixels-v0` is the same game observed through its screen
instead: 48x48 grayscale frames, the last FRAME_STACK of them stacked so a
single observation shows motion. That one is for the CNN policy.
"""

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config


class PixelcopterEnv(gym.Env):
    """PLE's Pixelcopter: fly through gaps without hitting the walls.

    Observation, obs_type="state": the 7 values of `game.getGameState()`
    (height, velocity, distances to ceiling/floor/next gate, next gate's
    top/bottom). obs_type="pixels": the screen as a (48, 48) uint8 grayscale
    image.
    Actions: 0 = flap up, 1 = do nothing.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, obs_type: str = "state"):
        # Headless, as gym_pygame does with display=False. Must be set before
        # pygame is imported by PLE.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            from ple import PLE
            from ple.games.pixelcopter import Pixelcopter
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Pixelcopter needs the PyGame Learning Environment. Install it with:\n"
                '  pip install "git+https://github.com/ntasfi/PyGame-Learning-Environment.git"'
            ) from e

        if obs_type not in ("state", "pixels"):
            raise ValueError(f'obs_type must be "state" or "pixels", got {obs_type!r}')
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.game = Pixelcopter()
        # PLE defaults to rng=24, which replays the same levels every run.
        # None seeds from the OS; pass reset(seed=...) to get repeatable levels.
        self.ple = PLE(self.game, fps=30, display_screen=False, rng=None)
        self.ple.init()

        self.action_set = self.ple.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))
        if obs_type == "pixels":
            height, width, _ = self.render().shape
            self.observation_space = spaces.Box(0, 255, shape=(height, width), dtype=np.uint8)
        else:
            n_obs = len(self.game.getGameState())
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        if self.obs_type == "pixels":
            # Standard luminance weights; color adds nothing the CNN needs here.
            gray = self.render() @ np.array([0.299, 0.587, 0.114])
            return np.round(gray).astype(np.uint8)
        return np.array(list(self.game.getGameState().values()), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.rng.seed(seed)
        self.ple.reset_game()
        if self.obs_type == "pixels":
            # reset_game() rebuilds the level but only step() draws it, so the
            # screen still shows the last game's crash (or black, the first
            # time). Draw the new level without advancing it, the same calls
            # Pixelcopter.step() ends with.
            self.game.screen.fill((0, 0, 0))
            self.game.player_group.draw(self.game.screen)
            self.game.block_group.draw(self.game.screen)
            self.game.terrain_group.draw(self.game.screen)
        return self._obs(), {}

    def step(self, action):
        reward = self.ple.act(self.action_set[action])
        terminated = self.ple.game_over()
        return self._obs(), float(reward), terminated, False, {}

    def render(self):
        # PLE returns (width, height, 3); video frames are (height, width, 3).
        return np.transpose(self.ple.getScreenRGB(), (1, 0, 2))


gym.register(
    id=config.PIXELCOPTER_ENV_ID,
    entry_point=PixelcopterEnv,
    # Without a cap a good agent never dies, so record_video would never end.
    max_episode_steps=config.PIXELCOPTER_HYPERPARAMETERS["max_t"],
)


gym.register(
    id=config.PIXELCOPTER_PIXELS_ENV_ID,
    entry_point=PixelcopterEnv,
    kwargs=dict(obs_type="pixels"),
    max_episode_steps=config.PIXELCOPTER_PIXELS_HYPERPARAMETERS["max_t"],
)


def make_env(env_id: str, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    if env_id == config.PIXELCOPTER_PIXELS_ENV_ID:
        # One frame can't show which way the copter is moving; a stack can.
        env = gym.wrappers.FrameStackObservation(env, config.FRAME_STACK)
    return env
