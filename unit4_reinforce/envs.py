"""Environment construction, including a Gymnasium port of Pixelcopter.

The course notebook gets Pixelcopter from `gym_pygame`, which only supports
the legacy `gym` API. This module wraps the same PyGame Learning Environment
(PLE) game the way `gym_pygame` does, but speaks the Gymnasium API used by
every other unit, and registers it under the same `Pixelcopter-PLE-v0` id.
"""

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config


class PixelcopterEnv(gym.Env):
    """PLE's Pixelcopter: fly through gaps without hitting the walls.

    Observation: the 7 values of `game.getGameState()` (height, velocity,
    distances to ceiling/floor/next gate, next gate's top/bottom).
    Actions: 0 = flap up, 1 = do nothing.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
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

        self.render_mode = render_mode
        self.game = Pixelcopter()
        # PLE defaults to rng=24, which replays the same levels every run.
        # None seeds from the OS; pass reset(seed=...) to get repeatable levels.
        self.ple = PLE(self.game, fps=30, display_screen=False, rng=None)
        self.ple.init()

        self.action_set = self.ple.getActionSet()
        self.action_space = spaces.Discrete(len(self.action_set))
        n_obs = len(self.game.getGameState())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        return np.array(list(self.game.getGameState().values()), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.rng.seed(seed)
        self.ple.reset_game()
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


def make_env(env_id: str, render_mode: str | None = None):
    return gym.make(env_id, render_mode=render_mode)
