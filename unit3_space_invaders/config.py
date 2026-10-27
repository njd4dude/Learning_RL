"""Shared settings for the Unit 3 Space Invaders (RL Zoo + DQN) project.

Mirrors unit2's pattern: one place for env ids, default paths, and Hub targets
so train / evaluate / record / hub stay thin wrappers around RL Baselines3 Zoo.
"""

from __future__ import annotations

# --- Defaults (course unit 3) ---
DEFAULT_ALGO = "dqn"
DEFAULT_ENV = "SpaceInvadersNoFrameskip-v4"
DEFAULT_CONFIG = "dqn.yml"
LOG_FOLDER = "logs"
# TensorBoard server watches this tree. Event files live in each Zoo run's
# ``<logs>/<algo>/<env>_N/tb/`` (not a separate tb/ counter).
TENSORBOARD_LOG_FOLDER = LOG_FOLDER
AUTO_START_TENSORBOARD = True
TENSORBOARD_PORT = 6006
VIDEO_FOLDER = "videos"
VIDEO_FPS = 30
VIDEO_EPISODES = 2
ENJOY_TIMESTEPS = 5_000

# Optional second env from the notebook's "load a powerful model" section.
BEAMRIDER_ENV = "BeamRiderNoFrameskip-v4"
BEAMRIDER_ORGA = "sb3"
TRAINED_FOLDER = "rl_trained"

ENVIRONMENTS = {
    "space-invaders": DEFAULT_ENV,
    "beam-rider": BEAMRIDER_ENV,
}

DEFAULT_ENVIRONMENT = "space-invaders"

# RL Zoo / SB3 device. "auto" picks CUDA when a GPU build of Torch is installed.
DEVICE = "auto"
# "subproc" runs each env in its own process (best with n_envs > 1 in dqn.yml).
VEC_ENV = "subproc"
# On -i resume the zip has weights but usually no replay buffer. Collect this
# many steps before gradient updates so batches aren't tiny/correlated.
RESUME_WARMUP_STEPS = 50_000
