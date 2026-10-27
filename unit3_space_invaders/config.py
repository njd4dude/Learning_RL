"""Shared settings for the Unit 3 Space Invaders (RL Zoo + DQN) project.

Mirrors unit2's pattern: one place for env ids, default paths, and Hub targets
so train / evaluate / record / hub stay thin wrappers around RL Baselines3 Zoo.
"""

from __future__ import annotations

# --- Defaults (course unit 3) ---
DEFAULT_ALGO = "dqn"
DEFAULT_ENV = "SpaceInvadersNoFrameskip-v4"
DEFAULT_CONFIG = "dqn.yml"
# Existing Zoo runs (evaluate / record / push default here).
LOG_FOLDER = "logs"
# New ``train`` runs go here so TensorBoard stays clean; pass ``-f logs`` for
# the archive. Event files live in each run's ``tb/`` subfolder.
TRAIN_LOG_FOLDER = "logs_active"
# Empty string = TensorBoard watches the train ``-f`` folder. Set explicitly
# only if you want TB to watch a different tree than where zips are saved.
TENSORBOARD_LOG_FOLDER = ""
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
# Save ``replay_buffer.pkl`` next to each run's final zip so ``-i`` resumes with
# the same transitions instead of an empty buffer (~5.6 GB for 100k Atari
# frames; halve it with optimize_memory_usage in dqn.yml).
SAVE_REPLAY_BUFFER = True
# Fallback for checkpoints without a saved buffer: collect this many steps
# before gradient updates so batches aren't tiny/correlated.
RESUME_WARMUP_STEPS = 50_000
