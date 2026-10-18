"""Shared settings for the LunarLander PPO project.

Centralizing these makes it easy to tweak the environment, the training
hyperparameters, or the Hugging Face Hub target without touching the
training/evaluation/upload logic itself.
"""

# --- Environment ---
ENV_ID = "LunarLander-v3"
N_ENVS = 16  # number of parallel environments used during training

# --- Model ---
MODEL_NAME = "ppo-LunarLander-v2"
MODEL_ARCHITECTURE = "PPO"
MODEL_PATH = f"{MODEL_NAME}.zip"

# PPO hyperparameters (tuned for LunarLander in the Deep RL Course)
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs",
)

TRAIN_TIMESTEPS = 500_000

# Save a checkpoint every N timesteps during training, so progress survives
# a Ctrl+C or crash even if the final save never happens.
CHECKPOINT_FREQ = 50_000
CHECKPOINT_DIR = "./checkpoints"

# --- Hugging Face Hub ---
# Change this to "<your-username>/ppo-LunarLander-v2" before pushing.
REPO_ID = "<your-username>/ppo-LunarLander-v2"
COMMIT_MESSAGE = "Upload PPO LunarLander agent"

# --- Video recording ---
VIDEO_FOLDER = "videos"
VIDEO_EPISODES = 3
