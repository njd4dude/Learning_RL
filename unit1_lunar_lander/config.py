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

TRAIN_TIMESTEPS = 1_0000_000

# DQN hyperparameters tuned for LunarLander-v3 in RL Baselines3 Zoo.
# DQN learns from one environment and a replay buffer, so this block is
# separate from the PPO settings above. 100k steps is the zoo budget.
DQN_MODEL_NAME = "dqn-LunarLander-v3"
DQN_TIMESTEPS = 1_000_000
DQN_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50_000,
    learning_starts=0,
    gamma=0.99,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./logs",
)

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
