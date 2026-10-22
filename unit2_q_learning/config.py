"""Shared settings for the Q-Learning (FrozenLake + Taxi) project.

Centralizing these makes it easy to tweak an environment's hyperparameters
or Hugging Face Hub target without touching the training/evaluation/upload
logic itself. Each environment gets its own settings block since they don't
share a state/action space or a training budget.
"""

# --- FrozenLake-v1 (4x4) ---
# Non-slippery: every action goes exactly where you intend (deterministic).
# Slippery: each action has a chance to slide sideways instead (stochastic),
# so the same greedy policy can take different paths across episodes.
FROZENLAKE_ENV_ID = "FrozenLake-v1"
FROZENLAKE_MAP_NAME = "4x4"

FROZENLAKE_MODEL_NAME = "q-FrozenLake-v1-4x4-noSlippery"
FROZENLAKE_MODEL_PATH = f"{FROZENLAKE_MODEL_NAME}.pkl"
FROZENLAKE_SLIPPERY_MODEL_NAME = "q-FrozenLake-v1-4x4-slippery"
FROZENLAKE_SLIPPERY_MODEL_PATH = f"{FROZENLAKE_SLIPPERY_MODEL_NAME}.pkl"

FROZENLAKE_KWARGS = dict(
    n_training_episodes=10_000,
    learning_rate=0.7,
    min_learning_rate=0.7,  # same as learning_rate => no LR decay
    lr_decay_rate=0.0,
    n_eval_episodes=100,
    max_steps=99,
    gamma=0.95,
    eval_seed=[],
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.0005,
    # Mid-training greedy eval for the debug plots (0 = skip).
    train_eval_every=1_000,
    n_train_eval_episodes=50,
)

# Slippery: decay LR so early episodes learn fast, late ones mostly settle.
# lr = min_lr + (max_lr - min_lr) * exp(-lr_decay_rate * episode)
FROZENLAKE_SLIPPERY_KWARGS = dict(
    n_training_episodes=500_000,
    learning_rate=0.7,  # starting / max LR
    min_learning_rate=0.001,
    lr_decay_rate=0.00001,  # ~near min_lr by a few hundred thousand episodes
    n_eval_episodes=500,
    max_steps=99,
    gamma=0.95,
    eval_seed=[],
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.00005,
    train_eval_every=10_000,
    n_train_eval_episodes=50,
)

# --- Taxi-v3 ---
TAXI_ENV_ID = "Taxi-v4"  # Taxi-v3 is deprecated as of gymnasium>=1.1
TAXI_MODEL_NAME = "q-taxi"
TAXI_MODEL_PATH = f"{TAXI_MODEL_NAME}.pkl"

# DO NOT MODIFY eval_seed: it lets every student's agent be evaluated from
# the same starting states, so results are comparable across submissions.
TAXI_EVAL_SEED = [
    16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148,
    146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55, 161, 131, 184, 51,
    170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9, 113,
    58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55,
    31, 85, 38, 112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148,
]

TAXI_KWARGS = dict(
    n_training_episodes=40_000,
    learning_rate=0.7,
    min_learning_rate=0.7,
    lr_decay_rate=0.0,
    n_eval_episodes=100,
    max_steps=99,
    gamma=0.95,
    eval_seed=TAXI_EVAL_SEED,
    max_epsilon=1.0,
    min_epsilon=0.05,
    decay_rate=0.005,
    train_eval_every=2_000,
    n_train_eval_episodes=50,
)

ENVIRONMENTS = {
    "frozenlake": dict(
        env_id=FROZENLAKE_ENV_ID,
        env_kwargs=dict(map_name=FROZENLAKE_MAP_NAME, is_slippery=False),
        model_name=FROZENLAKE_MODEL_NAME,
        model_path=FROZENLAKE_MODEL_PATH,
        train_kwargs=FROZENLAKE_KWARGS,
    ),
    "frozenlake-slippery": dict(
        env_id=FROZENLAKE_ENV_ID,
        env_kwargs=dict(map_name=FROZENLAKE_MAP_NAME, is_slippery=True),
        model_name=FROZENLAKE_SLIPPERY_MODEL_NAME,
        model_path=FROZENLAKE_SLIPPERY_MODEL_PATH,
        train_kwargs=FROZENLAKE_SLIPPERY_KWARGS,
    ),
    "taxi": dict(
        env_id=TAXI_ENV_ID,
        env_kwargs=dict(),
        model_name=TAXI_MODEL_NAME,
        model_path=TAXI_MODEL_PATH,
        train_kwargs=TAXI_KWARGS,
    ),
}

DEFAULT_ENVIRONMENT = "frozenlake"

# --- Hugging Face Hub ---
# Change this to "<your-username>/<model-name>" before pushing.
USERNAME = "<your-username>"

# --- Training runs ---
# Each `python main.py train` writes a unique folder here with model.pkl + run.txt.
RUNS_FOLDER = "runs"

# --- Video recording ---
VIDEO_FOLDER = "videos"
VIDEO_FPS = 1
VIDEO_EPISODES = 3
