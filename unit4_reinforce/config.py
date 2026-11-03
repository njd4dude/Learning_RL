"""Shared settings for the Unit 4 REINFORCE (CartPole + Pixelcopter) project.

Same pattern as unit2: one settings block per environment, so train /
evaluate / record / hub never need environment-specific globals. Values are
the course notebook's defaults.
"""

# --- CartPole-v1 ---
CARTPOLE_ENV_ID = "CartPole-v1"
CARTPOLE_HYPERPARAMETERS = dict(
    h_size=16,
    n_hidden_layers=1,
    n_training_episodes=1000,
    n_evaluation_episodes=10,
    max_t=1000,
    gamma=1.0,
    lr=1e-2,
)

# --- Pixelcopter-PLE-v0 ---
# Much harder than CartPole: deeper network, smaller LR, far more episodes.
PIXELCOPTER_ENV_ID = "Pixelcopter-PLE-v0"
PIXELCOPTER_HYPERPARAMETERS = dict(
    h_size=64,
    n_hidden_layers=2,  # fc1 -> fc2 (h_size*2) -> fc3
    n_training_episodes=50_000,
    n_evaluation_episodes=50,
    max_t=10_000,
    gamma=0.99,
    lr=1e-4,
)

ENVIRONMENTS = {
    "cartpole": dict(
        env_id=CARTPOLE_ENV_ID,
        model_name="Reinforce-cartpole",
        hyperparameters=CARTPOLE_HYPERPARAMETERS,
        print_every=100,
        # Mid-training eval for the TensorBoard eval/ charts (0 = skip).
        train_eval_every=50,
        n_train_eval_episodes=10,
    ),
    "pixelcopter": dict(
        env_id=PIXELCOPTER_ENV_ID,
        model_name="Reinforce-pixelcopter",
        hyperparameters=PIXELCOPTER_HYPERPARAMETERS,
        print_every=1000,
        train_eval_every=500,
        n_train_eval_episodes=10,
    ),
}

DEFAULT_ENVIRONMENT = "cartpole"

# The policy is a tiny MLP fed one state at a time, so moving every step to
# the GPU and back costs more than it saves. "cuda" works if you want it.
DEVICE = "cpu"

# --- Hugging Face Hub ---
USERNAME = "njdonato"

# --- Training runs ---
# Each `python main.py train` writes a unique folder here with model.pt,
# hyperparameters.json, run.txt, and TensorBoard logs under tb/.
RUNS_FOLDER = "runs"
AUTO_START_TENSORBOARD = True
TENSORBOARD_PORT = 6006

# --- Video recording ---
VIDEO_FOLDER = "videos"
VIDEO_FPS = 30
VIDEO_EPISODES = 5
