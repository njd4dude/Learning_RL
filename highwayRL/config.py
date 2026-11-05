"""Shared settings for the highway-v0 project."""

ENV_ID = "highway-v0"

# Reward changes from highway-env's defaults. By default, normalizing pays about
# 0.7 a step just for staying alive, so always braking scored as well as any
# trained model. Without it, surviving earns little and only speed pays, so
# crashes cost more to stop the agent getting reckless. Rewards from runs under
# different settings don't compare, so evaluate.py scores everything with these.
ENV_CONFIG = dict(
    normalize_reward=False,
    collision_reward=-5,
)

# --- DQN (train.py) ---
# highway-env's documented DQN settings. A slower variant (learning_rate 1e-4,
# buffer_size 50,000, target_update_interval 1,000) crashed far more: 51%
# against 20% at 40k steps. gamma is low because only the next
# few seconds of traffic matter. gradient_steps=-1 keeps one update per
# transition with parallel envs, as the single-env recipe does.
TIMESTEPS = 40_000
N_ENVS = 12  # the simulator is slow, at about 6 steps a second per env
DQN_KWARGS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15_000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=-1,
    target_update_interval=50,
)

# --- PPO (train.py --algo ppo) ---
# highway-env's documented PPO settings, except for n_steps: that's per env,
# so with 12 envs, 64 keeps the recipe's 768 steps between updates.
PPO_TIMESTEPS = 200_000
PPO_KWARGS = dict(
    policy="MlpPolicy",
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    n_steps=64,
    batch_size=64,
    n_epochs=10,
    learning_rate=5e-4,
    gamma=0.8,
)

# --- Evaluation ---
EVAL_EPISODES = 48  # final score, also used by evaluate.py
TRAIN_EVAL_EPISODES = 24
TRAIN_EVAL_STEPS = 2_500
MAX_EPISODE_STEPS = 40  # highway-v0's "duration"; shorter episodes crashed

# --- Saving ---
# Each run writes runs/<timestamp>_<algo>_<name>/ with best_model.zip (top
# mid-training eval), model.zip (the last step), run.txt, TensorBoard logs under
# tb/, and later eval.txt and videos/. runs/ is gitignored, so EXPERIMENTS.md
# keeps the tracked record of what each run tested.
RUNS_FOLDER = "runs"
EXPERIMENTS_FILE = "EXPERIMENTS.md"
CHECKPOINT_STEPS = 5_000

# What differs per algorithm. PPO takes far more steps, so it evals less often.
ALGORITHMS = {
    "dqn": dict(
        kwargs=DQN_KWARGS,
        timesteps=TIMESTEPS,
        eval_steps=TRAIN_EVAL_STEPS,
        checkpoint_steps=CHECKPOINT_STEPS,
        log_interval=100,  # episodes
    ),
    "ppo": dict(
        kwargs=PPO_KWARGS,
        timesteps=PPO_TIMESTEPS,
        eval_steps=10_000,
        checkpoint_steps=25_000,
        log_interval=10,  # updates
    ),
}

# --- Video recording ---
VIDEO_FPS = 15  # the simulator's 15 frames a second, so videos play in real time
VIDEO_EPISODES = 3
