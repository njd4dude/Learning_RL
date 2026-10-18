"""Build and train a PPO agent on LunarLander.

Usage:
    python train.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import config


def build_env(n_envs: int = config.N_ENVS):
    """Create a vectorized LunarLander environment.

    Stacking multiple independent environments together gives the agent
    more diverse experience per training step.
    """
    return make_vec_env(config.ENV_ID, n_envs=n_envs)


def build_model(env) -> PPO:
    """Instantiate a PPO agent with the course's tuned hyperparameters."""
    return PPO(env=env, **config.PPO_KWARGS)


def build_checkpoint_callback(n_envs: int = config.N_ENVS) -> CheckpointCallback:
    """Periodically save the model during training.

    With `n_envs` parallel environments, each call to `env.step()` advances
    the timestep counter by `n_envs`, so `save_freq` is divided down to land
    on roughly the requested real-timestep interval.
    """
    return CheckpointCallback(
        save_freq=max(config.CHECKPOINT_FREQ // n_envs, 1),
        save_path=config.CHECKPOINT_DIR,
        name_prefix=config.MODEL_NAME,
    )


def train(timesteps: int = config.TRAIN_TIMESTEPS, save_path: str = config.MODEL_NAME) -> PPO:
    """Train a PPO agent, saving periodic checkpoints and a final model.

    Safe to interrupt with Ctrl+C: the most recent progress is saved to
    `save_path` either way (either because training finished, or via the
    `finally` block below), and intermediate checkpoints along the way live
    in `config.CHECKPOINT_DIR`.
    """
    env = build_env()
    model = build_model(env)
    checkpoint_callback = build_checkpoint_callback()

    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current progress...")
    finally:
        model.save(save_path)
        print(f"Saved model to {save_path}.zip")

    return model


if __name__ == "__main__":
    train()
