"""Build and train a PPO agent on LunarLander.

Usage:
    python train.py                 # start a new agent from scratch
    python train.py --load          # continue from the last saved zip
    python train.py --load PATH     # continue from a specific zip or checkpoint
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import configure_logger

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


def save_final_checkpoint(model: PPO) -> Path:
    """Save the current weights into the checkpoint folder.

    Uses the same name as the periodic callback, so a run that stops between
    the 50k marks still leaves a snapshot at its exact timestep.
    """
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{config.MODEL_NAME}_{model.num_timesteps}_steps"
    model.save(path)
    return path


def start_new_log_run(model: PPO) -> None:
    """Open a fresh TensorBoard folder for this execution.

    Resuming passes ``reset_num_timesteps=False``, which would otherwise
    keep writing into the latest ``PPO_N`` directory. Setting the logger
    first still increments that number. The agent's own step counter is
    unchanged, so checkpoints stay on the loaded timeline.
    """
    logger = configure_logger(
        model.verbose,
        model.tensorboard_log,
        tb_log_name="PPO",
        reset_num_timesteps=True,
    )
    model.set_logger(logger)
    print(f"Logging this run to {logger.dir}")


def resolve_model_path(path: str) -> Path:
    """Find a saved model, accepting a path with or without the `.zip` suffix."""
    candidate = Path(path)
    options = [candidate] if candidate.suffix == ".zip" else [candidate, Path(f"{path}.zip")]
    for option in options:
        if option.is_file():
            return option
    tried = ", ".join(str(option) for option in options)
    raise SystemExit(f"No saved model found ({tried}). Train from scratch, or pass a checkpoint path.")


def train(
    timesteps: int = config.TRAIN_TIMESTEPS,
    save_path: str = config.MODEL_NAME,
    load_path: str | None = None,
) -> PPO:
    """Train a PPO agent, saving periodic checkpoints and a final model.

    Pass `load_path` to keep training an existing zip. Omit it to start a
    new agent. When resuming, `timesteps` is how many additional steps to
    train, and the agent's timestep counter keeps going so checkpoints
    line up with the earlier run. Each execution still gets its own
    TensorBoard folder.

    Safe to interrupt with Ctrl+C: the most recent progress is saved to
    `save_path` and to `config.CHECKPOINT_DIR` either way (either because
    training finished, or via the `finally` block below). Intermediate
    checkpoints along the way also live in `config.CHECKPOINT_DIR`.
    """
    env = build_env()
    if load_path is None:
        print("Starting a new agent from scratch.")
        model = build_model(env)
        reset_num_timesteps = True
    else:
        resolved = resolve_model_path(load_path)
        print(f"Resuming from {resolved}")
        model = PPO.load(resolved, env=env)
        reset_num_timesteps = False
        print(f"Already trained for {model.num_timesteps:,} timesteps; training {timesteps:,} more.")

    checkpoint_callback = build_checkpoint_callback()
    start_new_log_run(model)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current progress...")
    finally:
        model.save(save_path)
        checkpoint_path = save_final_checkpoint(model)
        print(f"Saved model to {save_path}.zip")
        print(f"Saved checkpoint to {checkpoint_path}.zip")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load",
        nargs="?",
        const=config.MODEL_NAME,
        default=None,
        metavar="PATH",
        help="Resume from a saved model. With no PATH, uses the last saved zip.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_path=args.load)
