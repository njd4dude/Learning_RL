"""Train a DQN agent on LunarLander.

Usage:
    python train_dqn.py                 # start a new agent from scratch
    python train_dqn.py --load          # continue from the last saved zip
    python train_dqn.py --load PATH     # continue from a specific zip or checkpoint
"""

import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

import config
from train import resolve_model_path, start_new_log_run


def replay_buffer_path(model_path: Path) -> Path:
    """Replay-buffer file that belongs next to a saved DQN zip.

    Checkpoints use Stable-Baselines3's name, ``<prefix>_replay_buffer_<steps>_steps.pkl``.
    The final model uses ``<name>_replay_buffer.pkl``.
    """
    name = model_path.name.removesuffix(".zip")
    if name.endswith("_steps"):
        prefix, steps = name[: -len("_steps")].rsplit("_", 1)
        return model_path.with_name(f"{prefix}_replay_buffer_{steps}_steps.pkl")
    return model_path.with_name(f"{name}_replay_buffer.pkl")


def save_replay_buffer(model: DQN, model_path: str) -> Path:
    path = replay_buffer_path(Path(model_path))
    model.save_replay_buffer(path)
    return path


def train(
    timesteps: int = config.DQN_TIMESTEPS,
    save_path: str = config.DQN_MODEL_NAME,
    load_path: str | None = None,
) -> DQN:
    """Train a DQN agent and save it apart from the PPO zip.

    Pass `load_path` to keep training an existing zip. The replay buffer is
    saved beside the zip and loaded again on resume, so the old landings
    come along with the weights. Omit `load_path` to start a new agent.
    """
    env = make_vec_env(config.ENV_ID, n_envs=1)
    if load_path is None:
        print("Starting a new DQN agent from scratch.")
        model = DQN(env=env, **config.DQN_KWARGS)
        reset_num_timesteps = True
    else:
        resolved = resolve_model_path(load_path)
        print(f"Resuming from {resolved}")
        model = DQN.load(resolved, env=env)
        reset_num_timesteps = False
        buffer_path = replay_buffer_path(resolved)
        if buffer_path.is_file():
            model.load_replay_buffer(buffer_path)
            print(
                f"Loaded {model.replay_buffer.size():,} stored steps from {buffer_path.name}. "
                f"Already trained for {model.num_timesteps:,} timesteps; training {timesteps:,} more."
            )
        else:
            print(
                f"No replay buffer at {buffer_path.name}. "
                f"Continuing from the weights with an empty buffer "
                f"after {model.num_timesteps:,} timesteps; training {timesteps:,} more."
            )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.CHECKPOINT_DIR,
        name_prefix=config.DQN_MODEL_NAME,
        save_replay_buffer=True,
    )
    start_new_log_run(model, tb_log_name="DQN")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name="DQN",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current progress...")
    finally:
        model.save(save_path)
        buffer_path = save_replay_buffer(model, save_path)
        print(f"Saved model to {save_path}.zip")
        print(f"Saved replay buffer to {buffer_path}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load",
        nargs="?",
        const=config.DQN_MODEL_NAME,
        default=None,
        metavar="PATH",
        help="Resume from a saved DQN model. With no PATH, uses the last saved zip.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_path=args.load)
