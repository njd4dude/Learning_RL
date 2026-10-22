"""Train a Q-Learning agent on FrozenLake or Taxi and save its Q-table.

Usage:
    python train.py                              # non-slippery FrozenLake (default)
    python train.py --env frozenlake-slippery    # slippery FrozenLake
    python train.py --env taxi                   # Taxi-v4

Edit hyperparameters in config.py, then re-run. Each run writes a unique
folder under `runs/` with the model pickle, run.txt, final PNG curves, and
live TensorBoard logs under `tb/`.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import config
from plot_training import save_training_plots
from q_learning import evaluate_agent, initialize_q_table, train as run_training


def build_env(env_id: str, env_kwargs: dict):
    return gym.make(env_id, render_mode="rgb_array", **env_kwargs)


def _new_run_dir(env_name: str) -> Path:
    """Unique folder for this training session, e.g. runs/20261021_153012_frozenlake-slippery."""
    runs = Path(config.RUNS_FOLDER)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs / f"{stamp}_{env_name}"
    # Sweeps can finish two trials in the same second; add microseconds if needed.
    if run_dir.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = runs / f"{stamp}_{env_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_log(
    run_dir: Path,
    env_name: str,
    settings: dict,
    train_kwargs: dict,
    model_path: Path,
    mean_reward: float,
    std_reward: float,
):
    """Human-readable record of what was trained and where the model lives."""
    lines = [
        f"env: {env_name}",
        f"env_id: {settings['env_id']}",
        f"env_kwargs: {settings['env_kwargs']}",
        f"model_path: {model_path}",
        f"trained_at: {datetime.now().isoformat(timespec='seconds')}",
        f"mean_reward: {mean_reward:.4f}",
        f"std_reward: {std_reward:.4f}",
        f"tensorboard: tensorboard --logdir {run_dir / 'tb'}",
        "",
        "hyperparameters:",
    ]
    for key, value in train_kwargs.items():
        lines.append(f"  {key}: {value}")

    log_path = run_dir / "run.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def train(env_name: str = config.DEFAULT_ENVIRONMENT, overrides: dict | None = None):
    """Train a fresh Q-table for `env_name` and pickle it under `runs/`.

    Hyperparameters come from `config.ENVIRONMENTS[env_name]["train_kwargs"]`.
    Pass `overrides` to replace selected keys for this run only (used by
    sweep.py). The run folder is created up front so TensorBoard can stream
    live metrics during training; a final PNG is written when training ends.
    """
    settings = config.ENVIRONMENTS[env_name]
    env = build_env(settings["env_id"], settings["env_kwargs"])
    train_kwargs = {**settings["train_kwargs"], **(overrides or {})}
    if overrides:
        print(f"Overrides: {overrides}")

    state_space = env.observation_space.n
    action_space = env.action_space.n
    print(f"{env_name}: {state_space} states, {action_space} actions")

    # Create the run folder first so live TensorBoard logs have a place to go.
    run_dir = _new_run_dir(env_name)
    tb_dir = run_dir / "tb"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"Run folder:   {run_dir}")
    print(f"Live curves:  tensorboard --logdir {tb_dir}")

    # For long runs, logging every episode to TB is fine but can add overhead;
    # default to every episode under 50k, otherwise every 10.
    n_eps = train_kwargs["n_training_episodes"]
    tb_log_every = 1 if n_eps <= 50_000 else 10

    qtable = initialize_q_table(state_space, action_space)
    try:
        qtable, history = run_training(
            env,
            qtable,
            n_training_episodes=n_eps,
            learning_rate=train_kwargs["learning_rate"],
            gamma=train_kwargs["gamma"],
            max_steps=train_kwargs["max_steps"],
            max_epsilon=train_kwargs["max_epsilon"],
            min_epsilon=train_kwargs["min_epsilon"],
            decay_rate=train_kwargs["decay_rate"],
            min_learning_rate=train_kwargs.get("min_learning_rate"),
            lr_decay_rate=train_kwargs.get("lr_decay_rate", 0.0),
            train_eval_every=train_kwargs.get("train_eval_every", 0),
            n_train_eval_episodes=train_kwargs.get("n_train_eval_episodes", 50),
            tb_writer=tb_writer,
            tb_log_every=tb_log_every,
        )
    finally:
        tb_writer.close()

    # Fresh env for eval (same idea as evaluate.py), using config eval settings.
    eval_env = gym.make(settings["env_id"], **settings["env_kwargs"])
    mean_reward, std_reward = evaluate_agent(
        eval_env,
        train_kwargs["max_steps"],
        train_kwargs["n_eval_episodes"],
        qtable,
        train_kwargs["eval_seed"],
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    model = {
        "env_id": settings["env_id"],
        "env_kwargs": settings["env_kwargs"],
        **train_kwargs,
        "qtable": qtable,
    }

    model_path = run_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Raw series for later analysis, plus a readable PNG snapshot of the curves.
    history_path = run_dir / "history.npz"
    np.savez_compressed(history_path, **{k: np.asarray(v) for k, v in history.items()})
    plot_path = run_dir / "training_curves.png"
    save_training_plots(history, plot_path, title=f"{env_name}  mean_reward={mean_reward:.2f}")

    log_path = _write_run_log(
        run_dir, env_name, settings, train_kwargs, model_path, mean_reward, std_reward
    )
    print(f"Saved model to {model_path}")
    print(f"Run log:      {log_path}")
    print(f"Final PNG:    {plot_path}")
    print(f"TensorBoard:  tensorboard --logdir {tb_dir}")

    return {
        "model": model,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_path": model_path,
        "run_dir": run_dir,
        "train_kwargs": train_kwargs,
        "history": history,
        "plot_path": plot_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=list(config.ENVIRONMENTS),
        default=config.DEFAULT_ENVIRONMENT,
        help="Which environment to train on.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.env)
