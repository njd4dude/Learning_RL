"""Train a Q-Learning agent on FrozenLake or Taxi and save its Q-table.

Usage:
    python train.py                              # non-slippery FrozenLake (default)
    python train.py --env frozenlake-slippery    # slippery FrozenLake
    python train.py --env taxi                   # Taxi-v4

Edit hyperparameters in config.py, then re-run. Each run writes a unique
folder under `runs/` with the model pickle and a text file listing the
hyperparameters used for that session.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import config
from q_learning import evaluate_agent, initialize_q_table, train as run_training


def build_env(env_id: str, env_kwargs: dict):
    return gym.make(env_id, render_mode="rgb_array", **env_kwargs)


def _new_run_dir(env_name: str) -> Path:
    """Unique folder for this training session, e.g. runs/20261021_153012_frozenlake-slippery."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.RUNS_FOLDER) / f"{stamp}_{env_name}"
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
        "",
        "hyperparameters:",
    ]
    for key, value in train_kwargs.items():
        lines.append(f"  {key}: {value}")

    log_path = run_dir / "run.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def train(env_name: str = config.DEFAULT_ENVIRONMENT):
    """Train a fresh Q-table for `env_name` and pickle it under `runs/`.

    Hyperparameters come from `config.ENVIRONMENTS[env_name]["train_kwargs"]`.
    Change them in config.py between runs; each run gets its own folder so
    nothing is overwritten. After training, runs the usual eval and records
    the score in run.txt.
    """
    settings = config.ENVIRONMENTS[env_name]
    env = build_env(settings["env_id"], settings["env_kwargs"])
    train_kwargs = settings["train_kwargs"]

    state_space = env.observation_space.n
    action_space = env.action_space.n
    print(f"{env_name}: {state_space} states, {action_space} actions")

    qtable = initialize_q_table(state_space, action_space)
    qtable = run_training(
        env,
        qtable,
        n_training_episodes=train_kwargs["n_training_episodes"],
        learning_rate=train_kwargs["learning_rate"],
        gamma=train_kwargs["gamma"],
        max_steps=train_kwargs["max_steps"],
        max_epsilon=train_kwargs["max_epsilon"],
        min_epsilon=train_kwargs["min_epsilon"],
        decay_rate=train_kwargs["decay_rate"],
    )

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

    run_dir = _new_run_dir(env_name)
    model_path = run_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    log_path = _write_run_log(
        run_dir, env_name, settings, train_kwargs, model_path, mean_reward, std_reward
    )
    print(f"Saved model to {model_path}")
    print(f"Run log:      {log_path}")

    return model


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
