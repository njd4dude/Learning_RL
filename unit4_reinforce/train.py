"""Train a REINFORCE agent on CartPole or Pixelcopter and save it.

Usage:
    python train.py                     # CartPole-v1 (default)
    python train.py --env pixelcopter   # Pixelcopter-PLE-v0

Edit hyperparameters in config.py, then re-run. Each run writes a unique
folder under `runs/` with model.pt, hyperparameters.json, run.txt, and live
TensorBoard logs under `tb/`.
"""

import argparse
import json
import socket
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from envs import make_env
from policy import build_policy
from reinforce import evaluate_agent, reinforce


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ensure_tensorboard(logdir: Path, port: int = config.TENSORBOARD_PORT) -> None:
    """Start TensorBoard in the background if nothing is already on `port`."""
    url = f"http://localhost:{port}/"
    if _port_open(port):
        print(f"TensorBoard already running at {url}")
        return

    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--port", str(port)]
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"Started TensorBoard at {url}  (logdir={logdir})")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _new_run_dir(env_name: str) -> Path:
    """Unique folder for this training session, e.g. runs/20261103_153012_cartpole."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.RUNS_FOLDER) / f"{stamp}_{env_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_log(run_dir: Path, env_name: str, hyperparameters: dict, results: dict) -> Path:
    """Human-readable record of what was trained and how it scored."""
    lines = [
        f"env: {env_name}",
        f"trained_at: {datetime.now().isoformat(timespec='seconds')}",
        f"interrupted: {results['interrupted']}",
        f"episodes_completed: {results['episodes_completed']}/{hyperparameters['n_training_episodes']}",
        f"mean_reward: {results['mean_reward']:.4f}",
        f"std_reward: {results['std_reward']:.4f}",
        f"tensorboard: tensorboard --logdir {config.RUNS_FOLDER}",
        "",
        "hyperparameters:",
    ]
    lines += [f"  {key}: {value}" for key, value in hyperparameters.items()]

    log_path = run_dir / "run.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def train(env_name: str = config.DEFAULT_ENVIRONMENT, auto_tensorboard: bool | None = None):
    """Train a fresh policy for `env_name` and save it under `runs/`."""
    settings = config.ENVIRONMENTS[env_name]
    env = make_env(settings["env_id"])

    # The Hub's hyperparameters.json carries the env id and space sizes too,
    # so the model can be rebuilt from that file alone. A vector state is
    # stored as its length (course format); stacked frames as their full shape.
    obs_shape = env.observation_space.shape
    hyperparameters = {
        **settings["hyperparameters"],
        "env_id": settings["env_id"],
        "state_space": obs_shape[0] if len(obs_shape) == 1 else list(obs_shape),
        "action_space": int(env.action_space.n),
    }
    print(
        f"{env_name}: observation shape {obs_shape}, "
        f"{hyperparameters['action_space']} actions"
    )

    policy = build_policy(hyperparameters, config.DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])

    run_dir = _new_run_dir(env_name)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=5)
    print(f"Run folder:   {run_dir}")

    if auto_tensorboard is None:
        auto_tensorboard = config.AUTO_START_TENSORBOARD
    if auto_tensorboard:
        _ensure_tensorboard(Path(config.RUNS_FOLDER))

    # Separate env for eval, never the one used for training.
    eval_env = make_env(settings["env_id"])
    try:
        scores = reinforce(
            env,
            policy,
            optimizer,
            hyperparameters["n_training_episodes"],
            hyperparameters["max_t"],
            hyperparameters["gamma"],
            settings["print_every"],
            tb_writer=tb_writer,
            eval_env=eval_env,
            train_eval_every=settings.get("train_eval_every", 0),
            n_train_eval_episodes=settings.get("n_train_eval_episodes", 10),
        )

        print(f"Final eval over {hyperparameters['n_evaluation_episodes']} episodes (Ctrl+C to skip)...")
        try:
            mean_reward, std_reward = evaluate_agent(
                eval_env,
                hyperparameters["max_t"],
                hyperparameters["n_evaluation_episodes"],
                policy,
            )
            # Own tag: it uses n_evaluation_episodes, not the mid-training count,
            # so it shouldn't share a curve with eval/mean_reward.
            tb_writer.add_scalar("eval/final_mean_reward", mean_reward, len(scores))
            tb_writer.add_scalar("eval/final_std_reward", std_reward, len(scores))
        except KeyboardInterrupt:
            # A second Ctrl+C must not lose the trained policy: skip the score,
            # save anyway, and score it later with `python main.py evaluate`.
            mean_reward = std_reward = float("nan")
            print("\nFinal eval skipped; saving the model without a score.")
    finally:
        tb_writer.close()
        env.close()
        eval_env.close()
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "episodes_completed": len(scores),
        "interrupted": len(scores) < hyperparameters["n_training_episodes"],
    }

    model_path = run_dir / "model.pt"
    torch.save(
        {
            "env_name": env_name,
            "hyperparameters": hyperparameters,
            "state_dict": policy.state_dict(),
            **results,
        },
        model_path,
    )
    with open(run_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
    log_path = _write_run_log(run_dir, env_name, hyperparameters, results)

    print(f"Saved model to {model_path}")
    print(f"Run log:      {log_path}")
    return {"policy": policy, "scores": scores, "model_path": model_path, **results}


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
