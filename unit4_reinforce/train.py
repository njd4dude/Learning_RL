"""Train a REINFORCE agent on CartPole or Pixelcopter and save it.

Usage:
    python train.py                     # CartPole-v1 (default)
    python train.py --env pixelcopter   # Pixelcopter-PLE-v0
    python train.py --resume runs/<run>/model.pt                  # finish an interrupted run
    python train.py --resume runs/<run>/checkpoint.pt             # recover a crashed run
    python train.py --resume runs/<run>/model.pt --episodes 500   # train 500 more
    python train.py --resume runs/<run>/model.pt --lr 0.00005     # continue at a new learning rate

Edit hyperparameters in config.py, then re-run. Each run writes a unique
folder under `runs/` with model.pt, hyperparameters.json, run.txt, and live
TensorBoard logs under `tb/`. During training the folder also holds a rolling
checkpoint.pt, removed once model.pt is saved. Resuming writes a new folder
too, so the run it started from is never changed.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import config
from envs import make_env
from policy import build_policy
from reinforce import Progress, evaluate_agent, reinforce


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
        f"resumed_from: {results['resumed_from']}",
        f"interrupted: {results['interrupted']}",
        f"episodes_completed: {results['episodes_completed']}/{hyperparameters['n_training_episodes']}",
        f"mean_reward: {results['mean_reward']:.4f}",
        f"std_reward: {results['std_reward']:.4f}",
        f"total_steps: {results['total_steps']}",
        f"train_seconds: {results['train_seconds']:.1f}",
        f"fps: {results['fps']:.1f}",
        f"tensorboard: tensorboard --logdir {config.RUNS_FOLDER}",
        "",
        "hyperparameters:",
    ]
    lines += [f"  {key}: {value}" for key, value in hyperparameters.items()]

    log_path = run_dir / "run.txt"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def _progress_fields(progress: Progress) -> dict:
    """`progress` as the plain values stored in a checkpoint and in run.txt."""
    return {
        "episodes_completed": progress.episodes,
        "total_steps": progress.total_steps,
        "train_seconds": progress.train_seconds,
        "fps": progress.fps,
        "recent_scores": list(progress.recent_scores),
    }


def _restore_progress(checkpoint: dict) -> Progress:
    """Counters from a checkpoint. Older ones have episodes but no step counts."""
    return Progress(
        episodes=checkpoint["episodes_completed"],
        total_steps=checkpoint.get("total_steps", 0),
        train_seconds=checkpoint.get("train_seconds", 0.0),
        recent_scores=deque(checkpoint.get("recent_scores", []), maxlen=100),
    )


def _save_checkpoint(path: Path, env_name: str, hyperparameters: dict, policy, optimizer, **fields) -> None:
    """Write via a temp file, so a crash mid-save can't ruin the previous checkpoint."""
    temp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "env_name": env_name,
            "hyperparameters": hyperparameters,
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **fields,
        },
        temp_path,
    )
    os.replace(temp_path, path)


def train(
    env_name: str = config.DEFAULT_ENVIRONMENT,
    auto_tensorboard: bool | None = None,
    resume_from: str | Path | None = None,
    episodes: int | None = None,
    lr: float | None = None,
):
    """Train a policy and save it under `runs/`.

    `resume_from` continues a saved model.pt or checkpoint.pt with its own
    environment and hyperparameters, so `env_name` and config.py's values for
    that environment are ignored. `episodes` is how many to train this session;
    by default, whatever is left of the run's n_training_episodes. `lr` replaces
    the learning rate, saved or from config.py.
    """
    checkpoint = None
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=config.DEVICE, weights_only=True)
        env_name = checkpoint["env_name"]
    settings = config.ENVIRONMENTS[env_name]
    base_hyperparameters = checkpoint["hyperparameters"] if checkpoint else settings["hyperparameters"]

    progress = _restore_progress(checkpoint) if checkpoint else Progress()
    budget = base_hyperparameters["n_training_episodes"]
    n_episodes = budget - progress.episodes if episodes is None else episodes
    if n_episodes <= 0:
        raise SystemExit(
            f"Already trained {progress.episodes}/{budget} episodes. "
            "Pass --episodes N to train N more."
        )

    env = make_env(settings["env_id"])

    # The Hub's hyperparameters.json carries the env id and space sizes too,
    # so the model can be rebuilt from that file alone. A vector state is
    # stored as its length (course format); stacked frames as their full shape.
    obs_shape = env.observation_space.shape
    hyperparameters = {
        **base_hyperparameters,
        "n_training_episodes": progress.episodes + n_episodes,
        "lr": base_hyperparameters["lr"] if lr is None else lr,
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
    if checkpoint:
        policy.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Loading restores the saved lr too, so set the one this run uses.
            for group in optimizer.param_groups:
                group["lr"] = hyperparameters["lr"]
        else:
            print("No optimizer state in this model; starting Adam fresh.")
        print(f"Already trained for {progress.episodes} episodes; training {n_episodes} more.")

    run_dir = _new_run_dir(env_name)
    tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"), flush_secs=5)
    print(f"Run folder:   {run_dir}")

    if auto_tensorboard is None:
        auto_tensorboard = config.AUTO_START_TENSORBOARD
    if auto_tensorboard:
        _ensure_tensorboard(Path(config.RUNS_FOLDER))

    checkpoint_path = run_dir / "checkpoint.pt"

    def save_checkpoint() -> None:
        _save_checkpoint(
            checkpoint_path, env_name, hyperparameters, policy, optimizer, **_progress_fields(progress)
        )

    # Separate env for eval, never the one used for training.
    eval_env = make_env(settings["env_id"])
    try:
        scores = reinforce(
            env,
            policy,
            optimizer,
            n_episodes,
            hyperparameters["max_t"],
            hyperparameters["gamma"],
            settings["print_every"],
            progress,
            tb_writer=tb_writer,
            eval_env=eval_env,
            train_eval_every=settings.get("train_eval_every", 0),
            n_train_eval_episodes=settings.get("n_train_eval_episodes", 10),
            checkpoint_every=settings.get("checkpoint_every", 0),
            save_checkpoint=save_checkpoint,
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
            tb_writer.add_scalar("eval/final_mean_reward", mean_reward, progress.episodes)
            tb_writer.add_scalar("eval/final_std_reward", std_reward, progress.episodes)
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
        "interrupted": progress.episodes < hyperparameters["n_training_episodes"],
        "resumed_from": str(resume_from) if resume_from else None,
        **_progress_fields(progress),
    }

    model_path = run_dir / "model.pt"
    _save_checkpoint(model_path, env_name, hyperparameters, policy, optimizer, **results)
    # model.pt now has everything checkpoint.pt had.
    checkpoint_path.unlink(missing_ok=True)
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
        help="Which environment to train on. Ignored with --resume.",
    )
    parser.add_argument("--resume", metavar="MODEL_PATH", help="Continue from a saved model.pt or checkpoint.pt.")
    parser.add_argument("--episodes", type=int, help="Episodes to train this session.")
    parser.add_argument("--lr", type=float, help="Learning rate, replacing the saved or configured one.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.env, resume_from=args.resume, episodes=args.episodes, lr=args.lr)
