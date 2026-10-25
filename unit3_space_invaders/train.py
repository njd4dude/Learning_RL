"""Train a DQN agent with RL Baselines3 Zoo.

Usage:
    python train.py
    python train.py --env BeamRiderNoFrameskip-v4 -c dqn.yml
    python train.py -n 100000   # override yaml n_timesteps

By default training logs TensorBoard scalars under `tb/` and auto-starts
TensorBoard on port 6006 (same idea as unit2).
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

import config
from zoo import run_zoo_train_inprocess


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _ensure_tensorboard(logdir: Path, port: int | None = None) -> None:
    """Start TensorBoard in the background if nothing is already on `port`."""
    port = config.TENSORBOARD_PORT if port is None else port
    url = f"http://localhost:{port}/"
    if _port_open(port):
        print(f"TensorBoard already running at {url}")
        return

    logdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--port",
        str(port),
    ]
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


def train(
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.DEFAULT_ENV,
    log_folder: str = config.LOG_FOLDER,
    conf_file: str = config.DEFAULT_CONFIG,
    n_timesteps: int | None = None,
    progress: bool = True,
    seed: int | None = None,
    tensorboard_log: str | None = config.TENSORBOARD_LOG_FOLDER,
    auto_tensorboard: bool | None = None,
    device: str = config.DEVICE,
    vec_env: str = config.VEC_ENV,
) -> None:
    """Kick off `rl_zoo3.train` with our defaults / overrides."""
    conf_path = Path(conf_file)
    if not conf_path.is_file():
        raise FileNotFoundError(
            f"Hyperparameter file not found: {conf_path.resolve()}\n"
            f"(run from the unit3_space_invaders folder, or pass -c PATH)"
        )

    if auto_tensorboard is None:
        auto_tensorboard = config.AUTO_START_TENSORBOARD

    args = [
        "--algo",
        algo,
        "--env",
        env_id,
        "-f",
        log_folder,
        "-c",
        str(conf_path),
    ]
    if tensorboard_log:
        tb_dir = Path(tensorboard_log)
        tb_dir.mkdir(parents=True, exist_ok=True)
        args.extend(["-tb", str(tb_dir)])
        if auto_tensorboard:
            _ensure_tensorboard(tb_dir)
        else:
            print(f"Live curves:  tensorboard --logdir {tb_dir}")

    if n_timesteps is not None:
        args.extend(["-n", str(n_timesteps)])
    if seed is not None:
        args.extend(["--seed", str(seed)])
    if progress:
        args.append("--progress")
    if device:
        args.extend(["--device", device])
    if vec_env:
        args.extend(["--vec-env", vec_env])

    # In-process so Ctrl+C is handled by Zoo and still writes the final zip
    # (same idea as unit2's interrupt → save partial model).
    run_zoo_train_inprocess(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", default=config.DEFAULT_ALGO)
    parser.add_argument("--env", default=config.DEFAULT_ENV)
    parser.add_argument("-f", "--log-folder", default=config.LOG_FOLDER)
    parser.add_argument("-c", "--conf", default=config.DEFAULT_CONFIG)
    parser.add_argument(
        "-n",
        "--n-timesteps",
        type=int,
        default=None,
        help="Override n_timesteps from the yaml config",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Zoo's tqdm/rich progress bar",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard-log",
        default=config.TENSORBOARD_LOG_FOLDER,
        help="TensorBoard log dir (empty string to disable)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging and auto-start",
    )
    parser.add_argument(
        "--no-auto-tensorboard",
        action="store_true",
        help="Log TB events but do not auto-launch the TensorBoard server",
    )
    parser.add_argument(
        "--device",
        default=config.DEVICE,
        help='Torch device for SB3 (e.g. "auto", "cuda", "cpu")',
    )
    parser.add_argument(
        "--vec-env",
        choices=["dummy", "subproc"],
        default=config.VEC_ENV,
        help="VecEnv type (use subproc when n_envs > 1)",
    )
    args = parser.parse_args()

    tb = None if args.no_tensorboard else (args.tensorboard_log or None)
    train(
        algo=args.algo,
        env_id=args.env,
        log_folder=args.log_folder,
        conf_file=args.conf,
        n_timesteps=args.n_timesteps,
        progress=not args.no_progress,
        seed=args.seed,
        tensorboard_log=tb,
        auto_tensorboard=False if args.no_auto_tensorboard or args.no_tensorboard else None,
        device=args.device,
        vec_env=args.vec_env,
    )


if __name__ == "__main__":
    main()
