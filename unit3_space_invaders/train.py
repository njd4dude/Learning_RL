"""Train a DQN agent with RL Baselines3 Zoo.

Usage:
    python train.py
    python train.py --env BeamRiderNoFrameskip-v4 -c dqn.yml
    python train.py -n 100000   # override yaml n_timesteps
    python train.py -i latest -n 500000   # resume (reloads replay_buffer.pkl)

By default TensorBoard events are written under each Zoo run folder
(`logs/<algo>/<env>_N/tb/`) and TensorBoard auto-starts on port 6006
watching `logs/`.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import webbrowser
import zipfile
from pathlib import Path

import config
from zoo import run_zoo_train_inprocess, run_zoo_train_resume

# SB3 writes these last; a second Ctrl+C mid-save leaves a zip with only
# ``data`` / ``pytorch_variables.pth`` — load then fails with KeyError
# ``policy.optimizer``.
_SB3_REQUIRED_ZIP_MEMBERS = ("policy.pth", "policy.optimizer.pth")


def _is_loadable_sb3_zip(path: Path) -> bool:
    """True if ``path`` looks like a complete SB3 checkpoint zip."""
    if not path.is_file() or path.suffix != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        return False
    return all(name in names for name in _SB3_REQUIRED_ZIP_MEMBERS)


def resolve_trained_agent(
    spec: str,
    *,
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.DEFAULT_ENV,
    log_folder: str = config.LOG_FOLDER,
    load_best: bool = False,
) -> Path:
    """Resolve ``-i`` to a concrete ``.zip`` path.

    Accepts a filesystem path, ``latest`` / ``0`` (newest run), or an exp id
    like ``16``. Prefers the final env zip; falls back to ``best_model.zip``
    if that run was interrupted before / during the final save (missing or
    truncated zip).
    """
    from huggingface_sb3 import EnvironmentName
    from rl_zoo3.utils import get_latest_run_id, get_model_path

    raw = Path(spec)
    if raw.is_file() and raw.suffix == ".zip":
        path = raw.resolve()
        if not _is_loadable_sb3_zip(path):
            raise FileNotFoundError(
                f"{path} is not a complete SB3 checkpoint "
                f"(missing {', '.join(_SB3_REQUIRED_ZIP_MEMBERS)} — "
                "often a Ctrl+C mid-save). Prefer best_model.zip or an earlier exp."
            )
        return path

    key = spec.strip().lower()
    if key in {"latest", "0"}:
        exp_id = 0
    elif spec.isdigit():
        exp_id = int(spec)
    else:
        raise FileNotFoundError(
            f"trained-agent must be a .zip path, 'latest', or an exp id; got: {spec!r}"
        )

    env_name = EnvironmentName(env_id)
    algo_dir = Path(log_folder) / algo
    if exp_id == 0:
        exp_id = get_latest_run_id(str(algo_dir), env_name)
        print(f"Loading latest experiment, id={exp_id}")

    def _usable(path: Path) -> Path | None:
        return path.resolve() if _is_loadable_sb3_zip(path) else None

    # Walk downward so a missing/truncated newest zip doesn't block resume.
    for candidate_id in range(exp_id, 0, -1):
        try:
            _, model_path, _ = get_model_path(
                candidate_id,
                log_folder,
                algo,
                env_name,
                load_best=load_best,
            )
        except (ValueError, AssertionError, FileNotFoundError):
            model_path = None

        candidates: list[Path] = []
        if model_path is not None:
            path = Path(model_path)
            if path.suffix != ".zip":
                path = Path(f"{model_path}.zip")
            candidates.append(path)
        if not load_best:
            candidates.append(
                algo_dir / f"{env_name}_{candidate_id}" / "best_model.zip"
            )

        for path in candidates:
            chosen = _usable(path)
            if chosen is None:
                if path.is_file():
                    print(f"Skipping incomplete checkpoint: {path}")
                continue
            if candidate_id != exp_id:
                print(f"No usable zip in exp {exp_id}; using {chosen}")
            elif path.name == "best_model.zip" and not load_best:
                print(
                    f"Final zip in exp {candidate_id} incomplete; "
                    "using best_model.zip"
                )
            return chosen

    raise FileNotFoundError(
        f"No loadable checkpoint zip found under {algo_dir} for {env_id} "
        f"(tried exp ids {exp_id} → 1)"
    )


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
    eval_freq: int | None = None,
    trained_agent: str | None = None,
    load_best: bool = False,
    resume_warmup: int | None = None,
    save_replay_buffer: bool = config.SAVE_REPLAY_BUFFER,
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
        # Non-empty `-tb` enables Zoo/SB3 logging; zoo.py redirects events into
        # each run's ``save_path/tb/``. The server watches the log tree so run
        # names match experiment folders (e.g. …/SpaceInvaders…_40/tb).
        tb_watch = Path(tensorboard_log)
        tb_watch.mkdir(parents=True, exist_ok=True)
        args.extend(["-tb", str(tb_watch)])
        if auto_tensorboard:
            _ensure_tensorboard(tb_watch)
        else:
            print(f"Live curves:  tensorboard --logdir {tb_watch}")

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
    if eval_freq is not None:
        args.extend(["--eval-freq", str(eval_freq)])
    if save_replay_buffer:
        # Zoo writes replay_buffer.pkl beside the final zip (also after Ctrl+C)
        # and reloads it automatically when that zip is passed to -i.
        args.append("--save-replay-buffer")
    if trained_agent:
        agent_path = resolve_trained_agent(
            trained_agent,
            algo=algo,
            env_id=env_id,
            log_folder=log_folder,
            load_best=load_best,
        )
        print(f"Resuming from {agent_path}")
        args.extend(["-i", str(agent_path)])
        warmup = (
            config.RESUME_WARMUP_STEPS if resume_warmup is None else resume_warmup
        )
        # New Zoo run folder + loaded weights; timesteps/ε keep going.
        run_zoo_train_resume(args, agent_path, resume_warmup=warmup)
        return

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
        help=(
            "TensorBoard --logdir root (default: logs/). Events for each run "
            "are written under that run's own tb/ subfolder"
        ),
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
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Evaluate every N timesteps (Zoo default 25000; -1 disables)",
    )
    parser.add_argument(
        "-i",
        "--trained-agent",
        default=None,
        help=(
            "Load weights into a new run folder: .zip path, 'latest', or exp id "
            "(timestep/ε continue; -n is additional steps)"
        ),
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        help="With -i latest/exp-id, load best_model.zip instead of the final zip",
    )
    parser.add_argument(
        "--resume-warmup",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With -i and no saved replay buffer, collect N steps before gradient "
            f"updates (default: {config.RESUME_WARMUP_STEPS}; 0 disables)"
        ),
    )
    parser.add_argument(
        "--no-save-replay-buffer",
        action="store_true",
        help="Don't write replay_buffer.pkl (saves disk; resume falls back to warm-up)",
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
        eval_freq=args.eval_freq,
        trained_agent=args.trained_agent,
        load_best=args.load_best,
        resume_warmup=args.resume_warmup,
        save_replay_buffer=not args.no_save_replay_buffer,
    )


if __name__ == "__main__":
    main()
