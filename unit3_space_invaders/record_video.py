"""Record mp4 replays of a trained RL Zoo DQN agent (no Hub upload needed).

Usage (from this folder, with the project venv active):
    python record_video.py
    python record_video.py --episodes 3 --n-timesteps 3000
    python record_video.py -f logs/ --algo dqn --env SpaceInvadersNoFrameskip-v4

Videos are written under videos/ by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

import config
import rl_zoo3.import_envs  # noqa: F401
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import get_model_path


def _as_rgb_frame(render_out) -> np.ndarray:
    """Normalize VecEnv / Gymnasium render output to a single HxWx3 uint8 frame."""
    frame = render_out
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _resolve_model(
    folder: str,
    algo: str,
    env_name: EnvironmentName,
    exp_id: int,
    load_best: bool,
) -> tuple[str, str]:
    """Return (model_path, log_path), falling back to best_model.zip if needed."""
    try:
        _, model_path, log_path = get_model_path(
            exp_id, folder, algo, env_name, load_best=load_best
        )
        return model_path, log_path
    except ValueError:
        if load_best:
            raise
        print("Final model zip missing; trying best_model.zip instead.")
        _, model_path, log_path = get_model_path(
            exp_id, folder, algo, env_name, load_best=True
        )
        return model_path, log_path


def record(
    folder: str = config.LOG_FOLDER,
    algo: str = config.DEFAULT_ALGO,
    env_id: str = config.DEFAULT_ENV,
    exp_id: int = 0,
    n_timesteps: int = config.ENJOY_TIMESTEPS,
    episodes: int | None = None,
    out_dir: str = config.VIDEO_FOLDER,
    fps: int = config.VIDEO_FPS,
    seed: int = 0,
    deterministic: bool = False,
    load_best: bool = False,
) -> list[Path]:
    """Play the saved model and write one mp4 per finished Atari episode."""
    env_name = EnvironmentName(env_id)
    model_path, log_path = _resolve_model(folder, algo, env_name, exp_id, load_best)
    print(f"Loading {model_path}")

    set_random_seed(seed)

    stats_path = str(Path(log_path) / str(env_name))
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, test_mode=True)

    env = create_test_env(
        env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=seed,
        should_render=True,
        hyperparams=hyperparams,
        env_kwargs={"render_mode": "rgb_array"},
        vec_env_cls=ExperimentManager.default_vec_env_cls,
    )

    model = ALGOS[algo].load(
        model_path,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
        buffer_size=1,
        device="auto",
        seed=seed,
    )

    is_atari = ExperimentManager.is_atari(env_name.gym_id)
    # Match enjoy.py: Atari defaults to stochastic unless --deterministic.
    use_deterministic = deterministic or not is_atari

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    obs = env.reset()
    frames: list[np.ndarray] = [_as_rgb_frame(env.render())]
    saved: list[Path] = []
    episode_idx = 0
    steps = 0
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    try:
        while steps < n_timesteps:
            if episodes is not None and episode_idx >= episodes:
                break

            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=use_deterministic,
            )
            obs, _reward, done, infos = env.step(action)
            episode_start = done
            frames.append(_as_rgb_frame(env.render()))
            steps += 1

            episode_info = None
            if infos is not None:
                episode_info = infos[0].get("episode")

            if episode_info is not None:
                out = out_root / f"{env_name}-episode-{episode_idx}.mp4"
                imageio.mimsave(out, frames, fps=fps)
                score = float(episode_info["r"])
                length = int(episode_info["l"])
                print(f"Episode {episode_idx}: score={score:.1f} length={length}")
                print(f"Saved video to {out}")
                saved.append(out)
                episode_idx += 1
                frames = [_as_rgb_frame(env.render())]
    finally:
        env.close()

    if not saved:
        # Timestep budget ended mid-episode — still write what we have.
        out = out_root / f"{env_name}-partial.mp4"
        imageio.mimsave(out, frames, fps=fps)
        print(f"No full episode finished; saved partial clip to {out}")
        saved.append(out)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--folder", default=config.LOG_FOLDER, help="RL Zoo log folder")
    parser.add_argument("--algo", default=config.DEFAULT_ALGO)
    parser.add_argument("--env", default=config.DEFAULT_ENV)
    parser.add_argument("--exp-id", type=int, default=0, help="0 = latest experiment")
    parser.add_argument("-n", "--n-timesteps", type=int, default=config.ENJOY_TIMESTEPS)
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.VIDEO_EPISODES,
        help="Stop after this many finished episodes (0 = only use -n)",
    )
    parser.add_argument("--out-dir", default=config.VIDEO_FOLDER)
    parser.add_argument("--fps", type=int, default=config.VIDEO_FPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Greedy actions (Atari enjoy defaults to stochastic)",
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        help="Load best_model.zip instead of the final env zip",
    )
    args = parser.parse_args()

    episodes = None if args.episodes == 0 else args.episodes
    record(
        folder=args.folder,
        algo=args.algo,
        env_id=args.env,
        exp_id=args.exp_id,
        n_timesteps=args.n_timesteps,
        episodes=episodes,
        out_dir=args.out_dir,
        fps=args.fps,
        seed=args.seed,
        deterministic=args.deterministic,
        load_best=args.load_best,
    )


if __name__ == "__main__":
    main()
