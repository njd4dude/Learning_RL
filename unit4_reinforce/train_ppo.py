"""Train PPO on the 7-number Pixelcopter, to compare with REINFORCE.

Usage:
    python train_ppo.py                                # new run
    python train_ppo.py --timesteps 2000000            # new run of a chosen length
    python train_ppo.py --resume runs/<run> --timesteps 5000000   # train 5M more

Each run writes a unique folder under `runs/` with model.zip, vecnormalize.pkl,
run.txt, and live TensorBoard logs under `tb/`. Periodic saves go to
`checkpoints/`. Ctrl+C stops early and still saves. Scores use sampled actions,
as the REINFORCE evaluation does, so the two compare directly.
"""

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import config
from envs import make_env
from train import _ensure_tensorboard, _new_run_dir

RUN_NAME = "pixelcopter-ppo"


def _make_game():
    return make_env(config.PIXELCOPTER_ENV_ID)


def _normalized_env(n_envs: int, stats_path: Path | None = None, training: bool = True, subprocess: bool = False):
    """Vectorized game with rescaled observations; rewards stay as true scores.

    The 7 numbers have very different ranges, which PPO learns poorly from.
    Evaluation must reuse training's scaling, so it's saved alongside the model.
    """
    venv = make_vec_env(_make_game, n_envs=n_envs, vec_env_cls=SubprocVecEnv if subprocess else None)
    if stats_path is not None:
        env = VecNormalize.load(str(stats_path), venv)
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=False)
    env.training = training
    return env


class PpoPlayer:
    """A saved PPO run with the same `act` as the REINFORCE policies, for replays."""

    def __init__(self, run_dir: str | Path):
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / "model.zip", device="cpu")
        # Only the saved scaling is needed, not this env's game.
        self.normalizer = _normalized_env(1, run_dir / "vecnormalize.pkl", training=False)
        self.normalizer.close()

    def act(self, state) -> tuple[int, None]:
        obs = self.normalizer.normalize_obs(state[None])
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action[0]), None


def _write_run_log(run_dir: Path, model: PPO, resumed_from: str | None, mean: float, std: float) -> None:
    lines = [
        f"env: {RUN_NAME}",
        f"trained_at: {datetime.now().isoformat(timespec='seconds')}",
        f"resumed_from: {resumed_from}",
        f"total_steps: {model.num_timesteps}",
        f"mean_reward: {mean:.4f}",
        f"std_reward: {std:.4f}",
        f"eval_episodes: {config.PPO_EVAL_EPISODES}",
        "",
        "hyperparameters:",
    ]
    lines += [f"  {key}: {value}" for key, value in config.PPO_KWARGS.items()]
    (run_dir / "run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(run_dir: str | Path, episodes: int = config.PPO_EVAL_EPISODES) -> tuple[float, float]:
    """Score a saved run with sampled actions; return (mean, std) reward."""
    run_dir = Path(run_dir)
    env = _normalized_env(1, run_dir / "vecnormalize.pkl", training=False)
    model = PPO.load(run_dir / "model.zip", device="cpu")
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=False)
    env.close()
    return float(mean), float(std)


def train(timesteps: int = config.PPO_TIMESTEPS, resume_from: str | Path | None = None) -> Path:
    run_dir = _new_run_dir(RUN_NAME)
    print(f"Run folder:   {run_dir}")
    if config.AUTO_START_TENSORBOARD:
        _ensure_tensorboard(Path(config.RUNS_FOLDER))

    if resume_from is None:
        env = _normalized_env(config.PPO_N_ENVS, subprocess=True)
        model = PPO(env=env, device="cpu", verbose=1, **config.PPO_KWARGS)
    else:
        source = Path(resume_from)
        env = _normalized_env(config.PPO_N_ENVS, source / "vecnormalize.pkl", subprocess=True)
        model = PPO.load(source / "model.zip", env=env, device="cpu")
        print(f"Already trained for {model.num_timesteps:,} steps; training {timesteps:,} more.")
    model.set_logger(configure(str(run_dir / "tb"), ["stdout", "tensorboard"]))

    eval_env = _normalized_env(1, training=False)
    callbacks = [
        CheckpointCallback(
            save_freq=max(config.PPO_CHECKPOINT_STEPS // config.PPO_N_ENVS, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            n_eval_episodes=config.PPO_TRAIN_EVAL_EPISODES,
            eval_freq=max(config.PPO_TRAIN_EVAL_STEPS // config.PPO_N_ENVS, 1),
            deterministic=False,
            verbose=0,
        ),
    ]

    try:
        model.learn(total_timesteps=timesteps, callback=callbacks, reset_num_timesteps=resume_from is None)
    except KeyboardInterrupt:
        print("\nInterrupted; saving the partially trained model.")

    model.save(run_dir / "model.zip")
    env.save(str(run_dir / "vecnormalize.pkl"))
    env.close()
    eval_env.close()

    print(f"Final eval over {config.PPO_EVAL_EPISODES} episodes...")
    mean, std = evaluate(run_dir)
    print(f"mean_reward={mean:.2f} +/- {std:.2f}")
    _write_run_log(run_dir, model, str(resume_from) if resume_from else None, mean, std)
    print(f"Saved model to {run_dir / 'model.zip'}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timesteps", type=int, default=config.PPO_TIMESTEPS, help="Steps to train this session.")
    parser.add_argument("--resume", metavar="RUN_DIR", help="Continue from a run folder holding model.zip.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.timesteps, args.resume)
