"""Train a DQN or PPO agent on highway-v0.

Every run says what it tests, with all three of these:
    python train.py --name penalty10 --hypothesis "Fewer crashes" --change "Crash penalty -10"

Also takes `--algo ppo` for PPO, `--timesteps 50000` for a chosen length,
`--learning-rate 1e-4` or `--gamma 0.9` to override config.py, `--exploration-fraction 0.3`
for a DQN's epsilon to decay over 30% of the run instead of 10%, `--target-update-interval 500`
for a DQN's target network to update every 500 steps instead of 50, `--buffer-size 200000`
for a DQN to learn from its last 200,000 steps instead of 15,000, `--decisions-per-second 2`
to choose more often than once a second, `--default-reward` to train on
highway-env's reward rather than ENV_CONFIG, and `--collision-reward -2` or
`--high-speed-reward 2` to set the crash penalty or full-speed reward on top of
either, with `--right-lane-reward 0.5` doing the same for the rightmost lane's
reward. `--x-range 100` scales car distances by 100 m instead of 200 m, and can't
be combined with resuming a run that used another range. `--seed 1` sets the
run's random seed, so compare settings over several seeds, since one run can be
luck.

To continue from a saved model, in a new run folder:
    python train.py --resume runs/<run>/model.zip --timesteps 60000 --name ...

`--resume` also takes a file from a run's `checkpoints/`. `--timesteps` then means
additional steps, and the model keeps its saved learning rate unless you pass one.
Step counts carry on from the saved model. A resumed DQN reloads the run's
`replay_buffer.pkl` when resuming from its `model.zip`. Other files, and runs from
before that was saved, start with an empty buffer.

Each run writes a unique folder under `runs/` and adds an entry to
EXPERIMENTS.md. Ctrl+C stops early and still saves. Watch it with
`tensorboard --logdir runs`.
"""

import argparse
from datetime import datetime
from pathlib import Path

import config  # first, to limit math threads before torch loads

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl
import numpy as np

from evaluate import DrivingStats, describe, load_best, make_envs, run_algo, score

MODELS = {"dqn": DQN, "ppo": PPO}


class DrivingEvalCallback(EvalCallback):
    """Also logs crash rate, speed and distance, which compare across rewards."""

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        super()._log_success_callback(locals_, globals_)
        self.stats.update(locals_)

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self.stats = DrivingStats(self.eval_env.num_envs)
        keep_going = super()._on_step()
        if evaluating:
            for name, value in self.stats.results().items():
                self.logger.record(f"eval/{name}", value)
            self.logger.dump(self.num_timesteps)
        return keep_going


def _copy_into_buffer(old: ReplayBuffer, new: ReplayBuffer) -> None:
    """Copy a saved replay buffer into one of another size, oldest steps first.

    Each array is (steps per env, envs, ...). If the new one is smaller, only the
    newest steps fit.
    """
    if old.n_envs != new.n_envs:
        raise ValueError(f"Saved buffer has {old.n_envs} envs, this run has {new.n_envs}")
    order = np.r_[old.pos : old.buffer_size, 0 : old.pos] if old.full else np.arange(old.pos)
    order = order[-new.buffer_size :]
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        old_array, new_array = getattr(old, name, None), getattr(new, name, None)
        if old_array is not None and new_array is not None:
            new_array[: len(order)] = old_array[order]
    new.pos = len(order) % new.buffer_size
    new.full = len(order) == new.buffer_size


def _write_run_log(
    run_dir: Path,
    model: BaseAlgorithm,
    algo: str,
    seconds: float,
    results: dict,
    hypothesis: str,
    change: str,
    seed: int | None,
) -> None:
    lines = [
        f"hypothesis: {hypothesis}",
        f"change: {change}",
        f"env: {config.ENV_ID}",
        f"algo: {algo}",
        f"env_config: {config.ENV_CONFIG}",
        f"seed: {seed}",
        f"trained_at: {datetime.now().isoformat(timespec='seconds')}",
        f"total_steps: {model.num_timesteps}",
        f"train_seconds: {seconds:.0f}",
        f"result: {describe(results)}",
        "",
        "hyperparameters:",
    ]
    hyperparameters = {
        **config.ALGORITHMS[algo]["kwargs"],
        "learning_rate": model.learning_rate,
        "gamma": model.gamma,
    }
    if isinstance(model, DQN):
        hyperparameters["exploration_fraction"] = model.exploration_fraction
        hyperparameters["exploration_final_eps"] = model.exploration_final_eps
        hyperparameters["target_update_interval"] = model.target_update_interval
        hyperparameters["buffer_size"] = model.buffer_size
    lines += [f"  {key}: {value}" for key, value in hyperparameters.items()]
    (run_dir / "run.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_experiment(run_dir: Path, results: dict, hypothesis: str, change: str) -> None:
    entry = (
        f"\n### {run_dir.name}\n\n"
        f"- **Hypothesis:** {hypothesis}\n"
        f"- **Change:** {change}\n"
        f"- **Result:** {describe(results)} ({config.EVAL_EPISODES} episodes)\n"
        "- **Conclusion:** TODO\n"
    )
    with Path(config.EXPERIMENTS_FILE).open("a", encoding="utf-8") as log:
        log.write(entry)


def train(
    algo: str,
    timesteps: int | None,
    name: str,
    hypothesis: str,
    change: str,
    learning_rate: float | None = None,
    resume: Path | None = None,
    seed: int | None = None,
    gamma: float | None = None,
    exploration_fraction: float | None = None,
    target_update_interval: int | None = None,
    buffer_size: int | None = None,
) -> Path:
    settings = config.ALGORITHMS[algo]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.RUNS_FOLDER) / f"{stamp}_{algo}_{name}"
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}")

    env = make_envs(seed)
    eval_env = make_envs(None if seed is None else seed + 1000)
    overrides = {"learning_rate": learning_rate} if learning_rate else {}
    if gamma is not None:
        overrides["gamma"] = gamma
    if exploration_fraction is not None:
        overrides["exploration_fraction"] = exploration_fraction
    if target_update_interval is not None:
        overrides["target_update_interval"] = target_update_interval
    if buffer_size is not None:
        overrides["buffer_size"] = buffer_size
    if resume:
        model = MODELS[algo].load(resume, env=env, device="cpu", custom_objects=overrides)
        model.set_random_seed(seed)
        buffer = resume.parent / "replay_buffer.pkl"
        if resume.name == "model.zip" and buffer.is_file():
            if buffer_size is None:
                model.load_replay_buffer(buffer)
            else:
                _copy_into_buffer(load_from_pkl(buffer), model.replay_buffer)
            print(f"Loaded {buffer}: {model.replay_buffer.size() * config.N_ENVS} steps")
        change = f"{change} (resumed from {resume.as_posix()})"
    else:
        model = MODELS[algo](env=env, device="cpu", seed=seed, **{**settings["kwargs"], **overrides})
    model.set_logger(configure(str(run_dir / "tb"), ["stdout", "tensorboard"]))

    # These count calls, and each call is one step of every env.
    callbacks = [
        CheckpointCallback(
            save_freq=max(settings["checkpoint_steps"] // config.N_ENVS, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=algo,
        ),
        DrivingEvalCallback(
            eval_env,
            n_eval_episodes=config.TRAIN_EVAL_EPISODES,
            eval_freq=max(settings["eval_steps"] * config.decisions_per_second() // config.N_ENVS, 1),
            deterministic=True,
            best_model_save_path=str(run_dir),
            verbose=0,
        ),
    ]

    start = datetime.now()
    try:
        model.learn(
            total_timesteps=timesteps or settings["timesteps"],
            callback=callbacks,
            log_interval=settings["log_interval"],
            reset_num_timesteps=resume is None,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; saving the partially trained model.")
    seconds = (datetime.now() - start).total_seconds()

    model.save(run_dir / "model.zip")
    if isinstance(model, DQN):
        model.save_replay_buffer(run_dir / "replay_buffer.pkl")
    env.close()
    eval_env.close()

    print(f"Final eval over {config.EVAL_EPISODES} episodes...")
    results = score(load_best(run_dir))
    print(describe(results))
    _write_run_log(run_dir, model, algo, seconds, results, hypothesis, change, seed)
    _add_experiment(run_dir, results, hypothesis, change)
    print(f"Saved models to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", choices=config.ALGORITHMS, default="dqn")
    parser.add_argument("--timesteps", type=int, help="Defaults to the algorithm's value in config.py.")
    parser.add_argument("--learning-rate", type=float, help="Defaults to the algorithm's value in config.py.")
    parser.add_argument("--gamma", type=float, help="Defaults to the algorithm's value in config.py.")
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        help="DQN only: share of this run's steps that epsilon decays over. Defaults to 0.1.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        help="DQN only: steps between target network updates. Defaults to 50.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        help="DQN only: steps the replay buffer holds. Defaults to 15,000. Resuming keeps the saved steps.",
    )
    parser.add_argument("--default-reward", action="store_true", help="Ignore ENV_CONFIG for this run.")
    parser.add_argument("--collision-reward", type=float, help="Crash reward, like -2. Works with --default-reward.")
    parser.add_argument("--high-speed-reward", type=float, help="Full-speed reward, like 2. Works with --default-reward.")
    parser.add_argument("--right-lane-reward", type=float, help="Rightmost lane reward, like 0.5. Works with --default-reward.")
    parser.add_argument("--decisions-per-second", type=int, help="Defaults to 1. Scale --timesteps and --gamma to match.")
    parser.add_argument("--x-range", type=float, help="Metres that car distances are scaled by. Defaults to 200.")
    parser.add_argument("--seed", type=int, help="Repeat a setting with different seeds to see how much is luck.")
    parser.add_argument("--resume", type=Path, help="A saved .zip to continue from. Sets the algorithm too.")
    parser.add_argument("--name", required=True, help="Short label for the folder, like penalty10.")
    parser.add_argument("--hypothesis", required=True, help="What this run should show.")
    parser.add_argument("--change", required=True, help="What differs from the run it builds on.")
    args = parser.parse_args()
    if args.default_reward:
        config.ENV_CONFIG = {}
    if args.collision_reward is not None:
        config.ENV_CONFIG = {**config.ENV_CONFIG, "collision_reward": args.collision_reward}
    if args.high_speed_reward is not None:
        config.ENV_CONFIG = {**config.ENV_CONFIG, "high_speed_reward": args.high_speed_reward}
    if args.right_lane_reward is not None:
        config.ENV_CONFIG = {**config.ENV_CONFIG, "right_lane_reward": args.right_lane_reward}
    if args.decisions_per_second is not None:
        config.ENV_CONFIG = {**config.ENV_CONFIG, **config.decision_rate_config(args.decisions_per_second)}
    if args.x_range is not None:
        observation = config.observation_with_x_range(args.x_range)
        config.ENV_CONFIG = {**config.ENV_CONFIG, "observation": observation}
    algo = args.algo
    if args.resume:
        if not args.resume.is_file():
            parser.error(f"--resume file not found: {args.resume}")
        source = args.resume.parent
        algo = run_algo(source.parent if source.name == "checkpoints" else source)
    dqn_only = (args.exploration_fraction, args.target_update_interval, args.buffer_size)
    if algo != "dqn" and any(option is not None for option in dqn_only):
        parser.error("--exploration-fraction, --target-update-interval and --buffer-size only apply to DQN")
    train(
        algo, args.timesteps, args.name, args.hypothesis, args.change, args.learning_rate,
        args.resume, args.seed, args.gamma, args.exploration_fraction, args.target_update_interval,
        args.buffer_size,
    )
