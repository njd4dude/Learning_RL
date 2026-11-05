"""Train a DQN or PPO agent on highway-v0.

Every run says what it tests, with all three of these:
    python train.py --name penalty10 --hypothesis "Fewer crashes" --change "Crash penalty -10"

Also takes `--algo ppo` for PPO, and `--timesteps 50000` for a chosen length.

Each run writes a unique folder under `runs/` and adds an entry to
EXPERIMENTS.md. Ctrl+C stops early and still saves. Watch it with
`tensorboard --logdir runs`.
"""

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

import config
from evaluate import describe, load_best, make_envs, score

MODELS = {"dqn": DQN, "ppo": PPO}


def _write_run_log(
    run_dir: Path, model: BaseAlgorithm, algo: str, seconds: float, results: dict, hypothesis: str, change: str
) -> None:
    lines = [
        f"hypothesis: {hypothesis}",
        f"change: {change}",
        f"env: {config.ENV_ID}",
        f"algo: {algo}",
        f"env_config: {config.ENV_CONFIG}",
        f"trained_at: {datetime.now().isoformat(timespec='seconds')}",
        f"total_steps: {model.num_timesteps}",
        f"train_seconds: {seconds:.0f}",
        f"result: {describe(results)}",
        "",
        "hyperparameters:",
    ]
    lines += [f"  {key}: {value}" for key, value in config.ALGORITHMS[algo]["kwargs"].items()]
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


def train(algo: str, timesteps: int | None, name: str, hypothesis: str, change: str) -> Path:
    settings = config.ALGORITHMS[algo]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.RUNS_FOLDER) / f"{stamp}_{algo}_{name}"
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}")

    env = make_envs()
    eval_env = make_envs()
    model = MODELS[algo](env=env, device="cpu", **settings["kwargs"])
    model.set_logger(configure(str(run_dir / "tb"), ["stdout", "tensorboard"]))

    # These count calls, and each call is one step of every env.
    callbacks = [
        CheckpointCallback(
            save_freq=max(settings["checkpoint_steps"] // config.N_ENVS, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=algo,
        ),
        EvalCallback(
            eval_env,
            n_eval_episodes=config.TRAIN_EVAL_EPISODES,
            eval_freq=max(settings["eval_steps"] // config.N_ENVS, 1),
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
        )
    except KeyboardInterrupt:
        print("\nInterrupted; saving the partially trained model.")
    seconds = (datetime.now() - start).total_seconds()

    model.save(run_dir / "model.zip")
    env.close()
    eval_env.close()

    print(f"Final eval over {config.EVAL_EPISODES} episodes...")
    results = score(load_best(run_dir))
    print(describe(results))
    _write_run_log(run_dir, model, algo, seconds, results, hypothesis, change)
    _add_experiment(run_dir, results, hypothesis, change)
    print(f"Saved models to {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algo", choices=config.ALGORITHMS, default="dqn")
    parser.add_argument("--timesteps", type=int, help="Defaults to the algorithm's value in config.py.")
    parser.add_argument("--name", required=True, help="Short label for the folder, like penalty10.")
    parser.add_argument("--hypothesis", required=True, help="What this run should show.")
    parser.add_argument("--change", required=True, help="What differs from the run it builds on.")
    args = parser.parse_args()
    train(args.algo, args.timesteps, args.name, args.hypothesis, args.change)
