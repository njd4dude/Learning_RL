"""Score a trained agent with its best actions, reporting reward and crash rate.

Usage:
    python evaluate.py                  # newest run
    python evaluate.py runs/<run>       # a specific run folder
    python evaluate.py --config-reward  # on config.py's reward, not the run's own

Runs are scored on the reward they trained with by default. Mean speed, distance
and crash rate don't depend on the reward, so they compare across all runs. Each
score is added to `eval.txt` in the run folder.
"""

import argparse
import ast
from datetime import datetime
from pathlib import Path

import config  # first, to limit math threads before numpy and torch load

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

gym.register_envs(highway_env)


def make_envs(seed: int | None = None):
    """Parallel copies of the game, since one alone is very slow."""
    return make_vec_env(
        config.ENV_ID,
        n_envs=config.N_ENVS,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=config.ENV_CONFIG),
    )


def latest_run() -> Path:
    models = sorted(Path(config.RUNS_FOLDER).glob("*/model.zip"))
    if not models:
        raise SystemExit(f"No trained models under {config.RUNS_FOLDER}/. Run train.py first.")
    return models[-1].parent


def run_algo(run_dir: Path) -> str:
    """Read the algorithm from a folder named <date>_<time>_<algo>[_<name>]."""
    return run_dir.name.split("_")[2]


def load_best(run_dir: Path) -> DQN | PPO:
    """Load a run's best model, or its last one if it has no best saved.

    Training can get worse late on, so the last step is often not the best.
    """
    path = run_dir / "best_model.zip"
    if not path.is_file():
        path = run_dir / "model.zip"
    print(f"Loading {path}")
    model_class = PPO if run_algo(run_dir) == "ppo" else DQN
    return model_class.load(path, device="cpu")


def trained_reward(run_dir: Path) -> dict:
    """Read a run's reward settings. Runs from before they were logged used the default."""
    run_log = run_dir / "run.txt"
    if not run_log.is_file():  # stopped before it finished
        return config.ENV_CONFIG
    for line in run_log.read_text(encoding="utf-8").splitlines():
        if line.startswith("env_config: "):
            return ast.literal_eval(line.removeprefix("env_config: "))
    return {}


def score(model: DQN | PPO, episodes: int = config.EVAL_EPISODES) -> dict:
    env = make_envs()
    distances = []
    travelled = np.zeros(env.num_envs)

    def track_distance(local_vars: dict, _globals: dict) -> None:
        # Decisions are one simulated second apart, so each speed adds that many metres.
        i = local_vars["i"]
        travelled[i] += local_vars["info"]["speed"]
        if local_vars["done"]:
            distances.append(float(travelled[i]))
            travelled[i] = 0

    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=True,
        return_episode_rewards=True,
        callback=track_distance,
    )
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(lengths)),
        "crash_rate": float(np.mean(np.array(lengths) < config.MAX_EPISODE_STEPS)),
        "mean_distance": float(np.mean(distances)),
        "mean_speed": float(np.sum(distances) / np.sum(lengths)),
    }


def describe(results: dict) -> str:
    return (
        f"mean_reward={results['mean_reward']:.2f} +/- {results['std_reward']:.2f}, "
        f"mean_steps={results['mean_steps']:.1f}/{config.MAX_EPISODE_STEPS}, "
        f"crash_rate={results['crash_rate']:.0%}, "
        f"mean_speed={results['mean_speed']:.1f}m/s, mean_distance={results['mean_distance']:.0f}m"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Run folder. Defaults to the newest.")
    parser.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument("--config-reward", action="store_true", help="Use config.py's reward, not the run's own.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run()
    if not args.config_reward:
        config.ENV_CONFIG = trained_reward(run_dir)
    print(f"Scoring {run_dir} over {args.episodes} episodes...")
    result = describe(score(load_best(run_dir), args.episodes))
    print(result)
    stamp = datetime.now().isoformat(timespec="minutes")
    with (run_dir / "eval.txt").open("a", encoding="utf-8") as log:
        log.write(f"{stamp}  episodes={args.episodes}  reward={config.ENV_CONFIG}  {result}\n")
