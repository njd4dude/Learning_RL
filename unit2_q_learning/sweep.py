"""Run several training trials with different hyperparameters.

Edit the settings below, then:

    python sweep.py

Each trial uses the same path as a normal train (unique folder under runs/
with model.pkl + run.txt). When finished, a ranked summary is written next
to those runs.

Tips:
- Keep TRIALS small first (this can take a while on slippery with 100k+ eps).
- Set REPEATS=3 on promising configs later to average out seed luck.
- Only list keys you want to change; everything else comes from config.py.
"""

from __future__ import annotations

import itertools
import random
from datetime import datetime
from pathlib import Path

import config
from train import train

# ---------------------------------------------------------------------------
# Edit this block
# ---------------------------------------------------------------------------

ENV_NAME = "frozenlake-slippery"

# How many times to train each trial config (1 = once; 3 = average seeds).
REPEATS = 1

# "grid" expands GRID_VALUES into every combination.
# "list" runs TRIALS as written.
# "random" samples N_RANDOM trials from RANDOM_SPACE.
MODE = "random"

# Used when MODE == "list"
TRIALS: list[dict] = [
    {"n_training_episodes": 50_000, "learning_rate": 0.1, "min_learning_rate": 0.1, "lr_decay_rate": 0.0},
]

# Used when MODE == "grid" — keep this modest.
GRID_VALUES = {
    "n_training_episodes": [50_000, 100_000],
    "learning_rate": [0.1, 0.7],
    "lr_decay_rate": [0.0, 1e-5],
}

# For grid/random trials with lr_decay_rate == 0, pin min_learning_rate to learning_rate
# so decay stays fully off. When decay is on, use this floor:
DECAY_MIN_LEARNING_RATE = 0.001

# Wide random search over knobs that matter most for slippery Q-learning:
# - learning_rate / lr decay: how hard noisy slips rewrite the table
# - n_training_episodes: how much experience (incl. short 10k)
# - epsilon decay / min_epsilon: exploration schedule
# - gamma: how much the agent values future goal reward
N_RANDOM = 24
RANDOM_SEED = 42
RANDOM_SPACE = {
    "n_training_episodes": [10_000, 20_000, 50_000, 100_000, 200_000],
    "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5, 0.7],
    "lr_decay_rate": [0.0, 1e-6, 5e-6, 1e-5, 5e-5],
    "decay_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "min_epsilon": [0.01, 0.05, 0.1],
    "gamma": [0.9, 0.95, 0.99],
}

# ---------------------------------------------------------------------------


def _normalize_trial(trial: dict) -> dict:
    """Fill min_learning_rate sensibly when the trial sets lr decay on/off."""
    out = dict(trial)
    lr = out.get("learning_rate")
    lr_decay = out.get("lr_decay_rate")
    if lr is not None and lr_decay is not None and "min_learning_rate" not in out:
        out["min_learning_rate"] = lr if lr_decay == 0 else DECAY_MIN_LEARNING_RATE
    return out


def build_trials() -> list[dict]:
    if MODE == "list":
        return [_normalize_trial(t) for t in TRIALS]

    if MODE == "grid":
        keys = list(GRID_VALUES)
        combos = []
        for values in itertools.product(*(GRID_VALUES[k] for k in keys)):
            combos.append(_normalize_trial(dict(zip(keys, values))))
        return combos

    if MODE == "random":
        rng = random.Random(RANDOM_SEED)
        trials = []
        for _ in range(N_RANDOM):
            trial = {key: rng.choice(choices) for key, choices in RANDOM_SPACE.items()}
            trials.append(_normalize_trial(trial))
        return trials

    raise ValueError(f"Unknown MODE={MODE!r}; use 'grid', 'list', or 'random'.")


def run_sweep():
    trials = build_trials()
    total = len(trials) * REPEATS
    print(f"Sweep on {ENV_NAME}: {len(trials)} configs x {REPEATS} repeats = {total} runs")
    print(f"Mode={MODE}")

    results = []
    run_index = 0
    for trial_i, overrides in enumerate(trials, start=1):
        repeat_scores = []
        repeat_paths = []
        for rep in range(1, REPEATS + 1):
            run_index += 1
            print("\n" + "=" * 72)
            print(f"Trial {trial_i}/{len(trials)} repeat {rep}/{REPEATS} (run {run_index}/{total})")
            print(f"Overrides: {overrides}")
            print("=" * 72)

            outcome = train(ENV_NAME, overrides=overrides)
            repeat_scores.append(outcome["mean_reward"])
            repeat_paths.append(str(outcome["run_dir"]))

        mean_over_repeats = sum(repeat_scores) / len(repeat_scores)
        results.append(
            {
                "overrides": overrides,
                "scores": repeat_scores,
                "mean": mean_over_repeats,
                "run_dirs": repeat_paths,
            }
        )

    results.sort(key=lambda r: r["mean"], reverse=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(config.RUNS_FOLDER) / f"sweep_{stamp}_{ENV_NAME}_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"env: {ENV_NAME}",
        f"mode: {MODE}",
        f"repeats: {REPEATS}",
        f"finished_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "ranked results (best first):",
        "",
    ]
    for rank, row in enumerate(results, start=1):
        scores = ", ".join(f"{s:.4f}" for s in row["scores"])
        lines.append(f"#{rank}  mean={row['mean']:.4f}  scores=[{scores}]")
        lines.append(f"    overrides: {row['overrides']}")
        for path in row["run_dirs"]:
            lines.append(f"    run: {path}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Sweep finished. Top results:")
    for rank, row in enumerate(results[:5], start=1):
        print(f"  #{rank} mean={row['mean']:.4f}  {row['overrides']}")
    print(f"Summary: {summary_path}")
    return results


if __name__ == "__main__":
    run_sweep()
