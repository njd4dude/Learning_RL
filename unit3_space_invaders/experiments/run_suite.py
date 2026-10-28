"""Run the hyperparameter suite sequentially into logs_active/.

Usage (from unit3_space_invaders/, venv active):
    python experiments/run_suite.py
    python experiments/run_suite.py --only baseline lr_high
    python experiments/run_suite.py --seeds 0      # single seed, half the time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as train_mod

SUITE = [
    "baseline.yml",
    "lr_high.yml",
    "lr_low.yml",
    "buffer_small.yml",
    "batch_large.yml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of config stems, e.g. baseline lr_high",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Run every config once per seed (one seed can't separate luck from effect)",
    )
    # 10 episodes every 50k: less noisy points than Zoo's 5 every 25k, and
    # eval episodes (~2.5k steps each) don't eat the training time.
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    chosen = SUITE
    if args.only:
        wanted = {name if name.endswith(".yml") else f"{name}.yml" for name in args.only}
        chosen = [name for name in SUITE if name in wanted]
        missing = wanted - set(chosen)
        if missing:
            raise SystemExit(f"Unknown configs: {sorted(missing)}; pick from {SUITE}")

    first = True
    # Seed-major order: a full pass over every config finishes before the
    # next seed starts, so stopping early still leaves a complete comparison.
    for seed in args.seeds:
        for name in chosen:
            conf = Path(__file__).resolve().parent / name
            print("\n" + "=" * 60)
            print(f"Starting experiment: {name} (seed {seed})")
            print("=" * 60 + "\n")
            train_mod.train(
                conf_file=str(conf),
                seed=seed,
                save_replay_buffer=False,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                # Only auto-open TB for the first run; later runs keep writing
                # into the same logs_active/ tree the first server is watching.
                auto_tensorboard=first,
            )
            first = False
            print(f"\nFinished experiment: {name} (seed {seed})")


if __name__ == "__main__":
    main()
