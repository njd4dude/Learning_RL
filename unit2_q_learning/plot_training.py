"""Save training-curve plots for a Q-learning run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_training_plots(history: dict, out_path: str | Path, title: str = ""):
    """Write a multi-panel PNG of the metrics collected during training."""
    out_path = Path(out_path)
    rewards = np.asarray(history["episode_reward"], dtype=float)
    lengths = np.asarray(history["episode_length"], dtype=float)
    epsilons = np.asarray(history["epsilon"], dtype=float)
    lrs = np.asarray(history["learning_rate"], dtype=float)
    eval_xs = np.asarray(history.get("eval_episode", []), dtype=float)
    eval_ys = np.asarray(history.get("eval_mean_reward", []), dtype=float)

    n = len(rewards)
    window = max(1, n // 100)  # ~100 smooth points across the run
    episodes = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    if title:
        fig.suptitle(title)

    # Episode reward + rolling mean (+ optional mid-train greedy eval)
    ax = axes[0, 0]
    ax.plot(episodes, rewards, color="0.75", linewidth=0.4, label="episode reward")
    if n >= window:
        rolled = _rolling_mean(rewards, window)
        ax.plot(
            np.arange(window - 1, n),
            rolled,
            color="C0",
            linewidth=1.5,
            label=f"rolling mean (w={window})",
        )
    if len(eval_xs) > 0:
        ax.plot(eval_xs, eval_ys, "o-", color="C3", linewidth=1.5, label="greedy eval")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("Reward during training")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(episodes, lengths, color="0.75", linewidth=0.4, label="episode length")
    if n >= window:
        ax.plot(
            np.arange(window - 1, n),
            _rolling_mean(lengths, window),
            color="C1",
            linewidth=1.5,
            label=f"rolling mean (w={window})",
        )
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.set_title("Episode length")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color="C2", linewidth=1.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax.set_title("Exploration (epsilon)")

    ax = axes[1, 1]
    ax.plot(episodes, lrs, color="C4", linewidth=1.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
