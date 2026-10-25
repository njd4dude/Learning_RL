"""Thin helpers for calling RL Baselines3 Zoo.

Zoo's entrypoints (`rl_zoo3.train`, `.enjoy`, …) parse `sys.argv` themselves.
- `run_zoo` / `run_zoo_checked`: subprocess (`python -u -m …`) — good for
  enjoy/hub; live logs on Windows.
- `run_zoo_train_inprocess`: import and call `rl_zoo3.train.train()` in this
  process so Ctrl+C is caught by Zoo (it saves the model after interrupt).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence


def run_zoo(module: str, args: Sequence[str]) -> int:
    """Run `python -u -m <module> <args>` and return the process exit code."""
    cmd = [sys.executable, "-u", "-m", module, *args]
    print(">", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def run_zoo_checked(module: str, args: Sequence[str]) -> None:
    """Like `run_zoo`, but raise if the Zoo command fails."""
    code = run_zoo(module, args)
    if code != 0:
        raise SystemExit(code)


def run_zoo_train_inprocess(args: Sequence[str]) -> None:
    """Run RL Zoo training in-process so KeyboardInterrupt still saves the model.

    Zoo's ExperimentManager.learn() catches Ctrl+C, then train() always calls
    save_trained_model(). That only works reliably if training isn't a child
    subprocess that may be killed before the save finishes.
    """
    from rl_zoo3.train import train as zoo_train

    argv = ["rl_zoo3.train", *args]
    print(">", " ".join([sys.executable, "-m", *argv]))
    old_argv = sys.argv
    sys.argv = argv
    try:
        zoo_train()
    finally:
        sys.argv = old_argv
