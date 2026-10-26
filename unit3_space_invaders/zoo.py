"""Thin helpers for calling RL Baselines3 Zoo.

Zoo's entrypoints (`rl_zoo3.train`, `.enjoy`, …) parse `sys.argv` themselves.
- `run_zoo` / `run_zoo_checked`: subprocess (`python -u -m …`) — good for
  enjoy/hub; live logs on Windows.
- `run_zoo_train_inprocess`: import and call `rl_zoo3.train.train()` in this
  process so Ctrl+C is caught by Zoo (it saves the model after interrupt).
- `run_zoo_train_resume`: same, but keep training in the checkpoint's folder
  with a continuing timestep counter (unit1-style resume).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np


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


def _hydrate_eval_callback(callbacks: list, resume_dir: str) -> None:
    """Reload prior EvalCallback history from evaluations.npz so new evals append."""
    from stable_baselines3.common.callbacks import EvalCallback

    npz_path = Path(resume_dir) / "evaluations.npz"
    if not npz_path.is_file():
        return

    data = np.load(npz_path, allow_pickle=True)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    for callback in callbacks:
        if not isinstance(callback, EvalCallback):
            continue

        callback.evaluations_timesteps = [int(t) for t in timesteps]
        callback.evaluations_results = [list(row) for row in results]
        callback.evaluations_length = [list(row) for row in ep_lengths]
        if "successes" in data.files:
            callback.evaluations_successes = [list(row) for row in data["successes"]]

        if callback.evaluations_results:
            means = [float(np.mean(row)) for row in callback.evaluations_results]
            callback.best_mean_reward = max(means)
            print(
                f"Loaded {len(callback.evaluations_timesteps)} prior evals from {npz_path} "
                f"(best mean so far {callback.best_mean_reward:.1f})"
            )
        break


def run_zoo_train_resume(args: Sequence[str], checkpoint: Path) -> None:
    """Continue training in the checkpoint's experiment folder (unit1-style).

    Unlike Zoo's default ``-i`` path (new ``…_N`` folder, timestep counter
    reset), this:

    - writes back into the same run directory as ``checkpoint``
    - keeps ``model.num_timesteps`` going (``reset_num_timesteps=False``)
    - opens a fresh TensorBoard subfolder whose x-axis continues that count
    - appends to existing ``*.monitor.csv`` and ``evaluations.npz`` history
    """
    from rl_zoo3 import exp_manager as em_mod
    from rl_zoo3.exp_manager import AsyncEval
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.utils import configure_logger

    resume_dir = str(checkpoint.resolve().parent)
    original_init = em_mod.ExperimentManager.__init__
    original_learn = em_mod.ExperimentManager.learn
    original_create_callbacks = em_mod.ExperimentManager.create_callbacks

    def patched_init(self, *a, **kw):  # type: ignore[no-untyped-def]
        original_init(self, *a, **kw)
        if self.continue_training:
            self.save_path = resume_dir
            self.params_path = os.path.join(self.save_path, self.env_name)
            # Keep prior episode rows in 0.monitor.csv, 1.monitor.csv, …
            self.monitor_kwargs = {**self.monitor_kwargs, "override_existing": False}
            print(f"Resuming in-place at {self.save_path}")
            print("Monitor CSVs will append (not overwrite)")

    def patched_create_callbacks(self) -> None:
        original_create_callbacks(self)
        if self.continue_training:
            _hydrate_eval_callback(self.callbacks, self.save_path)

    def patched_learn(self, model: BaseAlgorithm) -> None:
        # Same idea as unit1 start_new_log_run: new TB folder, same step clock.
        if self.tensorboard_log:
            logger = configure_logger(
                self.verbose,
                self.tensorboard_log,
                tb_log_name=self.algo.upper(),
                reset_num_timesteps=True,
            )
            model.set_logger(logger)
            print(f"Logging this resume to {logger.dir}")

        print(
            f"Already trained for {model.num_timesteps:,} timesteps; "
            f"training {self.n_timesteps:,} more."
        )

        kwargs: dict = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}
        elif self.log_interval < -1:
            kwargs = {"log_interval": None}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        if self.algo == "ars" and self.n_envs > 1:
            kwargs["async_eval"] = AsyncEval(
                [lambda: self.create_envs(n_envs=1, no_log=True) for _ in range(self.n_envs)],
                model.policy,
            )

        kwargs["reset_num_timesteps"] = False

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            pass
        finally:
            if len(self.callbacks) > 0:
                self.callbacks[0].on_training_end()
            try:
                assert model.env is not None
                model.env.close()
            except EOFError:
                pass

    em_mod.ExperimentManager.__init__ = patched_init  # type: ignore[method-assign]
    em_mod.ExperimentManager.create_callbacks = patched_create_callbacks  # type: ignore[method-assign]
    em_mod.ExperimentManager.learn = patched_learn  # type: ignore[method-assign]
    try:
        run_zoo_train_inprocess(args)
    finally:
        em_mod.ExperimentManager.__init__ = original_init  # type: ignore[method-assign]
        em_mod.ExperimentManager.create_callbacks = original_create_callbacks  # type: ignore[method-assign]
        em_mod.ExperimentManager.learn = original_learn  # type: ignore[method-assign]
