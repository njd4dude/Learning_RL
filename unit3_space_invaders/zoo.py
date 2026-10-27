"""Thin helpers for calling RL Baselines3 Zoo.

Zoo's entrypoints (`rl_zoo3.train`, `.enjoy`, …) parse `sys.argv` themselves.
- `run_zoo` / `run_zoo_checked`: subprocess (`python -u -m …`) — good for
  enjoy/hub; live logs on Windows.
- `run_zoo_train_inprocess`: import and call `rl_zoo3.train.train()` in this
  process so Ctrl+C is caught by Zoo (it saves the model after interrupt).
- `run_zoo_train_resume`: load weights into a *new* run folder, but keep the
  timestep counter / exploration schedule going (unlike Zoo's default `-i`).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


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


def run_zoo_train_resume(
    args: Sequence[str],
    checkpoint: Path,
    *,
    resume_warmup: int = 0,
) -> None:
    """Continue from checkpoint weights in a fresh Zoo experiment folder.

    Zoo's plain ``-i`` already creates a new ``…_N`` directory and loads the
    zip. We only patch ``learn`` so ``reset_num_timesteps=False``: the step
    counter and exploration schedule keep going (ε stays low after a long
    run) instead of resetting as if training started from scratch.

    ``resume_warmup`` pushes ``learning_starts`` forward so an empty replay
    buffer can refill before gradient updates (zips usually omit the buffer).
    """
    from rl_zoo3 import exp_manager as em_mod
    from rl_zoo3.exp_manager import AsyncEval
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.utils import configure_logger

    original_learn = em_mod.ExperimentManager.learn

    def patched_learn(self, model: BaseAlgorithm) -> None:
        # New TB subfolder for this experiment; x-axis continues prior steps.
        if self.tensorboard_log:
            logger = configure_logger(
                self.verbose,
                self.tensorboard_log,
                tb_log_name=self.algo.upper(),
                reset_num_timesteps=True,
            )
            model.set_logger(logger)
            print(f"Logging this run to {logger.dir}")

        print(f"Loaded weights from {checkpoint}")
        print(f"New experiment folder: {self.save_path}")
        print(
            f"Already trained for {model.num_timesteps:,} timesteps; "
            f"training {self.n_timesteps:,} more."
        )

        if resume_warmup > 0 and hasattr(model, "learning_starts"):
            model.learning_starts = int(model.num_timesteps) + int(resume_warmup)
            print(
                f"Resume warm-up: collecting {resume_warmup:,} steps before "
                f"updates (learning_starts={model.learning_starts:,})."
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
            print("\nTraining interrupted — will save current progress...")
        finally:
            # Don't let progress-bar / SubprocVecEnv teardown eat a second Ctrl+C
            # before Zoo's save_trained_model() runs.
            try:
                if len(self.callbacks) > 0:
                    self.callbacks[0].on_training_end()
            except KeyboardInterrupt:
                pass
            try:
                assert model.env is not None
                model.env.close()
            except (EOFError, KeyboardInterrupt):
                pass

    em_mod.ExperimentManager.learn = patched_learn  # type: ignore[method-assign]
    try:
        run_zoo_train_inprocess(args)
    finally:
        em_mod.ExperimentManager.learn = original_learn  # type: ignore[method-assign]
