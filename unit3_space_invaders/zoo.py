"""Thin helpers for calling RL Baselines3 Zoo.

Zoo's entrypoints (`rl_zoo3.train`, `.enjoy`, …) parse `sys.argv` themselves.
- `run_zoo` / `run_zoo_checked`: subprocess (`python -u -m …`) — good for
  enjoy/hub; live logs on Windows.
- `run_zoo_train_inprocess`: import and call `rl_zoo3.train.train()` in this
  process so Ctrl+C is caught by Zoo (it saves the model after interrupt).
- `run_zoo_train_resume`: load weights into a *new* run folder, but keep the
  timestep counter / exploration schedule going (unlike Zoo's default `-i`).

TensorBoard: Zoo+SB3 normally write `tb/<env>/DQN_N` with a counter independent
of `logs/<algo>/<env>_N`. We redirect events into each run's own ``tb/``
subfolder so curves and configs share one directory.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


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


def tb_dir_for_run(save_path: str | Path) -> Path:
    """Event files for one Zoo experiment live here (next to monitors / zips)."""
    return Path(save_path) / "tb"


def attach_run_tensorboard(model, save_path: str | Path, *, verbose: int = 1) -> Path:
    """Point SB3's logger at ``<save_path>/tb`` (no separate DQN_N counter)."""
    from stable_baselines3.common.logger import configure

    tb_dir = tb_dir_for_run(save_path)
    tb_dir.mkdir(parents=True, exist_ok=True)
    formats = ["stdout", "tensorboard"] if verbose >= 1 else ["tensorboard"]
    model.set_logger(configure(str(tb_dir), format_strings=formats))
    print(f"TensorBoard events: {tb_dir}")
    return tb_dir


def fit_replay_buffer(model) -> int:
    """Make a reloaded replay buffer match the model's ``n_envs`` / ``buffer_size``.

    SB3 stores transitions as ``(buffer_size // n_envs, n_envs, …)``, so a
    buffer saved with 16 envs breaks ``add()`` once the yaml says 12. Plain
    (non memory-optimized) DQN transitions are independent, so we flatten them
    oldest → newest and re-pack the most recent ones into a fresh buffer.

    Returns the number of transitions available after fitting (0 = unusable).
    """
    old = model.replay_buffer
    rows = old.buffer_size if old.full else old.pos
    if old.n_envs == model.n_envs and old.buffer_size == max(
        model.buffer_size // model.n_envs, 1
    ):
        return rows * old.n_envs

    new = type(old)(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=old.optimize_memory_usage,
        handle_timeout_termination=old.handle_timeout_termination,
    )
    if old.optimize_memory_usage:
        # next_obs is implied by the next row per env column; can't re-pack.
        print(
            f"Saved replay buffer has n_envs={old.n_envs} but model uses "
            f"n_envs={model.n_envs} (optimize_memory_usage); discarding it."
        )
        model.replay_buffer = new
        return 0
    import numpy as np

    # Chronological order: when full, the oldest row sits at ``pos``.
    start = old.pos if old.full else 0
    total = rows * old.n_envs
    keep_rows = min(total // new.n_envs, new.buffer_size)
    keep = keep_rows * new.n_envs

    fields = ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")
    chunk = 8192  # transitions per copy, keeps peak RAM low for image obs
    for a in range(0, keep, chunk):
        flat_idx = np.arange(total - keep + a, total - keep + min(a + chunk, keep))
        src_rows = (start + flat_idx // old.n_envs) % old.buffer_size
        src_cols = flat_idx % old.n_envs
        for name in fields:
            dst = getattr(new, name)
            dst.reshape(-1, *dst.shape[2:])[a : a + len(flat_idx)] = getattr(old, name)[
                src_rows, src_cols
            ]

    new.pos = keep_rows % new.buffer_size
    new.full = keep_rows == new.buffer_size
    model.replay_buffer = new
    print(
        f"Re-packed replay buffer n_envs {old.n_envs} -> {new.n_envs}: "
        f"kept {keep:,} of {total:,} transitions."
    )
    return keep


@contextmanager
def _colocated_tensorboard_patches() -> Iterator[None]:
    """Redirect Zoo TB writes into each experiment's ``save_path/tb``."""
    from rl_zoo3 import exp_manager as em_mod

    original_init = em_mod.ExperimentManager.__init__
    original_learn = em_mod.ExperimentManager.learn

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Zoo sets tensorboard_log to ``< -tb >/<env>``; SB3 would then create
        # ``…/DQN_N`` with its own counter. Keep TB enabled but pin the path to
        # this run folder; ``attach_run_tensorboard`` writes the actual events.
        if self.tensorboard_log is not None:
            self.tensorboard_log = self.save_path

    def patched_learn(self, model):
        if self.tensorboard_log:
            attach_run_tensorboard(model, self.save_path, verbose=self.verbose)
        return original_learn(self, model)

    em_mod.ExperimentManager.__init__ = patched_init  # type: ignore[method-assign]
    em_mod.ExperimentManager.learn = patched_learn  # type: ignore[method-assign]
    try:
        yield
    finally:
        em_mod.ExperimentManager.__init__ = original_init  # type: ignore[method-assign]
        em_mod.ExperimentManager.learn = original_learn  # type: ignore[method-assign]


def run_zoo_train_inprocess(args: Sequence[str]) -> None:
    """Run RL Zoo training in-process so KeyboardInterrupt still saves the model.

    Zoo's ExperimentManager.learn() catches Ctrl+C, then train() always calls
    save_trained_model(). That only works reliably if training isn't a child
    subprocess that may be killed before the save finishes.
    """
    from rl_zoo3.train import train as zoo_train

    argv = ["rl_zoo3.train", *args]
    print(">", " ".join([sys.executable, "-u", "-m", *argv]))
    old_argv = sys.argv
    sys.argv = argv
    try:
        with _colocated_tensorboard_patches():
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

    original_learn = em_mod.ExperimentManager.learn

    def patched_learn(self, model: BaseAlgorithm) -> None:
        # Co-locate TB under this new run folder (same as a fresh train).
        if self.tensorboard_log:
            attach_run_tensorboard(model, self.save_path, verbose=self.verbose)

        print(f"Loaded weights from {checkpoint}")
        print(f"New experiment folder: {self.save_path}")
        print(
            f"Already trained for {model.num_timesteps:,} timesteps; "
            f"training {self.n_timesteps:,} more."
        )

        # Zoo auto-loads ``replay_buffer.pkl`` sitting next to the checkpoint.
        buffered = 0
        if (checkpoint.parent / "replay_buffer.pkl").is_file() and hasattr(
            model, "replay_buffer"
        ):
            buffered = fit_replay_buffer(model)
            if buffered:
                print(f"Resumed with {buffered:,} transitions in the replay buffer.")
        else:
            print(f"No replay_buffer.pkl next to {checkpoint.name}; buffer starts empty.")

        if not buffered and resume_warmup > 0 and hasattr(model, "learning_starts"):
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

    # Init patch only (learn is the resume-aware version below).
    original_init = em_mod.ExperimentManager.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.tensorboard_log is not None:
            self.tensorboard_log = self.save_path

    em_mod.ExperimentManager.__init__ = patched_init  # type: ignore[method-assign]
    em_mod.ExperimentManager.learn = patched_learn  # type: ignore[method-assign]
    try:
        # Avoid stacking the default learn patch from run_zoo_train_inprocess.
        from rl_zoo3.train import train as zoo_train

        argv = ["rl_zoo3.train", *args]
        print(">", " ".join([sys.executable, "-u", "-m", *argv]))
        old_argv = sys.argv
        sys.argv = argv
        try:
            zoo_train()
        finally:
            sys.argv = old_argv
    finally:
        em_mod.ExperimentManager.__init__ = original_init  # type: ignore[method-assign]
        em_mod.ExperimentManager.learn = original_learn  # type: ignore[method-assign]
