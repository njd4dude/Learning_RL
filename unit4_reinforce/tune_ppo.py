"""Search PPO hyperparameters for the 7-number Pixelcopter with Optuna.

Usage:
    python tune_ppo.py                       # settings from config.py
    python tune_ppo.py --trials 40           # rerun to continue the same search

Each trial trains a new agent, is scored on sampled actions (as all the other
Pixelcopter scores are), and keeps its model and TensorBoard logs in its own
trial_NNN folder. Weak trials stop early. Progress lives in study.db, so a
stopped search picks up where it left off when run again.
"""

import argparse
import json
from multiprocessing import Process
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import sync_envs_normalization

import config
import train_ppo

STUDY_NAME = "pixelcopter-ppo"


def _sample_hyperparameters(trial: optuna.Trial) -> tuple[dict, bool]:
    """One trial's PPO settings, and whether its entropy bonus fades to 0."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    if trial.suggest_categorical("decay_learning_rate", [True, False]):
        learning_rate = LinearSchedule(learning_rate, 0.0, 1.0)
    width = trial.suggest_categorical("net_width", [64, 128, 256])
    kwargs = dict(
        policy="MlpPolicy",
        learning_rate=learning_rate,
        n_steps=trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        n_epochs=trial.suggest_categorical("n_epochs", [4, 10, 20]),
        gamma=trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999]),
        gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 0.99]),
        clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.03, log=True),
        policy_kwargs=dict(net_arch=dict(pi=[width, width], vf=[width, width])),
    )
    return kwargs, trial.suggest_categorical("decay_ent_coef", [True, False])


class _TrialCallback(BaseCallback):
    """Fades the entropy bonus if asked, and stops trials falling behind the rest."""

    def __init__(self, trial: optuna.Trial, eval_env, decay_ent_coef: bool):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.decay_ent_coef = decay_ent_coef
        self.start_ent_coef = None
        self.next_report = config.TUNE_REPORT_STEPS
        self.pruned = False

    def _on_step(self) -> bool:
        if self.decay_ent_coef:
            if self.start_ent_coef is None:
                self.start_ent_coef = self.model.ent_coef
            self.model.ent_coef = self.start_ent_coef * self.model._current_progress_remaining

        if self.num_timesteps < self.next_report:
            return True
        self.next_report += config.TUNE_REPORT_STEPS
        sync_envs_normalization(self.training_env, self.eval_env)
        mean, _std = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=config.TUNE_REPORT_EPISODES, deterministic=False
        )
        self.logger.record("eval/mean_reward", mean)
        self.trial.report(mean, self.num_timesteps)
        self.pruned = self.trial.should_prune()
        return not self.pruned


def _objective(trial: optuna.Trial, folder: Path, timesteps: int) -> float:
    run_dir = folder / f"trial_{trial.number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    kwargs, decay_ent_coef = _sample_hyperparameters(trial)

    env = train_ppo._normalized_env(config.TUNE_N_ENVS, subprocess=True)
    eval_env = train_ppo._normalized_env(1, training=False)
    try:
        model = PPO(env=env, device="cpu", verbose=0, **kwargs)
        model.set_logger(configure(str(run_dir / "tb"), ["tensorboard"]))
        callback = _TrialCallback(trial, eval_env, decay_ent_coef)
        model.learn(total_timesteps=timesteps, callback=callback)
        if callback.pruned:
            raise optuna.TrialPruned()
        model.save(run_dir / "model.zip")
        env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()
        eval_env.close()

    mean, std = train_ppo.evaluate(run_dir, config.TUNE_EVAL_EPISODES)
    trial.set_user_attr("std", std)
    return mean


def _load_study(folder: Path) -> optuna.Study:
    folder.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{folder / 'study.db'}",
        direction="maximize",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=config.TUNE_PRUNE_AFTER_STEPS),
    )


def _worker(folder: Path, trials: int, timesteps: int) -> None:
    # Several searches share this machine, so each sticks to one torch thread.
    torch.set_num_threads(1)
    study = _load_study(folder)
    study.optimize(
        lambda trial: _objective(trial, folder, timesteps),
        callbacks=[MaxTrialsCallback(trials, states=(TrialState.COMPLETE, TrialState.PRUNED))],
    )


def _report(folder: Path) -> None:
    study = _load_study(folder)
    finished = [t for t in study.trials if t.state == TrialState.COMPLETE]
    n_pruned = sum(t.state == TrialState.PRUNED for t in study.trials)
    print(f"\n{len(finished)} trials finished, {n_pruned} stopped early.")
    for trial in sorted(finished, key=lambda t: t.value, reverse=True)[:5]:
        print(f"trial {trial.number:3d}: {trial.value:8.2f} +/- {trial.user_attrs['std']:.2f}  {trial.params}")
    if finished:
        best = study.best_trial
        summary = dict(trial=best.number, score=best.value, params=best.params)
        (folder / "best.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=config.TUNE_TRIALS, help="Total trials in the search.")
    parser.add_argument("--workers", type=int, default=config.TUNE_WORKERS, help="Trials to run at once.")
    parser.add_argument("--timesteps", type=int, default=config.TUNE_TIMESTEPS, help="Steps per trial.")
    parser.add_argument("--folder", default=str(Path(config.RUNS_FOLDER) / "pixelcopter-ppo-tune"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder = Path(args.folder)
    _load_study(folder)  # Creates study.db before the workers race to.
    workers = [Process(target=_worker, args=(folder, args.trials, args.timesteps)) for _ in range(args.workers)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    _report(folder)
