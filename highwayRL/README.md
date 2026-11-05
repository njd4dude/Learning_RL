# highwayRL: driving on highway-v0

Experiments with [highway-env](https://highway-env.farama.org/)'s
[`highway-v0`](https://highway-env.farama.org/environments/highway/): drive
along a busy multi-lane road, going fast and avoiding crashes.

## The task

- **Observation:** a 5x5 array. One row per vehicle (you first, then the 4
  nearest), with columns `presence, x, y, vx, vy`. Other cars are relative to
  yours, and cars behind you are mostly left out (`see_behind` is off).
- **Actions:** 5 high-level choices: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`,
  `FASTER`, `SLOWER`.
- **Episode:** ends on a crash, or after 40 decisions (one per simulated second).
- **Reward:** mostly for driving fast, plus a bonus for keeping right, with a
  penalty for crashing.

## Setup

This folder has its own venv, separate from the one at the repo root:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run commands from `highwayRL/`.

## Usage

```bash
python try_env.py           # random actions, saves a video to runs/baselines/
python try_env.py --watch   # same, in a window

# train DQN, or PPO with --algo ppo
python train.py --name penalty10 --hypothesis "Fewer crashes" --change "Crash penalty -10"
# continue a saved model for 60,000 more steps, in a new run folder
python train.py --resume runs/<run>/model.zip --timesteps 60000 --name ... --hypothesis ... --change ...
python evaluate.py          # reward and crash rate of the newest run
python record_video.py      # save replays into the newest run's folder
python record_video.py --watch   # watch the newest run live in a window

tensorboard --logdir runs   # watch training
```

## Experiments

Every run tests one idea. `train.py` requires a name, hypothesis and change,
and adds an entry to [EXPERIMENTS.md](EXPERIMENTS.md); fill in its conclusion
after looking at the results. That file is tracked by git, while `runs/` isn't.

Everything from a run stays in its folder:

```
runs/20261104_215246_dqn_penalty5/
    run.txt             hypothesis, change, settings, final eval
    eval.txt            every evaluate.py score
    videos/             replays
    best_model.zip      top mid-training eval
    model.zip           the last step
    train_output.txt    console log, when saved
    tb/, checkpoints/
```

`runs/baselines/` holds the random-agent video and fixed-action scores.

## Files

| File              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `config.py`       | All settings: env id, DQN and PPO hyperparameters, eval and video |
| `EXPERIMENTS.md`  | Hypothesis, change, result and conclusion of every run   |
| `train.py`        | Trains DQN or PPO, keeping the best-scoring model under `runs/` |
| `evaluate.py`     | Scores a run: reward, steps survived, crash rate         |
| `record_video.py` | Saves `.mp4` replays of a run                            |
| `try_env.py`      | Plays random actions, to check the install               |

## Notes

- The simulator is slow, so training runs 12 envs in parallel. PPO manages about
  44 steps a second with one math thread per process, against 38 by default, and
  16 envs reached 48. `config.py` sets the one-thread limit.
- DQN can get worse late in training, so each run keeps `best_model.zip`, the
  top mid-training eval, beside `model.zip`, the last step. `evaluate.py` and
  `record_video.py` use the best one. The eval that picks it is only 24
  episodes, so its score runs a little high: trust the final eval in `run.txt`.
- `crash_rate` counts episodes that ended before step 40.
- Results and the current ranking are in [EXPERIMENTS.md](EXPERIMENTS.md).
- Fixed-action baselines over 200 episodes, with the default reward: always `SLOWER` scores 28.09 with
  0% crashes, always `IDLE` 16.06 with 94%, always `FASTER` 8.86 with 100%. A
  model has to beat 28.09 to be driving better than crawling along.
