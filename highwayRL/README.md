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
python try_env.py           # random actions, saves videos/random.mp4
python try_env.py --watch   # same, in a window

python train.py             # train DQN, writing a folder under runs/
python train.py --algo ppo  # train PPO instead
python evaluate.py          # reward and crash rate of the newest run
python record_video.py      # save replays of the newest run to videos/
python record_video.py --watch   # watch the newest run live in a window

tensorboard --logdir runs   # watch training
```

## Files

| File              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `config.py`       | All settings: env id, DQN and PPO hyperparameters, eval and video |
| `train.py`        | Trains DQN or PPO, keeping the best-scoring model under `runs/` |
| `evaluate.py`     | Scores a run: reward, steps survived, crash rate         |
| `record_video.py` | Saves `.mp4` replays of a run                            |
| `try_env.py`      | Plays random actions, to check the install               |

## Notes

- The simulator manages only about 6 steps a second per env, so training runs
  12 envs in parallel. 20,000 steps took about 17 minutes.
- DQN can get worse late in training, so each run keeps `best_model.zip`, the
  top mid-training eval, beside `model.zip`, the last step. `evaluate.py` and
  `record_video.py` use the best one. The eval that picks it is only 24
  episodes, so its score runs a little high: trust the final eval in `run.txt`.
- `crash_rate` counts episodes that ended before step 40.
- Best DQN so far: 20% crashes and a mean reward of 28.73 over 200 episodes, at
  40k steps. Slower DQN settings did worse, at 51% crashes.
- Fixed-action baselines over 200 episodes: always `SLOWER` scores 28.09 with
  0% crashes, always `IDLE` 16.06 with 94%, always `FASTER` 8.86 with 100%. A
  model has to beat 28.09 to be driving better than crawling along.
