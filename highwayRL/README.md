# highwayRL: driving on highway-v0

Experiments with [highway-env](https://highway-env.farama.org/)'s
[`highway-v0`](https://highway-env.farama.org/environments/highway/): drive
along a busy multi-lane road, going fast and avoiding crashes.

## The task

- **Observation:** a 5x5 array. One row per vehicle (you first, then the 4
  nearest), with columns `presence, x, y, vx, vy`.
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
```
