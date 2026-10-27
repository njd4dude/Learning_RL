# Unit 3: Deep Q-Learning on Space Invaders

Script-based version of the [Hugging Face Deep RL Course Unit 3
notebook](https://huggingface.co/deep-rl-course/unit3/introduction): train a
**DQN** agent on Atari Space Invaders with [RL Baselines3
Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), evaluate it, record replay
videos locally, and optionally push/pull models on the Hugging Face Hub.

The original Colab walkthrough is kept as `unit3.ipynb`. This folder follows the
same modular layout as `unit2_q_learning/`.

## Setup

From the repo root (with your venv active):

```bash
pip install -r unit3_space_invaders/requirements.txt
# or, as the course does:
# pip install git+https://github.com/DLR-RM/rl-baselines3-zoo
```

Run commands from `unit3_space_invaders/` so `dqn.yml` and `logs/` resolve.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Defaults: algo, env ids, log/video folders |
| `dqn.yml` | RL Zoo hyperparameter config (course defaults) |
| `zoo.py` | Helper that runs `python -u -m rl_zoo3.*` |
| `train.py` | Train via `rl_zoo3.train` |
| `evaluate.py` | Score via `rl_zoo3.enjoy --no-render` |
| `record_video.py` | Save local `.mp4` replays (no Hub needed) |
| `hub.py` | Push to / pull from the Hugging Face Hub |
| `main.py` | CLI wiring the above together |
| `unit3.ipynb` | Original course notebook |

## Usage

```bash
# Train (reads hyperparams from dqn.yml; override steps with -n)
# New runs go under logs_active/ so TensorBoard stays clean; older runs stay
# in logs/. Events land in <logdir>/dqn/<env>_N/tb/; TB auto-starts watching
# the train -f folder (next free port if :6006 is already taken).
# dqn.yml sets n_envs; train defaults to --vec-env subproc
python main.py train
python main.py train -n 100000
# python main.py train -f logs     # write into the old archive tree instead
# tensorboard --logdir logs_active/

# Score / record: defaults still look in logs/ (archive). For new runs:
python main.py evaluate -f logs_active
python main.py record -f logs_active --episodes 2
python main.py evaluate --exp-id 1
python main.py record --exp-id 1 --episodes 2

# Hub (requires `huggingface-cli login` first)
python main.py push --orga YOUR_HF_USER --repo-name dqn-SpaceInvadersNoFrameskip-v4
python main.py pull --env BeamRiderNoFrameskip-v4 --orga sb3 -f rl_trained/
```

Each module also works standalone, e.g. `python train.py -n 100000` or
`python record_video.py --exp-id 1`.

## What changed from the notebook

- Dropped Colab-only bits (`apt`, `xvfb` / `pyvirtualdisplay`, `notebook_login`).
- Replaced `!python -m rl_zoo3...` cells with importable modules + a `main.py` CLI.
- Added local `record_video.py` so you can watch `.mp4` replays without publishing.
- Kept `dqn.yml` as the single source of training hyperparameters (Zoo expects it).

## Notes

- Training for the full `1e6` timesteps in `dqn.yml` is long on CPU; use `-n` to
  smoke-test, or a GPU for the full run.
- `enjoy` / evaluate print scores; they do **not** write video files. Use
  `record` for mp4s.
- If the latest experiment folder has no final `*.zip`, pass `--exp-id 1` or
  `--load-best` (record falls back to `best_model.zip` automatically).
- **Ctrl+C** during `train` is supported: RL Zoo catches the interrupt and still
  saves the current model into the experiment folder (training runs in-process
  for that reason).
- Each experiment folder under `logs/dqn/<env>_N/` holds monitors, zips, config,
  **and** TensorBoard events in a `tb/` subfolder. Point TensorBoard at `logs/`
  so run names match those folders (e.g. `dqn/SpaceInvaders…_40/tb`).
