# Unit 1: LunarLander with PPO

A simplified, script-based version of the [Hugging Face Deep RL Course Unit 1
notebook](https://huggingface.co/deep-rl-course/unit1/introduction): train a
PPO agent to land the LunarLander, evaluate it, record a replay video, and
optionally share it on the Hugging Face Hub.

## The task

[LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
gives the agent an 8-value observation (position, velocity, angle, angular
velocity, and leg contact flags) and 4 discrete actions (do nothing, fire
left/main/right engine). It scores +100/-100 for landing/crashing, plus
shaped rewards for approaching the pad smoothly; a score of 200+ counts as
"solved".

We solve it with **PPO** (Proximal Policy Optimization) from
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/), a
mix of value-based learning (estimating how good a state/action is) and
policy-based learning (directly learning a distribution over actions).

## Setup

```bash
pip install -r requirements.txt
```

Box2D environments (used by LunarLander) need `swig` and `cmake` installed on
your system first. On Windows, the easiest path is:

```bash
pip install swig
```

If you're on a headless Linux machine and want to record videos, see the
note at the top of `record_video.py` about setting up a virtual display.

## Files

| File              | Purpose                                                   |
|-------------------|------------------------------------------------------------|
| `config.py`       | All settings: env id, hyperparameters, model/repo names    |
| `train.py`        | Builds the vectorized env + PPO model, trains, saves it    |
| `evaluate.py`     | Loads a saved model and reports mean/std reward            |
| `record_video.py` | Plays episodes with a saved model and saves `.mp4` replays |
| `hub.py`          | Push a model to the Hugging Face Hub, or pull one down     |
| `main.py`         | CLI that wires the above together                          |

## Usage

```bash
# Train a new agent (defaults to 500k timesteps, ~15-20 min on CPU/GPU)
python main.py train

# Or keep training the last saved zip for another 500k timesteps
python main.py train --load

# Check how good it is
python main.py evaluate

# Save a few replay videos to ./videos/
python main.py record

# Share it on the Hugging Face Hub (requires `huggingface-cli login` first,
# and REPO_ID set in config.py)
python main.py push

# Try someone else's model from the Hub
python main.py pull Classroom-workshop/assignment2-omar ppo-LunarLander-v2.zip
```

Each module also works standalone, e.g. `python train.py` or
`python evaluate.py my_model.zip`.

## What changed from the original notebook

The original `unit1.py` was a Google Colab export: one long script mixing
markdown commentary, `!pip install` / `!apt install` shell lines, a
Colab-only virtual-display + forced-runtime-restart hack, and a couple of
unfinished `# TODO` exercises. This version:

- Strips Colab-only setup code (the forced restart, `!apt`/`!pip` lines —
  see `requirements.txt` instead) and leaves a note on virtual displays where
  it's actually needed (`record_video.py`).
- Fills in the exercise TODOs with the course's reference solution.
- Splits training / evaluation / video / Hub-upload concerns into their own
  files so each piece can be run, read, or modified independently.
- Reads hyperparameters and names from one `config.py` instead of having
  them hardcoded in several places.
