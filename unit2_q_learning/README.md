# Unit 2: Q-Learning with FrozenLake and Taxi

A simplified, script-based version of the [Hugging Face Deep RL Course Unit 2
notebook](https://huggingface.co/deep-rl-course/unit2/introduction): train a
tabular Q-Learning agent on FrozenLake and/or Taxi-v3, evaluate it, record a
replay video, and optionally share it on the Hugging Face Hub.

## The task

Both environments are small, discrete-state, discrete-action worlds, so one
`(state_space, action_space)` Q-table works for either:

- [FrozenLake-v1](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
  (4x4, non-slippery): navigate from start to goal without falling in a hole.
  16 states, 4 actions.
- [Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/): pick up
  a passenger and drop them at their destination. 500 states, 6 actions.
  (The course notebook uses `Taxi-v3`, but that id is deprecated as of
  `gymnasium>=1.1` in favor of `Taxi-v4`, which behaves the same way.)

**Q-Learning** is off-policy: it acts with an epsilon-greedy policy
(explore vs. exploit) but updates the Q-table using the greedy policy's
value, via the Bellman equation
`Q(s,a) += lr * (reward + gamma * max(Q(s')) - Q(s,a))`.

## Setup

```bash
pip install -r requirements.txt
```

## Files

| File             | Purpose                                                        |
|------------------|------------------------------------------------------------------|
| `config.py`      | All settings: env ids/kwargs, hyperparameters, per-env model names |
| `q_learning.py`  | The algorithm itself: Q-table, policies, train loop, eval loop  |
| `train.py`       | Builds an env, trains a Q-table, pickles it with its hyperparameters |
| `evaluate.py`    | Loads a pickled model and reports mean/std reward               |
| `record_video.py`| Plays one greedy episode and saves it as an `.mp4` replay        |
| `hub.py`         | Push a model to the Hugging Face Hub, or pull one down           |
| `main.py`        | CLI that wires the above together                                |

## Usage

```bash
# Train FrozenLake (the default) or Taxi
python main.py train
python main.py train --env taxi

# Check how good it is
python main.py evaluate q-FrozenLake-v1-4x4-noSlippery.pkl
python main.py evaluate q-taxi.pkl

# Save a replay video to ./videos/
python main.py record q-taxi.pkl

# Share it on the Hugging Face Hub (requires `huggingface-cli login` first)
python main.py push q-taxi.pkl <your-username>/q-taxi

# Try someone else's model from the Hub
python main.py pull <username>/q-FrozenLake-v1-4x4-noSlippery q-learning.pkl
```

Each module also works standalone, e.g. `python train.py --env taxi` or
`python evaluate.py q-taxi.pkl`.

## What changed from the original notebook

The original `copy_of_unit2.py` was a Google Colab export: markdown
commentary, `!pip install` / `!apt install` shell lines, a Colab-only virtual
display + forced-runtime-restart hack, unfinished `# TODO` exercises, and a
stray indented `notebook_login()` call left over from a copy/paste. This
version:

- Strips the Colab-only setup code and notebook login cell (see
  `requirements.txt`, and log in once with `huggingface-cli login` instead).
- Fills in the exercise TODOs with the course's reference solution, and fixes
  the indentation bug around the second `notebook_login()` call.
- Pulls the Q-Learning algorithm (table, policies, train/eval loops) out into
  `q_learning.py`, shared by both environments instead of being copy-pasted
  with FrozenLake- and Taxi-specific globals.
- Splits training / evaluation / video / Hub-upload concerns into their own
  files, each runnable and configurable independently.
- Reads hyperparameters and names from one `config.py`, keyed per
  environment, instead of reassigning the same global variables twice.
