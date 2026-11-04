# Unit 4: Policy Gradient (REINFORCE) with CartPole and Pixelcopter

Script-based version of the [Hugging Face Deep RL Course Unit 4
notebook](https://huggingface.co/deep-rl-course/unit4/introduction): code
**REINFORCE** from scratch in PyTorch, train it on CartPole-v1 and/or
Pixelcopter, evaluate it, record a replay video, and optionally share it on
the Hugging Face Hub.

The original Colab walkthrough is kept as `unit4.ipynb`. This folder follows
the same modular layout as `unit2_q_learning/`.

## The task

- [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/):
  keep a pole balanced on a moving cart. 4 observation values, 2 actions.
  Capped at 500 steps, so 500 is a perfect score.
- [Pixelcopter](https://pygame-learning-environment.readthedocs.io/en/latest/user/games/pixelcopter.html):
  fly a copter through gaps in a scrolling cave. 7 observation values, 2
  actions (flap / do nothing). Much harder, with a deeper net and ~50k episodes.
- **Pixelcopter from pixels** (`--env pixelcopter-pixels`, the course's
  suggested follow-up): the same game, but the agent sees the screen instead
  of the 7 numbers. Each observation is the last 4 frames, 48x48 grayscale
  (shape `(4, 48, 48)`), so it can see motion. A CNN policy has to learn to
  find the copter and the walls itself.

**REINFORCE** is a Monte Carlo policy-gradient method. Unlike Q-Learning and
DQN, it learns the policy directly (a network that outputs action
probabilities) instead of a value function. After each full episode it
raises the log-probability of every action taken in proportion to the
(discounted, standardized) return that followed it:
`loss = -sum_t log pi(a_t|s_t) * G_t`.

## Setup

```bash
pip install -r requirements.txt
```

CartPole needs only Gymnasium. Pixelcopter also needs the PyGame Learning
Environment (`ple`), installed from GitHub by the last line of
`requirements.txt`.

Run commands from `unit4_reinforce/`.

## Files

| File              | Purpose                                                         |
|-------------------|-----------------------------------------------------------------|
| `config.py`       | All settings: env ids, per-env hyperparameters, Hub username, device |
| `envs.py`         | `make_env`, plus a Gymnasium port of Pixelcopter registered as `Pixelcopter-PLE-v0` (state) and `Pixelcopter-PLE-pixels-v0` (frames) |
| `policy.py`       | The policy networks (MLP with 1 or 2 hidden layers, or a CNN for frames) and their `act()` sampler |
| `reinforce.py`    | The algorithm itself: discounted returns, training loop, eval loop |
| `train.py`        | Trains a policy and saves it, with TensorBoard logs, under `runs/` |
| `evaluate.py`     | Loads a saved model and reports mean/std reward                  |
| `record_video.py` | Plays one episode and saves it as an `.mp4` replay               |
| `visualize.py`    | Video of what the CNN sees: input frames, every conv feature map, action probabilities |
| `hub.py`          | Push a model to the Hugging Face Hub, or pull one down           |
| `main.py`         | CLI that wires the above together                                |
| `unit4.ipynb`     | Original course notebook                                         |

## Usage

```bash
# Train CartPole (the default) or Pixelcopter
python main.py train
python main.py train --env pixelcopter
python main.py train --env pixelcopter-pixels   # CNN on screen frames
python main.py visualize   # what the newest pixels model sees -> videos/<run>-network.mp4
python main.py visualize --untrained   # same view with random filters, to compare

# Check how good the newest run is (or pass a specific runs/.../model.pt)
python main.py evaluate
python main.py evaluate runs/20261103_115100_cartpole/model.pt

# Save a replay video to ./videos/
python main.py record

# Share it on the Hugging Face Hub (requires `huggingface-cli login` first).
# repo_id defaults to <USERNAME>/Reinforce-cartpole or Reinforce-pixelcopter.
python main.py push runs/20261103_115100_cartpole/model.pt
python main.py push runs/20261103_115100_cartpole/model.pt njdonato/Reinforce-cartpole

# Try someone else's model from the Hub
python main.py pull <username>/Reinforce-cartpole
```

Each module also works standalone, e.g. `python train.py --env pixelcopter`
or `python evaluate.py`.

## What changed from the notebook

- Dropped Colab-only bits (`apt`, `xvfb` / `pyvirtualdisplay`, `notebook_login`,
  the `requirements-unit4.txt` / `sed` install cells, the NumPy `bool8` patch).
- Moved from legacy `gym` to **Gymnasium**, like units 2 and 3: `reset()`
  returns `(obs, info)` and `step()` returns `terminated, truncated` separately.
- Replaced `gym_pygame` (legacy-gym only) with a small Gymnasium wrapper in
  `envs.py` around the same PLE game, observation, and actions, registered
  under the same `Pixelcopter-PLE-v0` id. It also caps episodes at
  `max_t` steps, so a good agent can't make the replay video run forever.
- Merged the notebook's three `Policy` classes (the deliberately broken
  argmax one, CartPole's, and Pixelcopter's) into one class with
  `n_hidden_layers=1|2`. Layer names are unchanged, so course models still load.
- `reinforce()`, `evaluate_agent()`, and `push_to_hub()` take the environment
  as an argument. In the notebook they read the global `env` / `env_id`, so
  pushing Pixelcopter depended on which cell ran last.
- Training saves locally under `runs/<timestamp>_<env>/` (state dict +
  hyperparameters + `run.txt`), logs to TensorBoard, and handles Ctrl+C by
  saving the partly trained policy. The Hub upload still uses the course's
  format (`model.pt` as a whole pickled module), so the leaderboard can
  score it.
- Defaults to the CPU. The net is tiny and sees one state at a time, so GPU
  transfers cost more than they save. Set `DEVICE` in `config.py` to change it.

## Notes

- CartPole reaches 500 within the default 1000 episodes most runs, but
  REINFORCE is high-variance, so some seeds plateau or collapse. Retrain if
  one does.
- Pixelcopter at 50k episodes takes hours. Watch `train/avg100_reward` in
  TensorBoard and Ctrl+C once it levels off. The run is still saved.
- Evaluation samples from the policy (as the course does) rather than
  taking the argmax, so scores vary a little between runs.
- The pixel version uses the state version's gamma, learning rate, and
  episode budget, so the two TensorBoard runs compare directly. Expect it to
  learn much more slowly, if at all: REINFORCE's noisy updates must now also
  train the conv layers to see. Doing this well is what later units' methods
  (A2C, PPO) are for.
- `pull` unpickles `model.pt`, which runs code, so only pull repos you trust.
