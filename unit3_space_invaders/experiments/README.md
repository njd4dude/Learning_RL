# Hyperparameter experiments (learning rate, buffer, batch)

DQN runs (1M steps each) that change **one** knob at a time so you can see the effect in TensorBoard under `logs_active/`.

## What each knob does

| Hyperparam | Intuition |
|---|---|
| `learning_rate` | How big each network update is. Too high → unstable / noisy scores. Too low → learns very slowly. |
| `buffer_size` | How many past transitions are kept. Too small → forgets / overfits recent play. Larger → more diverse minibatches (more RAM). |
| `batch_size` | How many transitions per update. Larger → stabler gradients, but each update is heavier and can look “smoother but slower” early on. |

## Important

These configs set `learning_starts: 20_000` so learning actually starts during a short run. The main `dqn.yml` uses `100_000`, which means a 100k-step run would **never** update the network.

## Budget

Each config trains 1M steps with `gradient_steps: 4` (n_envs 4 × train_freq 4 = 16 transitions per rollout, 4 updates → replay ratio 8, like classic DQN). Earlier 300k-step / `gradient_steps: 1` runs got only ~18k updates and every config looked the same.

`run_suite.py` runs every config for seeds 0 and 1 (10 runs, roughly 4–8 hours total) and evaluates 10 episodes every 50k steps. Use `--seeds 0` for a single pass.

## Run them

From `unit3_space_invaders/` (venv on):

```powershell
# one at a time (compare in TensorBoard)
python main.py train -c experiments/baseline.yml --seed 0 --no-save-replay-buffer
python main.py train -c experiments/lr_high.yml --seed 0 --no-save-replay-buffer
python main.py train -c experiments/lr_low.yml --seed 0 --no-save-replay-buffer
python main.py train -c experiments/buffer_small.yml --seed 0 --no-save-replay-buffer
python main.py train -c experiments/batch_large.yml --seed 0 --no-save-replay-buffer

# or run the whole suite sequentially
python experiments/run_suite.py
```

Open the TensorBoard URL printed at train start (watches `logs_active/`). Compare:

- `rollout/ep_rew_mean` — game score (main signal)
- `train/loss` — update noise / instability
- `rollout/exploration_rate` — should match across runs (same schedule)

Same `--seed 0` keeps the env randomness comparable; runs still won’t be identical because of GPU nondeterminism, but trends are readable.

## After a run

```powershell
python main.py evaluate -f logs_active --exp-id N
python main.py record -f logs_active --exp-id N --episodes 2
```
