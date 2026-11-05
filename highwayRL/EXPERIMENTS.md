# Experiments

One entry per run, oldest first. `train.py` adds each new entry; write its
conclusion once you've looked at the results.

Each run's folder under `runs/` has its videos, models and eval scores.

## Ranking

Sorted by distance per episode. Distance, speed and crash rate don't depend on
the reward, so they compare across every run. Own reward is the score on the
reward that run trained with, so those only compare between runs with the same
reward. All are `evaluate.py --own-reward` over 200 episodes, on `best_model.zip`
(the smoke test has only `model.zip`).

Always `SLOWER` crawls at 20 m/s for about 800 m. The maximum is 1,200 m: 30 m/s
for 40 steps. Crash rates are accurate to about ±6 to ±7 points at the higher
rates, and two scores of the same model can differ by more than that. Each result
is one training run, with no repeat seeds.

| Run | Trained with | Distance | Crash rate | Mean speed | Own reward |
|---|---|---|---|---|---|
| `20261104_210951_dqn_penalty2` | No normalizing, -2 crash | 921 m | 46% | 29.2 m/s | 12.84 |
| `20261104_215246_dqn_penalty5` | No normalizing, -5 crash | 916 m | 40% | 27.9 m/s | 10.74 |
| `20261104_171828_dqn_40k` | Default reward | 864 m | 17% | 23.9 m/s | 29.59 |
| `20261104_190015_ppo_200k` | Default reward | 825 m | 22% | 24.1 m/s | 28.26 |
| `20261104_161334_dqn_20k` | Default reward | 824 m | 20% | 23.2 m/s | 28.31 |
| `20261105_113120_ppo_40k-lr3e-4` | No normalizing, -5 crash | 819 m | 8% | 21.4 m/s | 4.55 |
| `20261105_102757_ppo_20k-lr1e-4` | No normalizing, -5 crash | 798 m | 2% | 20.1 m/s | 2.26 |
| `20261105_105138_ppo_20k-lr1e-4-default-reward` | Default reward | 794 m | 1% | 20.0 m/s | 27.79 |
| `20261105_111332_ppo_20k-lr3e-4` | No normalizing, -5 crash | 786 m | 8% | 20.8 m/s | 2.71 |
| `20261104_180445_dqn_40k-slow-settings` | Default reward | 765 m | 40% | 24.0 m/s | 25.68 |
| `20261105_121448_ppo_65k-lr3e-4-resumed` | No normalizing, -5 crash | 752 m | 30% | 22.3 m/s | 4.35 |
| `20261104_161212_dqn_smoke-test` | Default reward | 288 m | 100% | 28.5 m/s | 8.73 |

Best by goal: `penalty5` or `penalty2` for speed and distance, `dqn_40k` for
balance, and `ppo_40k-lr3e-4` for safety while still beating crawling.

## Runs

Rewards below are scored with no normalizing and a -5 crash penalty, so their
scale can differ from the table above. The hypotheses for these first
seven runs were reconstructed afterwards from commits and notes, not written
before training.

### 20261104_161212_dqn_smoke-test

- **Hypothesis:** The training pipeline runs end to end.
- **Change:** First run, 600 steps with highway-env's documented DQN settings.
- **Result:** mean_reward=-0.84, crash_rate=100% (200 episodes)
- **Conclusion:** The pipeline works. This score is only a floor.

### 20261104_161334_dqn_20k

- **Hypothesis:** DQN with the documented settings learns to avoid crashes in 20k steps.
- **Change:** 20,000 steps, up from 600.
- **Result:** mean_reward=5.69, crash_rate=24% (200 episodes, best model at about 15k steps)
- **Conclusion:** It learns, but got worse late on: the final model crashed 62%
  in its 48-episode eval. Keep the best model from now on.

### 20261104_171828_dqn_40k

- **Hypothesis:** Doubling training to 40k steps cuts crashes further.
- **Change:** 40,000 steps, up from 20,000.
- **Result:** mean_reward=6.88, crash_rate=20% (200 episodes)
- **Conclusion:** Not shown. 20% against 24% is within noise.

### 20261104_180445_dqn_40k-slow-settings

- **Hypothesis:** Slower DQN settings steady the late-training decline.
- **Change:** learning_rate 5e-4 to 1e-4, buffer_size 15,000 to 50,000,
  target_update_interval 50 to 1,000.
- **Result:** mean_reward=4.17, crash_rate=50% (200 episodes)
- **Conclusion:** Wrong at this length: far more crashes than the 40k run. Three
  settings changed at once, so it can't say which one hurt.

### 20261104_190015_ppo_200k

- **Hypothesis:** PPO drives better than DQN.
- **Change:** PPO for 200,000 steps, with its documented settings.
- **Result:** mean_reward=7.15, crash_rate=24% (200 episodes)
- **Conclusion:** Tied with the best DQN on the same reward. Its eval reward stayed
  flat from the first eval at 10k steps. The only PPO run on the new reward so far
  is the 20k one with the lower learning rate.
- **Near 40k steps:** PPO saved no 40k model, so its nearest checkpoints stand in
  for it, against the 40k DQN run's 6.88 reward and 20% crashes. Over 200 episodes,
  25k scored 4.19 with 24% crashes, and 50k scored 4.52 with 22%. DQN looks ahead
  on reward, with crashes level, but its model is the best of 16 evals, while
  these checkpoints weren't selected.

### 20261104_210951_dqn_penalty2

- **Hypothesis:** Removing normalizing stops survival alone from scoring well, so
  the agent learns to drive faster.
- **Change:** normalize_reward True to False, and collision_reward -1 to -2.
- **Result:** mean_reward=11.26, crash_rate=44% (200 episodes)
- **Conclusion:** It drives faster, earning about twice the reward per step of the
  default-reward runs, but crashes about twice as often as the 40k run. Why isn't
  settled: two settings changed at once, and best-model selection changed too.

### 20261104_215246_dqn_penalty5

- **Hypothesis:** A stronger crash penalty cuts crashes without losing speed.
- **Change:** collision_reward -2 to -5.
- **Result:** mean_reward=10.85, crash_rate=42% (200 episodes)
- **Conclusion:** Not shown: 42% against 44% is within noise. At gamma 0.8 this
  reward trains about like the default one (around -6), so that doesn't explain
  the gap to the default-reward runs. Repeat seeds would help.

### 20261105_102757_ppo_20k-lr1e-4

- **Hypothesis:** A lower learning rate gets PPO to a better 20k score than its 25k checkpoint (4.19 reward, 24% crashes)
- **Change:** PPO for 20,000 steps, learning_rate 5e-4 to 1e-4, on the new reward (no normalizing, -5 crash)
- **Result:** mean_reward=1.92 +/- 1.70, mean_steps=39.6/40, crash_rate=1% (200 episodes)
- **Conclusion:** Wrong on reward: 1.92 is well below the checkpoint's 4.19. The score
  matches always `SLOWER` (about 1.9 with 1% crashes), so it looks like it learned to
  crawl. Two things changed from the checkpoint, so this can't blame the learning rate alone.
- **Videos:** 3 episodes from `best_model.zip` in the run's `videos/` folder. All lasted
  40 steps with no crash, with rewards of 1.37, 4.24 and 2.71.

### 20261105_105138_ppo_20k-lr1e-4-default-reward

- **Hypothesis:** On the default reward, a lower learning rate beats the PPO 25k checkpoint (4.19 reward, 24% crashes)
- **Change:** 20k low learning rate PPO, with the default reward instead of the new one
- **Result:** mean_reward=2.00 +/- 1.69, mean_steps=39.5/40, crash_rate=2% (200 episodes,
  new reward scale). On the default reward's scale: 28.00, 0% crashes (48 episodes).
- **Conclusion:** Wrong on reward: 2.00 is well below the checkpoint's 4.19. It tied the
  new-reward 20k run and matched always `SLOWER`, so the reward isn't what caused the
  apparent crawling. The lower learning rate or the short run is more likely, and one
  run each can't separate those.

### 20261105_111332_ppo_20k-lr3e-4

- **Hypothesis:** A higher learning rate gets 20k PPO past crawling, beating the 1e-4 run's 1.92 reward on the new reward
- **Change:** learning_rate 1e-4 to 3e-4, compared with 20261105_102757_ppo_20k-lr1e-4
- **Result:** mean_reward=3.16 +/- 2.63, mean_steps=38.2/40, crash_rate=8% (200 episodes)
- **Conclusion:** Supported: 3.16 beats the 1e-4 run's 1.92 by more than noise, and the
  score is now above always `SLOWER`, with more crashes (8% against 1%). It's still
  slow driving, at about 0.09 reward per step before crash penalties. One run each,
  so repeat seeds would confirm this.

### 20261105_113120_ppo_40k-lr3e-4

- **Hypothesis:** Doubling steps to 40k lifts PPO past the 20k run's 3.16 reward, and toward DQN penalty5's 10.85
- **Change:** timesteps 20,000 to 40,000, compared with 20261105_111332_ppo_20k-lr3e-4
- **Result:** mean_reward=4.31 +/- 3.02, mean_steps=37.6/40, crash_rate=10% (200 episodes)
- **Conclusion:** Partly supported: 4.31 beats 20k's 3.16 by more than noise, but it's
  far from DQN penalty5's 10.85 at the same step count. PPO is much safer (10% crashes
  against 42%) but slower, at about 0.13 reward per step before crash penalties against
  0.40. Mid-training evals rose to 4.73, then gave 4.18 at the end. One run each.

### 20261105_121448_ppo_65k-lr3e-4-resumed

- **Hypothesis:** More steps let PPO catch DQN penalty5's speed (10.85 reward) while staying safer than its 42% crashes
- **Change:** Resumed the 40k run's best model (about 30k steps) on the same learning rate and reward. Planned 100k, stopped by hand at about 65k, so it has no `model.zip`.
- **Result:** mean_reward=5.19 +/- 3.83, mean_steps=36.4/40, crash_rate=18% (200 episodes, best model at about 60k steps)
- **Conclusion:** Partly supported, on too few steps to judge: reward rose from 4.31 and crashes from 10%, as it drove a little faster, at about 0.17 reward per step before crash penalties. It's still far from penalty5's 10.85 and 0.40. `ep_rew_mean` showed no clear trend, while evals rose slowly. One run.
- **Rescore:** A second 200 episodes on the same model and reward gave 4.35 with 30% crashes, so the first score ran high. Over both, it's about 4.77 with 24% crashes, level with the 40k run on reward and crashing more. That undoes the gain above.
