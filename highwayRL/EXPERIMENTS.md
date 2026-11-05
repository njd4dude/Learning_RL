# Experiments

One entry per run, oldest first. `train.py` adds each new entry; write its
conclusion once you've looked at the results.

Each run's folder under `runs/` has its videos, models and eval scores.

## Ranking

All scored by `evaluate.py` on `best_model.zip` (the smoke test has only
`model.zip`) over 200 episodes, with the same reward: `normalize_reward=False`,
`collision_reward=-5`. Always `SLOWER` scores about 1.9 on this scale.

Mean reward is accurate to about ±0.6 to ±1.1, and crash rate to about ±6 to ±7
points at the higher rates. Each result is one training run, with no repeat
seeds, so a real difference between runs could still be luck in training.

| Rank | Run | Trained with | Mean reward | Crash rate | Mean steps |
|---|---|---|---|---|---|
| 1 (tie) | `20261104_210951_dqn_penalty2` | No normalizing, -2 crash | 11.26 | 44% | 31.2 |
| 1 (tie) | `20261104_215246_dqn_penalty5` | No normalizing, -5 crash | 10.85 | 42% | 32.7 |
| 3 (tie) | `20261104_190015_ppo_200k` | Default reward | 7.15 | 24% | 33.8 |
| 3 (tie) | `20261104_171828_dqn_40k` | Default reward | 6.88 | 20% | 35.2 |
| 5 | `20261104_161334_dqn_20k` | Default reward | 5.69 | 24% | 35.6 |
| 6 | `20261104_180445_dqn_40k-slow-settings` | Default reward | 4.17 | 50% | 29.7 |
| 7 | `20261104_161212_dqn_smoke-test` | Default reward | -0.84 | 100% | 10.4 |

## Runs

Scores below are from the ranking table above. The hypotheses for these first
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
- **Change:** PPO for 200,000 steps, with its documented settings. Still the only PPO run.
- **Result:** mean_reward=7.15, crash_rate=24% (200 episodes)
- **Conclusion:** Tied with the best DQN on the same reward. Its eval reward stayed
  flat from the first eval at 10k steps. PPO is untested on the new reward.

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
