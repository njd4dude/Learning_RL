# Experiments

One entry per run, oldest first. `train.py` adds each new entry; write its
conclusion once you've looked at the results.

Each run's folder under `runs/` has its videos, models and eval scores.

## Ranking

Sorted by distance per episode. Distance, speed and crash rate don't depend on
the reward, so they compare across every run. Own reward is the score on the
reward that run trained with, so those only compare between runs with the same
reward. All are `evaluate.py` over 200 episodes, on `best_model.zip`
(the smoke test has only `model.zip`).

Always `SLOWER` crawls at 20 m/s for about 800 m. The maximum is 1,200 m: 30 m/s
for 40 steps. Crash rates are accurate to about ±6 to ±7 points at the higher
rates, and two scores of the same model can differ by more than that. Each result
is one training run, with no repeat seeds.

| Run | Trained with | Distance | Crash rate | Mean speed | Own reward |
|---|---|---|---|---|---|
| `20261107_161057_dqn_240k-penalty2-lr25-resumed` | No normalizing, -2 crash, learning rate 2.5e-4 | 1102 m | 14% | 29.5 m/s | 16.96 |
| `20261108_111309_dqn_320k-penalty2-lr125-resumed` | No normalizing, -2 crash, learning rate 1.25e-4 | 1075 m | 18% | 29.5 m/s | 16.35 |
| `20261107_095805_dqn_320k-default-2hz-seed2-lr25-resumed` | Normalized, -1 crash, 2 decisions a second, gamma 0.9, learning rate 2.5e-4 | 1069 m | 24% | 29.7 m/s | 69.99 (80 steps, so not comparable) |
| `penalty2_chain/20261105_170129_dqn_160k-penalty2-resumed` | No normalizing, -2 crash | 1057 m | 21% | 29.6 m/s | 15.84 |
| `20261108_142659_dqn_320k-penalty2-lr25-resumed` | No normalizing, -2 crash, learning rate 2.5e-4 | 1056 m | 24% | 29.7 m/s | 16.14 |
| `default_chain/20261106_130918_dqn_120k-default-resumed` | Normalized, -1 crash | 1053 m | 23% | 29.5 m/s | 33.95 |
| `penalty2_chain/20261105_163251_dqn_120k-penalty2-resumed` | No normalizing, -2 crash | 1049 m | 22% | 29.3 m/s | 15.59 |
| `20261107_165432_dqn_200k-default-lr25-resumed` | Normalized, -1 crash, learning rate 2.5e-4 | 1017 m | 25% | 29.2 m/s | 33.42 |
| `20261106_215013_dqn_240k-default-2hz-seed2` | Normalized, -1 crash, 2 decisions a second, gamma 0.9 | 1009 m | 34% | 29.6 m/s | 65.17 (80 steps, so not comparable) |
| `20261106_155133_dqn_120k-default-gamma09-seed2` | Normalized, -1 crash, gamma 0.9 | 1001 m | 28% | 28.9 m/s | 32.54 |
| `penalty2_chain/20261105_160009_dqn_80k-penalty2-resumed` | No normalizing, -2 crash | 992 m | 32% | 29.4 m/s | 14.62 |
| `20261106_212422_dqn_160k-default-gamma09-seed2-resumed` | Normalized, -1 crash, gamma 0.9 | 986 m | 19% | 27.3 m/s | 32.79 |
| `20261108_122129_dqn_40k-off2-seed1` | No normalizing, -2 crash, 40k, seed 1 | 979 m | 44% | 29.5 m/s | 13.99 |
| `20261106_142353_dqn_120k-default-seed2` | Normalized, -1 crash | 977 m | 34% | 29.6 m/s | 31.76 |
| `normalized_speed2_chain/20261105_215327_dqn_120k-normalized-speed2-resumed` | Normalized, -2 crash, 2 full speed | 972 m | 42% | 29.7 m/s | 31.83 |
| `normalized_speed2_xrange100_chain/20261106_120112_dqn_80k-normalized-speed2-xrange100-resumed` | Normalized, -2 crash, 2 full speed, car distance scaled to 100 m | 972 m | 38% | 29.8 m/s | 31.75 |
| `20261107_104023_dqn_120k-default-gamma09-explore03-seed2` | Normalized, -1 crash, gamma 0.9, epsilon decays over 30% of steps | 970 m | 40% | 29.7 m/s | 31.34 |
| `20261108_134724_dqn_40k-off2-seed3` | No normalizing, -2 crash, 40k, seed 3 | 964 m | 39% | 29.3 m/s | 13.63 |
| `normalized_speed2_chain/20261105_200605_dqn_80k-normalized-speed2-resumed` | Normalized, -2 crash, 2 full speed | 947 m | 42% | 29.6 m/s | 30.87 |
| `penalty2_chain/20261104_210951_dqn_penalty2` | No normalizing, -2 crash | 921 m | 46% | 29.2 m/s | 12.84 |
| `20261104_215246_dqn_penalty5` | No normalizing, -5 crash | 916 m | 40% | 27.9 m/s | 10.74 |
| `20261108_130435_dqn_40k-off2-seed2` | No normalizing, -2 crash, 40k, seed 2 | 873 m | 52% | 28.9 m/s | 11.93 |
| `default_chain/20261104_171828_dqn_40k` | Normalized, -1 crash | 864 m | 17% | 23.9 m/s | 29.59 |
| `20261106_135856_dqn_40k-normalized-speed2-crash5` | Normalized, -5 crash, 2 full speed | 853 m | 18% | 23.4 m/s | 29.36 |
| `20261108_115946_dqn_40k-norm2-seed1` | Normalized, -2 crash, 40k, seed 1 | 849 m | 6% | 21.9 m/s | 32.89 |
| `20261108_132555_dqn_40k-norm2-seed3` | Normalized, -2 crash, 40k, seed 3 | 840 m | 8% | 22.2 m/s | 32.39 |
| `normalized_penalty2_chain/20261105_172705_dqn_80k-normalized-penalty2-resumed` | Normalized, -2 crash | 839 m | 27% | 24.3 m/s | 30.60 |
| `normalized_penalty2_chain/20261105_135955_dqn_40k-normalized-penalty2` | Normalized, -2 crash | 836 m | 17% | 23.0 m/s | 31.37 |
| `20261105_153008_dqn_20k-norm2-seed2` | Normalized, -2 crash | 832 m | 2% | 21.1 m/s | 32.78 |
| `20261108_124242_dqn_40k-norm2-seed2` | Normalized, -2 crash, 40k, seed 2 | 832 m | 20% | 23.5 m/s | 31.23 |
| `20261105_133450_dqn_40k-default-reward-repeat` | Normalized, -1 crash | 830 m | 49% | 27.1 m/s | 27.08 |
| `20261104_190015_ppo_200k` | Normalized, -1 crash | 825 m | 22% | 24.1 m/s | 28.26 |
| `20261104_161334_dqn_20k` | Normalized, -1 crash | 824 m | 20% | 23.2 m/s | 28.31 |
| `ppo_lr3e-4_chain/20261105_113120_ppo_40k-lr3e-4` | No normalizing, -5 crash | 819 m | 8% | 21.4 m/s | 4.55 |
| `20261105_144614_dqn_20k-norm1-seed2` | Normalized, -1 crash | 818 m | 6% | 21.4 m/s | 28.22 |
| `20261107_180111_dqn_30k-default-gamma099-target500-seed2` | Normalized, -1 crash, gamma 0.99, target update every 500 steps, 30k steps | 808 m | 1% | 20.2 m/s | 29.12 |
| `20261105_154440_dqn_20k-norm2-seed3` | Normalized, -2 crash | 800 m | 2% | 20.2 m/s | 32.78 |
| `20261107_202041_dqn_60k-default-gamma099-target500-seed2-resumed` | Normalized, -1 crash, gamma 0.99, target update every 500 steps, 60k steps | 800 m | 4% | 20.4 m/s | 28.92 |
| `20261105_102757_ppo_20k-lr1e-4` | No normalizing, -5 crash | 798 m | 2% | 20.1 m/s | 2.26 |
| **Always `SLOWER` (baseline, no model)** | Not trained | 795 m | 2% | 20.0 m/s | 27.85 normalized, -1; 1.96 no normalizing, -5 |
| `20261105_105138_ppo_20k-lr1e-4-default-reward` | Normalized, -1 crash | 794 m | 1% | 20.0 m/s | 27.79 |
| `20261107_174508_dqn_30k-default-gamma099-seed2` | Normalized, -1 crash, gamma 0.99, 30k steps | 792 m | 3% | 20.1 m/s | 28.66 |
| `20261105_111332_ppo_20k-lr3e-4` | No normalizing, -5 crash | 786 m | 8% | 20.8 m/s | 2.71 |
| `normalized_speed2_xrange100_chain/20261106_112536_dqn_40k-normalized-speed2-xrange100` | Normalized, -2 crash, 2 full speed, car distance scaled to 100 m | 780 m | 68% | 28.7 m/s | 24.99 |
| `normalized_speed2_chain/20261105_193556_dqn_40k-normalized-speed2` | Normalized, -2 crash, 2 full speed | 779 m | 70% | 29.5 m/s | 25.11 |
| `20261105_150125_dqn_20k-norm1-seed3` | Normalized, -1 crash | 770 m | 18% | 21.9 m/s | 26.65 |
| `20261104_180445_dqn_40k-slow-settings` | Normalized, -1 crash | 765 m | 40% | 24.0 m/s | 25.68 |
| `ppo_lr3e-4_chain/20261105_121448_ppo_65k-lr3e-4-resumed` | No normalizing, -5 crash | 752 m | 30% | 22.3 m/s | 4.35 |
| `20261105_143119_dqn_20k-norm1-seed1` | Normalized, -1 crash | 739 m | 30% | 22.6 m/s | 25.40 |
| `20261106_124654_dqn_40k-normalized-speed2-right05` | Normalized, -2 crash, 2 full speed, 0.5 right lane | 709 m | 78% | 29.4 m/s | 22.05 |
| `20261105_151550_dqn_20k-norm2-seed1` | Normalized, -2 crash | 678 m | 43% | 22.2 m/s | 25.82 |
| `20261104_161212_dqn_smoke-test` | Normalized, -1 crash | 288 m | 100% | 28.5 m/s | 8.73 |

Best by goal: `160k-penalty2-resumed` or `120k-penalty2-resumed` for speed, distance and
balance, and
`ppo_40k-lr3e-4` for safety while still beating crawling. These picks
are tentative: repeating `dqn_40k` with the same settings gave 49% crashes against
its 17%, so a single DQN run can land far from another with the same settings.

## Seed comparison

DQN for 20k steps on seeds 1, 2 and 3, scored over 200 episodes. Averages are of the
three runs. Own rewards aren't compared, since they're on different scales.

| Reward | Seed 1 | Seed 2 | Seed 3 | Mean distance | Mean crash rate | Mean speed |
|---|---|---|---|---|---|---|
| Normalized, -1 crash | 739 m, 30% | 818 m, 6% | 770 m, 18% | 776 m | 18% | 21.9 m/s |
| Normalized, -2 crash | 678 m, 43% | 832 m, 2% | 800 m, 2% | 770 m | 16% | 21.2 m/s |

No clear difference: the spread between seeds is far larger than the gap between the
rewards. Medians favor -2 (2% against 18% crashes), but seed 1 went the other way, at
43%. Seed 3 on -2 ran at 20.2 m/s, close to crawling. At 20k, none of these beat the
always `SLOWER` baseline on speed by much. Three seeds can't separate these two.

### Normalizing on or off, 40k

DQN for 40k steps on seeds 1, 2 and 3, both with a -2 crash, scored over 200
episodes. Mid-training evals were 48 episodes every 5,000 steps.

| Reward | Seed 1 | Seed 2 | Seed 3 | Mean distance | Mean crash rate | Mean speed |
|---|---|---|---|---|---|---|
| Normalized, -2 crash | 849 m, 6% | 832 m, 20% | 840 m, 8% | 840 m | 11% | 22.5 m/s |
| Not normalized, -2 crash | 979 m, 44% | 873 m, 52% | 964 m, 39% | 939 m | 45% | 29.2 m/s |

Not normalizing went farther on every seed, by 41 to 130 m, the first setting to
beat seed-to-seed spread. The two learn different things first: normalized runs
drive safely but slowly, close to always braking's 20 m/s, since surviving pays
about 0.7 a step; not normalized runs drive at nearly full speed and crash about
half the time. At 40k that trade favors speed. It may not hold with more training:
normalized chains later reached full speed too, and the best of each style ended
close, at 1,102 m not normalized against 1,069 m normalized.

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

### default_chain/20261104_171828_dqn_40k

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

### penalty2_chain/20261104_210951_dqn_penalty2

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

### ppo_lr3e-4_chain/20261105_113120_ppo_40k-lr3e-4

- **Hypothesis:** Doubling steps to 40k lifts PPO past the 20k run's 3.16 reward, and toward DQN penalty5's 10.85
- **Change:** timesteps 20,000 to 40,000, compared with 20261105_111332_ppo_20k-lr3e-4
- **Result:** mean_reward=4.31 +/- 3.02, mean_steps=37.6/40, crash_rate=10% (200 episodes)
- **Conclusion:** Partly supported: 4.31 beats 20k's 3.16 by more than noise, but it's
  far from DQN penalty5's 10.85 at the same step count. PPO is much safer (10% crashes
  against 42%) but slower, at about 0.13 reward per step before crash penalties against
  0.40. Mid-training evals rose to 4.73, then gave 4.18 at the end. One run each.

### ppo_lr3e-4_chain/20261105_121448_ppo_65k-lr3e-4-resumed

- **Hypothesis:** More steps let PPO catch DQN penalty5's speed (10.85 reward) while staying safer than its 42% crashes
- **Change:** Resumed the 40k run's best model (about 30k steps) on the same learning rate and reward. Planned 100k, stopped by hand at about 65k, so it has no `model.zip`.
- **Result:** mean_reward=5.19 +/- 3.83, mean_steps=36.4/40, crash_rate=18% (200 episodes, best model at about 60k steps)
- **Conclusion:** Partly supported, on too few steps to judge: reward rose from 4.31 and crashes from 10%, as it drove a little faster, at about 0.17 reward per step before crash penalties. It's still far from penalty5's 10.85 and 0.40. `ep_rew_mean` showed no clear trend, while evals rose slowly. One run.
- **Rescore:** A second 200 episodes on the same model and reward gave 4.35 with 30% crashes, so the first score ran high. Over both, it's about 4.77 with 24% crashes, level with the 40k run on reward and crashing more. That undoes the gain above.

### 20261105_133450_dqn_40k-default-reward-repeat

- **Hypothesis:** On the default reward, penalty2's settings drive slower but crash less, matching dqn_40k (864 m, 17% crashes)
- **Change:** Reward back to default from penalty2. Same settings as dqn_40k, so this repeats it with new random luck
- **Result:** mean_reward=27.08 +/- 10.97, mean_steps=30.6/40, crash_rate=49%, mean_speed=27.1m/s, mean_distance=830m (200 episodes)
- **Conclusion:** Wrong: it didn't match `dqn_40k`. With identical settings it crashed 49% against 17%, and drove 27.1 m/s against 23.9. That's far beyond scoring noise, so training luck moves single DQN runs a lot. It looks like `penalty2` and `penalty5`, which fits the math that this reward trains like about -5. So the new reward may not be what made those runs faster. Compare settings over several seeds from now on.

### normalized_penalty2_chain/20261105_135955_dqn_40k-normalized-penalty2

- **Hypothesis:** Normalizing with a -2 crash penalty drives more cautiously than the default reward, with fewer crashes and lower speed
- **Change:** collision_reward -1 to -2, keeping normalizing
- **Result:** mean_reward=31.37 +/- 8.09, mean_steps=36.3/40, crash_rate=17%, mean_speed=23.0m/s, mean_distance=836m (200 episodes)
- **Conclusion:** Consistent so far: 17% crashes against 49% for the -1 repeat, and slower, at 23.0 m/s against 27.1. But `dqn_40k` got the same 17% on -1, so one run can't separate this from luck. The 20k seed comparison tests it. It didn't crawl, which was my prediction.

### 20261105_143119_dqn_20k-norm1-seed1

- **Hypothesis:** Baseline for the seed comparison: normalized, -1 crash, 20k steps
- **Change:** Seed 1 of 3, on the -1 crash reward
- **Result:** mean_reward=25.40 +/- 8.96, mean_steps=32.7/40, crash_rate=30%, mean_speed=22.6m/s, mean_distance=739m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### 20261105_144614_dqn_20k-norm1-seed2

- **Hypothesis:** Baseline for the seed comparison: normalized, -1 crash, 20k steps
- **Change:** Seed 2 of 3, on the -1 crash reward
- **Result:** mean_reward=28.22 +/- 5.39, mean_steps=38.2/40, crash_rate=6%, mean_speed=21.4m/s, mean_distance=818m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### 20261105_150125_dqn_20k-norm1-seed3

- **Hypothesis:** Baseline for the seed comparison: normalized, -1 crash, 20k steps
- **Change:** Seed 3 of 3, on the -1 crash reward
- **Result:** mean_reward=26.65 +/- 8.05, mean_steps=35.2/40, crash_rate=18%, mean_speed=21.9m/s, mean_distance=770m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### 20261105_151550_dqn_20k-norm2-seed1

- **Hypothesis:** Normalized, -2 crash, drives more safely than -1 across seeds
- **Change:** Seed 1 of 3, collision_reward -1 to -2, keeping normalizing
- **Result:** mean_reward=25.82 +/- 10.64, mean_steps=30.5/40, crash_rate=43%, mean_speed=22.2m/s, mean_distance=678m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### 20261105_153008_dqn_20k-norm2-seed2

- **Hypothesis:** Normalized, -2 crash, drives more safely than -1 across seeds
- **Change:** Seed 2 of 3, collision_reward -1 to -2, keeping normalizing
- **Result:** mean_reward=32.78 +/- 3.20, mean_steps=39.5/40, crash_rate=2%, mean_speed=21.1m/s, mean_distance=832m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### 20261105_154440_dqn_20k-norm2-seed3

- **Hypothesis:** Normalized, -2 crash, drives more safely than -1 across seeds
- **Change:** Seed 3 of 3, collision_reward -1 to -2, keeping normalizing
- **Result:** mean_reward=32.78 +/- 2.78, mean_steps=39.6/40, crash_rate=2%, mean_speed=20.2m/s, mean_distance=800m (200 episodes)
- **Conclusion:** See the seed comparison at the top.

### penalty2_chain/20261105_160009_dqn_80k-penalty2-resumed

- **Hypothesis:** penalty2 was still improving at 40k, so 40k more steps lift it past its 921 m without more crashes
- **Change:** 40,000 more steps on penalty2's model.zip, same no normalizing, -2 crash reward, empty replay buffer (resumed from runs/penalty2_chain/20261104_210951_dqn_penalty2/model.zip)
- **Result:** mean_reward=14.62 +/- 6.06, mean_steps=33.8/40, crash_rate=32%, mean_speed=29.4m/s, mean_distance=992m (200 episodes)
- **Conclusion:** Supported: 992 m against 921 m at the same speed, with crashes down from 46% to 32%. It kept high speed and got safer, the first run to do both. There was no dip from the empty replay buffer. One run, so it needs repeating, though the training curve kept rising.

### penalty2_chain/20261105_163251_dqn_120k-penalty2-resumed

- **Hypothesis:** Another 40k steps keeps cutting crashes below 32% while holding about 29 m/s
- **Change:** 40,000 more steps on the 80k run's model.zip with its saved replay buffer, same reward (resumed from runs/penalty2_chain/20261105_160009_dqn_80k-penalty2-resumed/model.zip)
- **Result:** mean_reward=15.59 +/- 5.50, mean_steps=35.8/40, crash_rate=22%, mean_speed=29.3m/s, mean_distance=1049m (200 episodes)
- **Conclusion:** Supported: crashes fell again, from 32% to 22%, at the same speed, and distance rose to 1,049 m. Crashes have gone 46%, 32%, 22% as steps went 40k, 80k, 120k. The saved replay buffer loaded with 15,000 steps. One run, so it needs repeating on other seeds.

### penalty2_chain/20261105_170129_dqn_160k-penalty2-resumed

- **Hypothesis:** Another 40k steps cuts crashes below 22% while holding about 29 m/s
- **Change:** 40,000 more steps on the 120k run's model.zip with its saved replay buffer, same reward (resumed from runs/penalty2_chain/20261105_163251_dqn_120k-penalty2-resumed/model.zip)
- **Result:** mean_reward=15.84 +/- 5.42, mean_steps=35.7/40, crash_rate=21%, mean_speed=29.6m/s, mean_distance=1057m (200 episodes)
- **Conclusion:** Not shown: 21% against 22% is within noise, and distance barely moved. `ep_rew_mean` stayed about 10 to 14 and evals about 12 to 17, with no trend, so this line of training looks to have plateaued around 120k. Going further on these settings probably won't cut crashes much.

### normalized_penalty2_chain/20261105_172705_dqn_80k-normalized-penalty2-resumed

- **Hypothesis:** With 40k more steps, normalized -2 gets faster and safer, as penalty2 did with normalizing off
- **Change:** 40,000 more steps on the normalized -2 run's model.zip, same reward, empty replay buffer (resumed from runs/normalized_penalty2_chain/20261105_135955_dqn_40k-normalized-penalty2/model.zip)
- **Result:** mean_reward=30.60 +/- 9.19, mean_steps=34.6/40, crash_rate=27%, mean_speed=24.3m/s, mean_distance=839m (200 episodes)
- **Conclusion:** Wrong: distance stayed about the same at 839 m, and crashes rose from 17% to 27%, near the edge of noise. `ep_rew_mean` stayed flat around 24 to 28, so more steps didn't help as they did with normalizing off. At 80k, normalizing off is 153 m farther and 5 m/s faster, at a similar crash rate (32%). One run each.

### normalized_speed2_chain/20261105_193556_dqn_40k-normalized-speed2

- **Hypothesis:** Raising high_speed_reward to 2 gives normalized -2 the same speed vs crash balance as normalizing off, so it drives near 29 m/s and beats the 40k normalized run's 836 m
- **Change:** Normalized -2 crash with high_speed_reward 2 instead of 0.4, 40k steps from scratch
- **Result:** mean_reward=25.11 +/- 12.09, mean_steps=26.4/40, crash_rate=70%, mean_speed=29.5m/s, mean_distance=779m (200 episodes)
- **Conclusion:** Half right: speed rose from 23.0 to 29.5 m/s, matching normalizing off, so the alive bonus was what kept normalized runs slow. But crashes reached 70%, so distance fell to 779 m, below the 836 m it was meant to beat and well below `dqn_penalty2`'s 921 m and 46% at the same speed and steps. One run, and identical settings have given 17% and 49%, so the crash gap may be luck.

### normalized_speed2_chain/20261105_200605_dqn_80k-normalized-speed2-resumed

- **Hypothesis:** With 40k more steps crashes fall from 70% while holding about 29 m/s, as normalizing off fell from 46% to 32%
- **Change:** 40,000 more steps on the 40k run's model.zip with its saved replay buffer, same reward (resumed from runs/normalized_speed2_chain/20261105_193556_dqn_40k-normalized-speed2/model.zip)
- **Result:** mean_reward=30.87 +/- 11.77, mean_steps=31.9/40, crash_rate=42%, mean_speed=29.6m/s, mean_distance=947m (200 episodes)
- **Conclusion:** Right: crashes fell from 70% to 42% and speed held at 29.6 m/s, so distance rose from 779 to 947 m, the best of any normalized run. Normalizing off at 80k is still ahead at 992 m and 32%, though that crash gap is about the size of the scoring noise. One run each.

### normalized_speed2_chain/20261105_215327_dqn_120k-normalized-speed2-resumed

- **Hypothesis:** Another 40k steps cuts crashes below 42% while holding about 29 m/s, closing on normalizing off's 1049 m and 22% at 120k
- **Change:** 40,000 more steps on the 80k run's model.zip with its saved replay buffer, same reward (resumed from runs/normalized_speed2_chain/20261105_200605_dqn_80k-normalized-speed2-resumed/model.zip)
- **Result:** mean_reward=31.83 +/- 11.12, mean_steps=32.7/40, crash_rate=42%, mean_speed=29.7m/s, mean_distance=972m (200 episodes)
- **Conclusion:** Wrong: crashes stayed at 42%, and distance only rose from 947 to 972 m, within noise. Normalizing off dropped to 22% and 1049 m at 120k, so the gap widened from 10 points to 20 and is now larger than scoring noise. Speed matches normalizing off, but crash avoidance doesn't yet. One run each.

### normalized_speed2_xrange100_chain/20261106_112536_dqn_40k-normalized-speed2-xrange100

- **Hypothesis:** Scaling car distances by 100 m instead of 200 m makes close gaps easier to separate, so Q(LANE_RIGHT) with a car 5-20 m behind gets closer to its real return, and crashes stay below the 40k run's 70%
- **Change:** x features_range +/-100 m instead of +/-200 m, on normalized -2 crash with high_speed_reward 2, 40k steps from scratch
- **Result:** mean_reward=24.99 +/- 11.74, mean_steps=27.2/40, crash_rate=68%, mean_speed=28.7m/s, mean_distance=780m (200 episodes)
- **Conclusion:** Half right: the Q-values improved but the driving didn't. With a car 5-20 m behind in the right lane and none just ahead there, Q(LANE_RIGHT) averaged 3.58 against a real return of 3.66 (256 states, bias -0.07, mean error 0.92), where the 200 m run gave 1.37 against 3.81 (269 states, bias -2.44, mean error 2.47). Negative Q-values there fell from 21% to 2%, though true Q-values can't go below 0. Crashes and distance matched the 200 m run at 68% and 780 m against 70% and 779 m, so fixing this estimate didn't fix crashes. One run each.

### normalized_speed2_xrange100_chain/20261106_120112_dqn_80k-normalized-speed2-xrange100-resumed

- **Hypothesis:** With 40k more steps crashes fall from 68% and at least match the 200 m run's 42% at 80k, now that Q-values with a car close behind are accurate
- **Change:** 40,000 more steps on the 40k run's model.zip with its saved replay buffer, same reward and +/-100 m distance scale (resumed from runs/normalized_speed2_xrange100_chain/20261106_112536_dqn_40k-normalized-speed2-xrange100/model.zip)
- **Result:** mean_reward=31.75 +/- 11.43, mean_steps=32.7/40, crash_rate=38%, mean_speed=29.8m/s, mean_distance=972m (200 episodes)
- **Conclusion:** Supported, but not as a clear win: crashes fell from 68% to 38% and distance rose from 780 to 972 m, against 42% and 947 m for the 200 m run at 80k, which is within scoring noise. The Q-value gain had also mostly gone: with a car 5-20 m behind, bias was -0.49 and mean error 1.38 here against -0.69 and 1.32 on the 200 m run. So the narrower range seems to have sped up learning those values at 40k rather than ended up somewhere better. One run each.

### 20261106_124654_dqn_40k-normalized-speed2-right05

- **Hypothesis:** Raising right_lane_reward to 0.5 restores the 25% lane-to-speed ratio of no normalizing, so it weaves less and crashes below the 40k normalized speed2 run's 70% while holding about 29 m/s
- **Change:** right_lane_reward 0.1 to 0.5, on normalized -2 crash with high_speed_reward 2, 40k steps from scratch
- **Result:** mean_reward=22.05 +/- 11.19, mean_steps=24.1/40, crash_rate=78%, mean_speed=29.4m/s, mean_distance=709m (200 episodes)
- **Conclusion:** Wrong: it moved right but didn't crash less. Over 40 matching episodes against the 40k run with 0.1, time in the right two lanes rose from 36% to 61%, but lane changes went from 44 to 49 per 100 steps, so it didn't weave less. Crashes were 78% against 70%, which is within scoring noise, at the same speed. So lane position doesn't look like the reason normalizing crashes more than no normalizing. These rewards also match no normalizing, -2 crash, under one scale and shift (x 1.11 + 2.22), so apart from the 40-step limit both have the same best driving, and a different goal can't explain the gap. Untested guess: normalized crash targets sit near the network's starting values, so crashes only look bad once safe Q-values climb. One run, on seed 1, at 40k rather than the 120k where the gap was measured.

### default_chain/20261106_130918_dqn_120k-default-resumed

- **Hypothesis:** On the default reward, 80k more steps cut crashes below dqn_40k's 17% and raise speed above 23.9 m/s, as extra steps did with normalizing off
- **Change:** 80,000 more steps on dqn_40k's model.zip, same default reward and DQN settings, empty replay buffer because that run saved none (resumed from runs/default_chain/20261104_171828_dqn_40k/model.zip)
- **Result:** mean_reward=33.95 +/- 9.44, mean_steps=35.7/40, crash_rate=23%, mean_speed=29.5m/s, mean_distance=1053m (200 episodes)
- **Conclusion:** Half right, and a surprise: speed rose from 23.9 to 29.5 m/s, but crashes went up from 17% to 23%, which is within noise of it. Distance rose from 864 to 1,053 m, level with the best no-normalizing runs (1,049 m and 22% at 120k) and well ahead of normalized speed2 at 120k (972 m, 42%). So the default reward reaches full speed on its own with more steps, and the alive bonus doesn't keep it slow as the 40k runs suggested. One run, and this model's 40k start was the lucky 17% one.

### 20261106_135856_dqn_40k-normalized-speed2-crash5

- **Hypothesis:** A 2.5:1 crash to speed ratio cuts crashes well below the 40k normalized speed2 run's 70%, though the smaller speed gain probably lowers speed from 29.5 m/s
- **Change:** collision_reward -2 to -5, on normalized with high_speed_reward 2, 40k steps from scratch
- **Result:** mean_reward=29.36 +/- 6.87, mean_steps=36.4/40, crash_rate=18%, mean_speed=23.4m/s, mean_distance=853m (200 episodes)
- **Conclusion:** Right on both counts: crashes fell from 70% to 18%, and speed fell from 29.5 to 23.4 m/s. Distance rose from 779 to 853 m. It drove almost exactly like `dqn_40k` (864 m, 17%, 23.9 m/s), whose gain from full speed is similar at 0.33 against 0.29 a step. So at 40k this reward behaves like the default one, and the real test is whether it also reaches full speed by 120k. One run.

### 20261106_142353_dqn_120k-default-seed2

- **Hypothesis:** A new seed on the default reward also reaches about 1,050 m and 23% crashes at 120k, so dqn_120k-default-resumed wasn't luck
- **Change:** Repeats the 120k default reward run on seed 2, in one go from scratch rather than 40k then an 80k resume with an empty replay buffer
- **Result:** mean_reward=31.76 +/- 11.63, mean_steps=33.0/40, crash_rate=34%, mean_speed=29.6m/s, mean_distance=977m (200 episodes)
- **Conclusion:** Partly: it reached the same 29.6 m/s, so the default reward does get to full speed by 120k, but crashes were 34% against 23% and distance 977 m against 1,053 m, about 1.5 to 2 times scoring noise. So `dqn_120k-default-resumed` was partly luck, and it also differs in training: it was 40k then an 80k resume with an empty buffer, against one run from scratch here. The gap to normalized speed2 at 120k (972 m, 42%) is now within noise, so the default reward isn't clearly safer at this length.

### 20261106_155133_dqn_120k-default-gamma09-seed2

- **Hypothesis:** Looking about 10 seconds ahead instead of 5 lets it move over earlier for slow traffic, cutting crashes well below 120k-default-seed2's 34% at about the same speed
- **Change:** gamma 0.8 to 0.9, same default reward, seed 2 and 120k steps as 120k-default-seed2
- **Result:** mean_reward=32.54 +/- 10.15, mean_steps=34.6/40, crash_rate=28%, mean_speed=28.9m/s, mean_distance=1001m (200 episodes)
- **Conclusion:** Not shown: 28% against 34% and 1,001 m against 977 m are both within scoring noise, and speed was 0.7 m/s lower. The 48-episode score of 17% and 1,075 m was far too kind, so judge only on the 200-episode score. Gamma 0.9 doesn't look harmful, but one run per setting can't show a gain this small.

### 20261106_212422_dqn_160k-default-gamma09-seed2-resumed

- **Hypothesis:** 40k more steps at gamma 0.9 cut crashes below its 28% at 120k, since longer horizons can take longer to learn
- **Change:** 40,000 more steps on the 120k gamma 0.9 run's model.zip with its saved replay buffer, same reward, gamma and seed (resumed from runs/20261106_155133_dqn_120k-default-gamma09-seed2/model.zip)
- **Result:** mean_reward=32.79 +/- 8.40, mean_steps=36.2/40, crash_rate=19%, mean_speed=27.3m/s, mean_distance=986m (200 episodes)
- **Conclusion:** Partly: crashes fell from 28% to 19%, near the edge of scoring noise, but speed fell from 28.9 to 27.3 m/s, so distance stayed about level at 986 m against 1,001 m. It looks like a safety for speed trade rather than better driving. One run.

### 20261106_215013_dqn_240k-default-2hz-seed2

- **Hypothesis:** Deciding twice a second lets it react to traffic sooner, cutting crashes well below 120k-default-seed2's 34% at about the same speed
- **Change:** 2 decisions a second instead of 1, with gamma 0.9 for the same horizon and 240k steps for the same driving time, same default reward and seed as 120k-default-seed2
- **Result:** mean_reward=65.17 +/- 19.99, mean_steps=68.3/80, crash_rate=34%, mean_speed=29.6m/s, mean_distance=1009m (200 episodes)
- **Conclusion:** Wrong: crashes matched `120k-default-seed2` exactly at 34%, at the same 29.6 m/s, and distance was within noise at 1,009 m against 977 m. Deciding twice as often didn't make it react better, at least with the same driving time and a 15,000 step buffer, which now holds half as much driving. Its reward isn't comparable with 1-decision runs, since episodes have 80 steps. Training was paused for 10.4 hours overnight, so `train_seconds` and TensorBoard's wall time include that; it really trained for about 74 minutes. One run.

### 20261107_095805_dqn_320k-default-2hz-seed2-lr25-resumed

- **Hypothesis:** Halving the learning rate lets the 240k model settle, cutting crashes below its 34% at the same speed
- **Change:** 80,000 more steps on the 240k two-decisions run's model.zip with its saved replay buffer, learning_rate 5e-4 to 2.5e-4, same reward, decision rate, gamma and seed (resumed from runs/20261106_215013_dqn_240k-default-2hz-seed2/model.zip)
- **Result:** mean_reward=69.99 +/- 16.94, mean_steps=72.0/80, crash_rate=24%, mean_speed=29.7m/s, mean_distance=1069m (200 episodes)
- **Conclusion:** Supported, though not proven: crashes fell from 34% to 24% and distance rose from 1,009 to 1,069 m at the same 29.7 m/s, the farthest of any run so far. The crash drop is at the edge of scoring noise, and it came with 80k more steps too, so the lower learning rate and the extra training can't be separated. Mid-training evals were best early on (17% and 21% at 245k and 265k) and drifted to 42% by 320k, so best_model.zip is probably an early checkpoint and the lucky pick of noisy 24-episode evals. One run.
  Rescoring the best-looking checkpoints over 200 episodes found nothing better: 245k scored 32% and 1,002 m, 255k 38% and 982 m, 295k 28% and 1,045 m, and 315k 34% and 1,011 m. best_model.zip is the 265k checkpoint, and it stayed best at 24% and 1,069 m, so the pick held up here. The two 17% evals at 245k and 255k were luck, which shows how far a 24-episode eval can be off.

### 20261107_104023_dqn_120k-default-gamma09-explore03-seed2

- **Hypothesis:** Exploring longer before settling finds safer lane changes, cutting crashes below 120k-default-gamma09-seed2's 28% at about the same speed
- **Change:** exploration_fraction 0.1 to 0.3, so epsilon decays over 36k steps instead of 12k, same default reward, gamma 0.9, seed 2 and 120k steps as 120k-default-gamma09-seed2
- **Result:** mean_reward=31.34 +/- 10.96, mean_steps=32.6/40, crash_rate=40%, mean_speed=29.7m/s, mean_distance=970m (200 episodes)
- **Conclusion:** Not supported: crashes rose from 28% to 40% and distance fell from 1,001 to 970 m, though speed was 0.8 m/s higher. The crash gap is beyond scoring noise, but identical settings have given 17% and 49% before, so one run can't show that slower exploration hurts. The 48-episode score of 31% and 1,021 m was again too kind. One run.

### 20261107_161057_dqn_240k-penalty2-lr25-resumed

- **Hypothesis:** Halving the learning rate lets the model settle, cutting crashes below 160k-penalty2's 21% at the same speed, as it did on the two-decisions run
- **Change:** 80,000 more steps on 160k-penalty2's model.zip with its saved replay buffer, learning_rate 5e-4 to 2.5e-4, same no-normalizing -2 crash reward and gamma 0.8; mid-training evals now 48 episodes every 5,000 steps (resumed from runs/penalty2_chain/20261105_170129_dqn_160k-penalty2-resumed/model.zip)
- **Result:** mean_reward=16.96 +/- 4.62, mean_steps=37.3/40, crash_rate=14%, mean_speed=29.5m/s, mean_distance=1102m (200 episodes)
- **Conclusion:** Supported: crashes fell from 21% to 14% and distance rose from 1,057 to 1,102 m at about the same speed, the farthest of any run so far. Like the two-decisions run, it got 80k more steps as well as the lower learning rate, so the two can't be separated, and the same change didn't help 200k-default-lr25. One run.

### 20261107_165432_dqn_200k-default-lr25-resumed

- **Hypothesis:** Halving the learning rate lets the model settle, cutting crashes below 120k-default-resumed's 23% at the same speed, as it did on the two-decisions run
- **Change:** 80,000 more steps on 120k-default-resumed's model.zip with its saved replay buffer, learning_rate 5e-4 to 2.5e-4, same default reward and gamma 0.8; mid-training evals now 48 episodes every 5,000 steps (resumed from runs/default_chain/20261106_130918_dqn_120k-default-resumed/model.zip)
- **Result:** mean_reward=33.42 +/- 10.59, mean_steps=34.9/40, crash_rate=25%, mean_speed=29.2m/s, mean_distance=1017m (200 episodes)
- **Conclusion:** Not supported: crashes stayed level at 25% against 23%, and distance fell from 1,053 to 1,017 m, both within noise. Halving the learning rate helped the penalty2 and two-decisions chains but not this one, so it isn't a reliable gain on its own. One run.

### 20261107_174508_dqn_30k-default-gamma099-seed2

- **Hypothesis:** Looking about 100 steps ahead, the whole episode, makes a crash cost the agent what it costs in distance; this quarter-way check shows whether DQN learns stably at gamma 0.99 before training further
- **Change:** gamma 0.9 to 0.99, same default reward and seed as 120k-default-gamma09-seed2, stopped at 30k steps with exploration_fraction 0.4 so epsilon still decays over 12k steps; mid-training evals now 48 episodes every 5,000 steps
- **Result:** best_model.zip: mean_reward=28.66 +/- 2.89, mean_steps=39.4/40, crash_rate=3%, mean_speed=20.1m/s, mean_distance=792m; model.zip at 30k: crash_rate=30%, mean_speed=21.9m/s, mean_distance=729m (200 episodes each)
- **Conclusion:** Not yet driving well, and unstable: best_model.zip just crawls at 20 m/s like always braking, and the last model crashed 30% at 21.9 m/s. Mid-training evals swung between 0% and 90 to 100% crashes, which looks like the instability a fast target update causes at this horizon. gamma 0.9 at the same point (its 29,952-step checkpoint) scored 44% at 24.8 m/s and 781 m, so 30k steps is early for any gamma. One run.

### 20261107_180111_dqn_30k-default-gamma099-target500-seed2

- **Hypothesis:** Updating the target network 10 times less often keeps gamma 0.99 stable, so it drives better at 30k steps than 30k-default-gamma099-seed2
- **Change:** target_update_interval 50 to 500, otherwise the same as 30k-default-gamma099-seed2
- **Result:** best_model.zip: mean_reward=29.12 +/- 1.10, mean_steps=39.9/40, crash_rate=1%, mean_speed=20.2m/s, mean_distance=808m; model.zip at 30k: crash_rate=38%, mean_speed=21.9m/s, mean_distance=673m (200 episodes each)
- **Conclusion:** Partly: mid-training evals improved steadily, 100%, 69%, 19%, 67%, 21% then 12% crashes, rather than swinging as with updates every 50 steps, so it looks more stable. But it hasn't learned to drive fast yet: best_model.zip crawls at 20 m/s, and the last model's 200-episode score of 38% was far worse than its 12% 48-episode eval. Worth training on to 120k. One run.

### 20261107_202041_dqn_60k-default-gamma099-target500-seed2-resumed

- **Hypothesis:** With more training, gamma 0.99 with slow target updates learns to drive fast without crashing, beating gamma 0.9's 44% and 781 m at 30k and heading toward its 28% and 1,001 m at 120k
- **Change:** 30,000 more steps on 30k-default-gamma099-target500-seed2's model.zip with its saved replay buffer, exploration_fraction back to 0.1 so epsilon stays at 0.05, same reward, gamma 0.99, target updates every 500 steps and seed (resumed from runs/20261107_180111_dqn_30k-default-gamma099-target500-seed2/model.zip)
- **Result:** best_model.zip: mean_reward=28.92 +/- 3.54, mean_steps=39.1/40, crash_rate=4%, mean_speed=20.4m/s, mean_distance=800m; model.zip at 60k: crash_rate=92%, mean_speed=27.4m/s, mean_distance=546m (200 episodes each)
- **Conclusion:** Not yet: mid-training evals swung between crawling and crashing, 77%, 98%, 25%, 2%, 48% then 94% crashes, so the steady improvement seen up to 30k didn't continue. best_model.zip still crawls at 20 m/s, and the last model drives fast but crashed 92% of the time. Epsilon stayed at 0.05, so this isn't extra exploration. One run.

### 20261108_111309_dqn_320k-penalty2-lr125-resumed

- **Hypothesis:** Halving the learning rate again lets the model settle further, cutting crashes below 240k-penalty2-lr25's 14% at the same speed
- **Change:** 80,000 more steps on 240k-penalty2-lr25's model.zip with its saved replay buffer, learning_rate 2.5e-4 to 1.25e-4, same no-normalizing -2 crash reward and gamma 0.8 (resumed from runs/20261107_161057_dqn_240k-penalty2-lr25-resumed/model.zip)
- **Result:** mean_reward=16.35 +/- 5.49, mean_steps=36.5/40, crash_rate=18%, mean_speed=29.5m/s, mean_distance=1075m (200 episodes)
- **Conclusion:** Not supported: 18% and 1,075 m against 14% and 1,102 m, within scoring noise, so halving the learning rate again didn't help. Mid-training evals stayed between 6% and 33% throughout, steadier than earlier chains, which suggests this chain has levelled off around 15% rather than still improving. The 48-episode score of 8% and 1,132 m was again too kind. One run.

### 20261108_115946_dqn_40k-norm2-seed1

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the normalized side
- **Change:** Seed 1 of 3, normalized reward with collision_reward -2, 40k steps
- **Result:** mean_reward=32.89 +/- 4.50, mean_steps=38.9/40, crash_rate=6%, mean_speed=21.9m/s, mean_distance=849m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_122129_dqn_40k-off2-seed1

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the not normalized side
- **Change:** Seed 1 of 3, normalize_reward off with collision_reward -2, 40k steps
- **Result:** mean_reward=13.99 +/- 5.64, mean_steps=33.2/40, crash_rate=44%, mean_speed=29.5m/s, mean_distance=979m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_124242_dqn_40k-norm2-seed2

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the normalized side
- **Change:** Seed 2 of 3, normalized reward with collision_reward -2, 40k steps
- **Result:** mean_reward=31.23 +/- 9.42, mean_steps=35.4/40, crash_rate=20%, mean_speed=23.5m/s, mean_distance=832m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_130435_dqn_40k-off2-seed2

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the not normalized side
- **Change:** Seed 2 of 3, normalize_reward off with collision_reward -2, 40k steps
- **Result:** mean_reward=11.93 +/- 6.10, mean_steps=30.2/40, crash_rate=52%, mean_speed=28.9m/s, mean_distance=873m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_132555_dqn_40k-norm2-seed3

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the normalized side
- **Change:** Seed 3 of 3, normalized reward with collision_reward -2, 40k steps
- **Result:** mean_reward=32.39 +/- 6.63, mean_steps=37.9/40, crash_rate=8%, mean_speed=22.2m/s, mean_distance=840m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_134724_dqn_40k-off2-seed3

- **Hypothesis:** Normalizing on or off, with a -2 crash, at 40k steps over seeds 1 to 3: this is the not normalized side
- **Change:** Seed 3 of 3, normalize_reward off with collision_reward -2, 40k steps
- **Result:** mean_reward=13.63 +/- 6.05, mean_steps=32.8/40, crash_rate=39%, mean_speed=29.3m/s, mean_distance=964m (200 episodes)
- **Conclusion:** See the 40k normalizing comparison at the top.

### 20261108_142659_dqn_320k-penalty2-lr25-resumed

- **Hypothesis:** Training 80k more steps at the same 2.5e-4 cuts crashes below 240k-penalty2-lr25's 14%, unlike halving the learning rate again (320k-penalty2-lr125, 18%)
- **Change:** 80,000 more steps on 240k-penalty2-lr25's model.zip with its saved replay buffer, same learning_rate 2.5e-4, no-normalizing -2 crash reward and gamma 0.8 (resumed from runs/20261107_161057_dqn_240k-penalty2-lr25-resumed/model.zip)
- **Result:** mean_reward=16.14 +/- 5.45, mean_steps=35.5/40, crash_rate=24%, mean_speed=29.7m/s, mean_distance=1056m (200 episodes)
- **Conclusion:** Not supported: 24% and 1,056 m against 14% and 1,102 m. With 320k-penalty2-lr125 at 18% and 1,075 m from the same starting model, neither learning rate improved on 240k-penalty2-lr25, so the chain has levelled off rather than being held back by the learning rate. Mid-training evals stayed between 12% and 35% without a trend. The 48-episode score of 17% and 1,135 m was again too kind. One run.

### 20261108_152042_dqn_80k-norm2-seed1-resumed

- **Hypothesis:** With more training, normalized runs speed up from about 22 m/s and close the gap to not normalizing's 939 m mean at 40k
- **Change:** 40,000 more steps on 40k-norm2-seed1's model.zip with its saved replay buffer, same normalized -2 crash reward, learning rate and seed (resumed from runs/20261108_115946_dqn_40k-norm2-seed1/model.zip)
- **Result:** mean_reward=34.51 +/- 5.48, mean_steps=38.4/40, crash_rate=8%, mean_speed=24.2m/s, mean_distance=931m (48 episodes)
- **Conclusion:** TODO
