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
| `penalty2_chain/20261105_170129_dqn_160k-penalty2-resumed` | No normalizing, -2 crash | 1057 m | 21% | 29.6 m/s | 15.84 |
| `penalty2_chain/20261105_163251_dqn_120k-penalty2-resumed` | No normalizing, -2 crash | 1049 m | 22% | 29.3 m/s | 15.59 |
| `penalty2_chain/20261105_160009_dqn_80k-penalty2-resumed` | No normalizing, -2 crash | 992 m | 32% | 29.4 m/s | 14.62 |
| `normalized_speed2_chain/20261105_215327_dqn_120k-normalized-speed2-resumed` | Normalized, -2 crash, 2 full speed | 972 m | 42% | 29.7 m/s | 31.83 |
| `normalized_speed2_chain/20261105_200605_dqn_80k-normalized-speed2-resumed` | Normalized, -2 crash, 2 full speed | 947 m | 42% | 29.6 m/s | 30.87 |
| `penalty2_chain/20261104_210951_dqn_penalty2` | No normalizing, -2 crash | 921 m | 46% | 29.2 m/s | 12.84 |
| `20261104_215246_dqn_penalty5` | No normalizing, -5 crash | 916 m | 40% | 27.9 m/s | 10.74 |
| `20261104_171828_dqn_40k` | Normalized, -1 crash | 864 m | 17% | 23.9 m/s | 29.59 |
| `normalized_penalty2_chain/20261105_172705_dqn_80k-normalized-penalty2-resumed` | Normalized, -2 crash | 839 m | 27% | 24.3 m/s | 30.60 |
| `normalized_penalty2_chain/20261105_135955_dqn_40k-normalized-penalty2` | Normalized, -2 crash | 836 m | 17% | 23.0 m/s | 31.37 |
| `20261105_153008_dqn_20k-norm2-seed2` | Normalized, -2 crash | 832 m | 2% | 21.1 m/s | 32.78 |
| `20261105_133450_dqn_40k-default-reward-repeat` | Normalized, -1 crash | 830 m | 49% | 27.1 m/s | 27.08 |
| `20261104_190015_ppo_200k` | Normalized, -1 crash | 825 m | 22% | 24.1 m/s | 28.26 |
| `20261104_161334_dqn_20k` | Normalized, -1 crash | 824 m | 20% | 23.2 m/s | 28.31 |
| `ppo_lr3e-4_chain/20261105_113120_ppo_40k-lr3e-4` | No normalizing, -5 crash | 819 m | 8% | 21.4 m/s | 4.55 |
| `20261105_144614_dqn_20k-norm1-seed2` | Normalized, -1 crash | 818 m | 6% | 21.4 m/s | 28.22 |
| `20261105_154440_dqn_20k-norm2-seed3` | Normalized, -2 crash | 800 m | 2% | 20.2 m/s | 32.78 |
| `20261105_102757_ppo_20k-lr1e-4` | No normalizing, -5 crash | 798 m | 2% | 20.1 m/s | 2.26 |
| **Always `SLOWER` (baseline, no model)** | Not trained | 795 m | 2% | 20.0 m/s | 27.85 normalized, -1; 1.96 no normalizing, -5 |
| `20261105_105138_ppo_20k-lr1e-4-default-reward` | Normalized, -1 crash | 794 m | 1% | 20.0 m/s | 27.79 |
| `20261105_111332_ppo_20k-lr3e-4` | No normalizing, -5 crash | 786 m | 8% | 20.8 m/s | 2.71 |
| `normalized_speed2_chain/20261105_193556_dqn_40k-normalized-speed2` | Normalized, -2 crash, 2 full speed | 779 m | 70% | 29.5 m/s | 25.11 |
| `20261105_150125_dqn_20k-norm1-seed3` | Normalized, -1 crash | 770 m | 18% | 21.9 m/s | 26.65 |
| `20261104_180445_dqn_40k-slow-settings` | Normalized, -1 crash | 765 m | 40% | 24.0 m/s | 25.68 |
| `ppo_lr3e-4_chain/20261105_121448_ppo_65k-lr3e-4-resumed` | No normalizing, -5 crash | 752 m | 30% | 22.3 m/s | 4.35 |
| `20261105_143119_dqn_20k-norm1-seed1` | Normalized, -1 crash | 739 m | 30% | 22.6 m/s | 25.40 |
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
