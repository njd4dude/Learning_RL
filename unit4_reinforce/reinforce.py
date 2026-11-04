"""The REINFORCE (Monte Carlo policy gradient) algorithm: training and evaluation loops."""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Progress:
    """Training counters that carry over when a run is resumed."""

    episodes: int = 0
    total_steps: int = 0
    train_seconds: float = 0.0
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def fps(self) -> float:
        return self.total_steps / self.train_seconds if self.train_seconds else 0.0


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Standardized discounted return G_t for every timestep of one episode.

    Computed back to front so each G_t reuses the one after it:
    G_t = r_(t+1) + gamma * G_(t+1). `appendleft` keeps them in time order.
    """
    returns = deque()
    for reward in reversed(rewards):
        next_return = returns[0] if returns else 0.0
        returns.appendleft(gamma * next_return + reward)

    # Standardizing the returns makes training much more stable. eps keeps
    # the division safe when every return is the same.
    eps = np.finfo(np.float32).eps.item()
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + eps)


def reinforce(
    env,
    policy,
    optimizer,
    n_training_episodes: int,
    max_t: int,
    gamma: float,
    print_every: int,
    progress: Progress,
    tb_writer=None,
    eval_env=None,
    train_eval_every: int = 0,
    n_train_eval_episodes: int = 10,
    checkpoint_every: int = 0,
    save_checkpoint=None,
) -> list[float]:
    """Train `policy` for `n_training_episodes` more; return this session's rewards.

    One update per episode: play it to the end with the current policy, then
    push up the log-probability of each action in proportion to the return
    that followed it. Ctrl+C stops early and returns the scores so far.

    With an `eval_env` and `train_eval_every > 0`, the policy is also scored
    on `eval_env` every `train_eval_every` episodes and logged under eval/.

    `progress` is updated in place, so a resumed run continues its episode and
    step counts. Every `checkpoint_every` episodes, `save_checkpoint()` is called.

    Seconds cover playing and updating only, not evals or logging, so FPS
    (steps per second) measures training speed and doesn't dip at each eval.
    """
    scores_deque = progress.recent_scores
    scores = []
    first_episode = progress.episodes + 1

    try:
        for i_episode in range(first_episode, first_episode + n_training_episodes):
            episode_start = time.perf_counter()
            saved_log_probs = []
            rewards = []
            state, _info = env.reset()
            for _t in range(max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, terminated, truncated, _info = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break

            score = float(sum(rewards))
            scores_deque.append(score)
            scores.append(score)

            returns = discounted_returns(rewards, gamma)
            # PyTorch minimizes, so gradient ascent on the return is descent on -return.
            policy_loss = (-torch.cat(saved_log_probs) * returns).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            progress.train_seconds += time.perf_counter() - episode_start
            progress.total_steps += len(rewards)
            progress.episodes = i_episode

            if tb_writer is not None:
                tb_writer.add_scalar("train/episode_reward", score, i_episode)
                tb_writer.add_scalar("train/avg100_reward", np.mean(scores_deque), i_episode)
                tb_writer.add_scalar("train/episode_length", len(rewards), i_episode)
                tb_writer.add_scalar("train/policy_loss", policy_loss.item(), i_episode)
                tb_writer.add_scalar("time/fps", progress.fps, i_episode)
                tb_writer.add_scalar("time/total_steps", progress.total_steps, i_episode)
                # Same curve, x-axis in steps: episodes get longer as the agent
                # improves, so this is the one that shows sample efficiency.
                tb_writer.add_scalar(
                    "train/avg100_reward_by_step", np.mean(scores_deque), progress.total_steps
                )

            if eval_env is not None and train_eval_every > 0 and i_episode % train_eval_every == 0:
                eval_mean, eval_std = evaluate_agent(eval_env, max_t, n_train_eval_episodes, policy)
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/mean_reward", eval_mean, i_episode)
                    tb_writer.add_scalar("eval/std_reward", eval_std, i_episode)

            if i_episode % print_every == 0:
                print(
                    f"Episode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}"
                    f"\tSteps: {progress.total_steps}\tFPS: {progress.fps:.0f}"
                )

            if save_checkpoint is not None and checkpoint_every > 0 and i_episode % checkpoint_every == 0:
                save_checkpoint()
    except KeyboardInterrupt:
        print(
            f"\nInterrupted after {len(scores)}/{n_training_episodes} episodes this session "
            "— returning the partially trained policy."
        )

    return scores


def evaluate_agent(env, max_steps: int, n_eval_episodes: int, policy) -> tuple[float, float]:
    """Play `n_eval_episodes` with the (stochastic) policy; return (mean, std) reward."""
    episode_rewards = []
    with torch.no_grad():
        for _episode in range(n_eval_episodes):
            state, _info = env.reset()
            total_reward = 0.0
            for _step in range(max_steps):
                action, _ = policy.act(state)
                state, reward, terminated, truncated, _info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))
