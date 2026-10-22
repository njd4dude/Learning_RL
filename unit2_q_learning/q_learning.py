"""The Q-Learning algorithm itself: table, policies, training, and evaluation loops.

This is the environment-agnostic core shared by FrozenLake and Taxi — both
are small discrete-state/discrete-action environments, so one Q-table shape
`(state_space, action_space)` and one training loop cover both.
"""

import random

import numpy as np
from tqdm import tqdm


def initialize_q_table(state_space: int, action_space: int) -> np.ndarray:
    """Create a Q-table of zeros with one row per state and column per action."""
    return np.zeros((state_space, action_space))


def greedy_policy(qtable: np.ndarray, state: int) -> int:
    """Exploitation: take the action with the highest Q-value for this state."""
    return int(np.argmax(qtable[state][:]))


def epsilon_greedy_policy(qtable: np.ndarray, state: int, epsilon: float, env) -> int:
    """With probability epsilon explore randomly, otherwise exploit the Q-table."""
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(qtable, state)
    return env.action_space.sample()


def _greedy_eval_mean(
    env,
    qtable: np.ndarray,
    max_steps: int,
    n_episodes: int,
    seed: list[int] | None = None,
) -> float:
    """Quiet mean reward under the greedy policy (used for mid-training curves).

    If `seed` is provided, episode i uses `seed[i]` so mid-train checks are
    comparable across the run (and across experiments when the list is fixed).
    """
    rewards = []
    for i in range(n_episodes):
        if seed:
            state, _info = env.reset(seed=seed[i])
        else:
            state, _info = env.reset()
        total = 0.0
        for _step in range(max_steps):
            action = greedy_policy(qtable, state)
            state, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        rewards.append(total)
    return float(np.mean(rewards))


def train(
    env,
    qtable: np.ndarray,
    n_training_episodes: int,
    learning_rate: float,
    gamma: float,
    max_steps: int,
    max_epsilon: float,
    min_epsilon: float,
    decay_rate: float,
    min_learning_rate: float | None = None,
    lr_decay_rate: float = 0.0,
    train_eval_every: int = 0,
    n_train_eval_episodes: int = 50,
    eval_seed: list[int] | None = None,
    tb_writer=None,
    tb_log_every: int = 1,
) -> tuple[np.ndarray, dict]:
    """Run the Q-Learning update loop and return `(qtable, history)`.

    Each episode decays epsilon (less exploration as training progresses).
    Learning rate can stay fixed, or decay the same way toward
    `min_learning_rate` when `lr_decay_rate > 0`:
    `lr = min_lr + (max_lr - min_lr) * exp(-lr_decay_rate * episode)`.

    `history` records per-episode reward/length/epsilon/lr, plus optional
    periodic greedy-eval scores when `train_eval_every > 0`.
    If `tb_writer` is a TensorBoard SummaryWriter, scalars are logged live
    every `tb_log_every` episodes (and on every greedy eval).
    """
    max_lr = learning_rate
    min_lr = learning_rate if min_learning_rate is None else min_learning_rate
    tb_log_every = max(1, int(tb_log_every))

    history = {
        "episode_reward": [],
        "episode_length": [],
        "epsilon": [],
        "learning_rate": [],
        "eval_episode": [],
        "eval_mean_reward": [],
    }

    try:
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            if lr_decay_rate > 0 and min_lr < max_lr:
                lr = min_lr + (max_lr - min_lr) * np.exp(-lr_decay_rate * episode)
            else:
                lr = max_lr

            state, _info = env.reset()
            total_reward = 0.0
            steps = 0

            for _step in range(max_steps):
                action = epsilon_greedy_policy(qtable, state, epsilon, env)
                new_state, reward, terminated, truncated, _info = env.step(action)

                qtable[state][action] = qtable[state][action] + lr * (
                    reward + gamma * np.max(qtable[new_state]) - qtable[state][action]
                )

                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
                state = new_state

            history["episode_reward"].append(total_reward)
            history["episode_length"].append(steps)
            history["epsilon"].append(float(epsilon))
            history["learning_rate"].append(float(lr))

            if tb_writer is not None and (episode + 1) % tb_log_every == 0:
                tb_writer.add_scalar("train/episode_reward", total_reward, episode)
                tb_writer.add_scalar("train/episode_length", steps, episode)
                tb_writer.add_scalar("train/epsilon", float(epsilon), episode)
                tb_writer.add_scalar("train/learning_rate", float(lr), episode)

            if train_eval_every > 0 and (episode + 1) % train_eval_every == 0:
                mid_seeds = None
                if eval_seed:
                    mid_seeds = eval_seed[:n_train_eval_episodes]
                eval_mean = _greedy_eval_mean(
                    env, qtable, max_steps, n_train_eval_episodes, seed=mid_seeds
                )
                history["eval_episode"].append(episode)
                history["eval_mean_reward"].append(eval_mean)
                if tb_writer is not None:
                    tb_writer.add_scalar("eval/mean_reward", eval_mean, episode)
                    tb_writer.flush()
    except KeyboardInterrupt:
        # Return whatever we have so train.py can still pickle / plot / log.
        done = len(history["episode_reward"])
        print(
            f"\nInterrupted after {done}/{n_training_episodes} episodes "
            "— returning partial Q-table."
        )

    if tb_writer is not None:
        tb_writer.flush()

    return qtable, history


def evaluate_agent(env, max_steps: int, n_eval_episodes: int, qtable: np.ndarray, seed: list[int]):
    """Evaluate the agent for `n_eval_episodes` and return (mean_reward, std_reward).

    `seed` is an optional list of per-episode reset seeds (used by Taxi so
    every agent is compared from the same starting states); pass an empty
    list to reset without a fixed seed.
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, _info = env.reset(seed=seed[episode])
        else:
            state, _info = env.reset()

        total_reward = 0
        for _step in range(max_steps):
            action = greedy_policy(qtable, state)
            new_state, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))
