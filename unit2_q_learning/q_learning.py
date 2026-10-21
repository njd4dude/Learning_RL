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
) -> np.ndarray:
    """Run the Q-Learning update loop and return the trained table.

    Each episode decays epsilon (less exploration as training progresses)
    and applies the Bellman update
    `Q(s,a) += lr * (reward + gamma * max(Q(s')) - Q(s,a))` at every step.
    """
    for episode in tqdm(range(n_training_episodes)):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state, _info = env.reset()

        for _step in range(max_steps):
            action = epsilon_greedy_policy(qtable, state, epsilon, env)
            new_state, reward, terminated, truncated, _info = env.step(action)

            qtable[state][action] = qtable[state][action] + learning_rate * (
                reward + gamma * np.max(qtable[new_state]) - qtable[state][action]
            )

            if terminated or truncated:
                break
            state = new_state

    return qtable


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
