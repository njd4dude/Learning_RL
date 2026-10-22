"""Quick benchmark: single-env vs SyncVectorEnv Q-learning on slippery FrozenLake.

Not wired into the main train path — just answers "does parallel help here?"

    python bench_parallel.py
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv

ENV_KWARGS = dict(map_name="4x4", is_slippery=True)
TOTAL_EPISODES = 50_000
MAX_STEPS = 99
LR = 0.1
GAMMA = 0.95
MAX_EPS = 1.0
MIN_EPS = 0.05
DECAY = 0.00005
N_ENVS_LIST = [1, 4, 8, 16]


def _make_env():
    return gym.make("FrozenLake-v1", **ENV_KWARGS)


def _epsilon(episode: int) -> float:
    return MIN_EPS + (MAX_EPS - MIN_EPS) * np.exp(-DECAY * episode)


def train_single(n_episodes: int) -> tuple[np.ndarray, float]:
    env = _make_env()
    q = np.zeros((env.observation_space.n, env.action_space.n))
    t0 = time.perf_counter()
    episodes_done = 0
    while episodes_done < n_episodes:
        eps = _epsilon(episodes_done)
        state, _ = env.reset()
        for _ in range(MAX_STEPS):
            if np.random.random() > eps:
                action = int(np.argmax(q[state]))
            else:
                action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action)
            q[state, action] += LR * (reward + GAMMA * np.max(q[new_state]) - q[state, action])
            state = new_state
            if terminated or truncated:
                break
        episodes_done += 1
    elapsed = time.perf_counter() - t0
    env.close()
    return q, elapsed


def train_vector(n_episodes: int, n_envs: int) -> tuple[np.ndarray, float]:
    """Run until ~n_episodes completed across all envs combined."""
    envs = SyncVectorEnv([_make_env for _ in range(n_envs)])
    q = np.zeros((envs.single_observation_space.n, envs.single_action_space.n))
    states, _ = envs.reset()
    episodes_done = 0
    # Approximate episode index for epsilon from total completed episodes.
    t0 = time.perf_counter()
    while episodes_done < n_episodes:
        eps = _epsilon(episodes_done)
        actions = np.empty(n_envs, dtype=np.int64)
        for i, state in enumerate(states):
            if np.random.random() > eps:
                actions[i] = int(np.argmax(q[int(state)]))
            else:
                actions[i] = envs.single_action_space.sample()

        new_states, rewards, terminations, truncations, _ = envs.step(actions)
        dones = terminations | truncations
        for i in range(n_envs):
            s = int(states[i])
            a = int(actions[i])
            ns = int(new_states[i])
            q[s, a] += LR * (rewards[i] + GAMMA * np.max(q[ns]) - q[s, a])
            if dones[i]:
                episodes_done += 1
        states = new_states
    elapsed = time.perf_counter() - t0
    envs.close()
    return q, elapsed


def main():
    print(f"Benchmark: {TOTAL_EPISODES} episodes, slippery 4x4 FrozenLake")
    print(f"{'n_envs':>8}  {'seconds':>10}  {'eps/sec':>10}  {'speedup':>8}")
    baseline = None
    for n_envs in N_ENVS_LIST:
        if n_envs == 1:
            _, elapsed = train_single(TOTAL_EPISODES)
        else:
            _, elapsed = train_vector(TOTAL_EPISODES, n_envs)
        rate = TOTAL_EPISODES / elapsed
        if baseline is None:
            baseline = elapsed
            speedup = 1.0
        else:
            speedup = baseline / elapsed
        print(f"{n_envs:8d}  {elapsed:10.2f}  {rate:10.0f}  {speedup:8.2f}x")


if __name__ == "__main__":
    main()
