#!/usr/bin/env python3
"""monte carlo algorithm"""
import numpy as np


def play_episode(env, policy, max_steps):
    """play episode"""
    episode = [[], []]
    state = env.reset()
    for _ in range(max_steps):
        action = policy(state)
        new_state = env.step(action)[0]
        episode[0].append(state)
        c = env.desc.reshape(env.observation_space.n)
        if c[new_state] == b'H':
            episode[1].append(-1)
            return episode
        if c[new_state] == b'G':
            episode[1].append(1)
            return episode
        episode[1].append(0)
        state = new_state
    return episode


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """monte carlo algorithm"""
    n = env.observation_space.n
    D = [gamma ** i for i in range(max_steps)]
    for _ in range(episodes):
        episode = play_episode(env, policy, max_steps)
        for i in range(len(episode[0])):
            x = episode[1][i:]
            y = episode[0][i]
            G = np.sum(np.array(x) * np.array(D[:len(x)]))
            V[y] += alpha * (G - V[y])
    return V
