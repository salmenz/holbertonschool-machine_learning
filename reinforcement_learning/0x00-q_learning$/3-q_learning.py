#!/usr/bin/env python3
"""Q-learning"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning"""
    R = []
    O_ep = epsilon

    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            o_v = Q[state, action]
            nxt_m = np.max(Q[new_state])
            n_v = (1 - alpha) * o_v + (alpha * (reward + gamma * nxt_m))
            Q[state, action] = n_v
            rewards += reward
            state = new_state
            if done:
                break
        epsilon = min_epsilon + (O_ep - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        R.append(rewards)
    return Q, R
