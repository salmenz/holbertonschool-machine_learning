#!/usr/bin/env python3
"""performs SARSA(λ)"""
import numpy as np


def greedy(state, Q, epsilon):
    """epsilon-greedy function"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return(action)


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs SARSA(λ)"""
    e = epsilon
    x = np.zeros((Q.shape))
    for ep in range(episodes):
        state = env.reset()
        action = greedy(state, Q, epsilon=epsilon)
        for _ in range(max_steps):
            x *= lambtha * gamma
            x[state, action] += 1.0
            new_s, reward, done, _ = env.step(action)
            new_action = greedy(new_s, Q, epsilon=epsilon)
            delta_t = reward + gamma * Q[new_s, new_action] - Q[state, action]
            Q[state, action] += alpha * delta_t * x[state, action]
            if done:
                break
            state = new_s
            action = new_action
        epsilon = min_epsilon + (e - min_epsilon) * np.exp(-epsilon_decay * ep)
    return Q
