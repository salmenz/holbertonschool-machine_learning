#!/usr/bin/env python3
"""implement a full training for CartPole-v1"""
from policy_gradient import policy, softmax_grad
import numpy as np
import matplotlib.pyplot as plt


def policy_gradient(state, prb, action):
    """compute the Monte-Carlo policy gradient"""
    softmax = softmax_grad(prb)[action, :]
    log = softmax / prb[0, action]
    gradient = state.T.dot(log[None, :])
    return gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """implement a full training for CartPole-v1"""
    weights = np.random.rand(4, 2)
    n = env.action_space.n
    episode_rewards = []
    for ep in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and not ep % 1000:
                env.render()
            probs = policy(state, weights)
            action = np.random.choice(n, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]
            grad = policy_gradient(state, probs, action)
            grads.append(grad)
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = next_state
            if done:
                break

        for i in range(len(grads)):
            s = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            weights += alpha * grads[i] * s

        episode_rewards.append(score)
        print("EP: " + str(ep) + " Score: " + str(score) + "        ",
              end="\r", flush=False)

    return episode_rewards
