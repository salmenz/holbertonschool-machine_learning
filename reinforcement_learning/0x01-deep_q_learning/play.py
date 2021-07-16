#!/usr/bin/env python3
"""Script that can display a game played by the agent trained by train.py"""
import gym
import h5py
import keras as K
from keras.optimizers import Adam
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
create_q_model = __import__('train').create_q_model


env = gym.make('Breakout-v0')
state = env.reset()
actions = env.action_space.n
model = K.models.load_model('policy.h5')
memory = SequentialMemory(limit=1000000, window_length=4)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=actions, memory=memory,
               policy=policy)
dqn.compile(optimizer=Adam(lr=.00025, clipnorm=1.0), metrics=['mae'])
dqn.test(env, nb_episodes=10, visualize=True)
