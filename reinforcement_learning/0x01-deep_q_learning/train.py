#!/usr/bin/env python3
import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import Callback, ModelIntervalCheckpoint


class ModelIntervalCheck(Callback):
    def __init__(self, filepath, interval, verbose=0, kmodel=None):
        """save every x steps"""
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.kmodel = kmodel
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """Save weights at interval steps during training"""
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('\nStep {}: saving kmodel to {}'.format(self.total_steps,
                                                          filepath))
        self.kmodel.save(filepath)


env = gym.make("Breakout-v0")
height, width, channels = env.observation_space.shape
actions = env.action_space.n


def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu',
              input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(height, width, channels, actions)
model.summary()


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                  value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(optimizer=Adam(lr=0.00025), metrics=['mae', 'accuracy'])
callback = [ModelIntervalCheck('policy.h5', 1000, 1, model)]
dqn.fit(env, nb_steps=10000, visualize=True, verbose=2)

model.save("policy.h5")
# scores = dqn.test(env, nb_episodes=10, visualize=True)

# print(np.mean(scores.history['episode_reward']))
