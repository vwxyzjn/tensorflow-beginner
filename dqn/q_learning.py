# https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# https://medium.freecodecamp.org/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc
# https://towardsdatascience.com/reinforcement-learning-with-python-8ef0242a2fa2

import gym
import math
import random
import numpy as np
from itertools import product

## Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
SEED = 1
NUM_EPISODES = 10000
PARTITION_STATE_NUMBER = 4
ALPHA = 0.1  # learning rate
GAMMA = 0.6  # discount factor
EPSILON = 0.1

## Initialize env
env = gym.make("CartPole-v0")
random.seed(SEED)
env.seed(SEED)

## State preprocessing
# State buckets based on https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
bucket_cart_position = np.linspace(-4.8, 4.8, PARTITION_STATE_NUMBER)
bucket_cart_velocity = np.linspace(-10, 10, PARTITION_STATE_NUMBER)
bucket_cart_pole_angle = np.linspace(-0.418, 0.418, PARTITION_STATE_NUMBER)
bucket_cart_velocity_at_tip = np.linspace(-10, 10, PARTITION_STATE_NUMBER)

# All possible states
all_states = {}
cartesian_produc = product(*[range(4) for _ in range(4)])
for index, item in enumerate(list(cartesian_produc)):
    all_states[item] = index


def process_state(raw_state):
    return all_states[
        (
            np.digitize([raw_state[0]], bucket_cart_position)[0],
            np.digitize([raw_state[1]], bucket_cart_velocity)[0],
            np.digitize([raw_state[2]], bucket_cart_pole_angle)[0],
            np.digitize([raw_state[3]], bucket_cart_velocity_at_tip)[0],
        )
    ]


## Initialize Q-table
q_table = np.zeros([len(all_states.values()), env.action_space.n])

## Start the env
for i_episode in range(NUM_EPISODES):
    raw_state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[process_state(raw_state)])
        old_raw_state = raw_state
        raw_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Update the q-table
        q_table[process_state(old_raw_state), action] = (1 - ALPHA) * q_table[
            process_state(old_raw_state), action
        ] + ALPHA * (reward + GAMMA * np.argmax(q_table[process_state(raw_state)]))

    if i_episode % 100 == 0:
        print("Episode: ", i_episode, "finished with rewards of ", episode_reward)
