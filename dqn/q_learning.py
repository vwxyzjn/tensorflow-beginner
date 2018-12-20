# https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# https://medium.freecodecamp.org/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc
# https://towardsdatascience.com/reinforcement-learning-with-python-8ef0242a2fa2

import gym
import random
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

## Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
SEED = 1
NUM_EPISODES = 10000
BUCKET_CART_POSITION_NUMBER = 6
BUCKET_CART_VELOCITY_NUMBER = 3
BUCKET_CART_POLE_ANGLE_NUMBER = 6
BUCKET_CART_VELOCITY_AT_TIP_NUMBER = 3
ALPHA = 0.3  # learning rate
GAMMA = 0.6  # discount factor
EPSILON = 0.2

## Initialize env
env = gym.make("CartPole-v0")
random.seed(SEED)
env.seed(SEED)

## State preprocessing
# State buckets based on https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
bucket_cart_position = np.linspace(-4.9, 4.9, BUCKET_CART_POSITION_NUMBER)
bucket_cart_velocity = [0]
bucket_cart_pole_angle = np.linspace(-0.419, 0.419, BUCKET_CART_POLE_ANGLE_NUMBER)
bucket_cart_velocity_at_tip = [0]

# All possible states
all_states = {}
cartesian_produc = product(
    *[
        range(1, BUCKET_CART_POSITION_NUMBER),
        range(BUCKET_CART_VELOCITY_NUMBER),
        range(1, BUCKET_CART_POLE_ANGLE_NUMBER),
        range(BUCKET_CART_VELOCITY_AT_TIP_NUMBER),
    ]
)
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
episode_rewards = []
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
        episode_rewards += [episode_reward]

plt.plot(episode_rewards)
