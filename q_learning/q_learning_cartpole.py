# https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# https://medium.freecodecamp.org/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc
# https://towardsdatascience.com/reinforcement-learning-with-python-8ef0242a2fa2

import gym
import random
import numpy as np
import math
from itertools import product
import matplotlib.pyplot as plt

## Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
SEED = 1
n_bins_angle = 10
NUM_EPISODES = 6000
MAX_NUM_STEPS = 200
n_bins = 8
BUCKET_CART_POSITION_NUMBER = 1
BUCKET_CART_VELOCITY_NUMBER = 1
BUCKET_CART_POLE_ANGLE_NUMBER = 6
BUCKET_CART_VELOCITY_AT_TIP_NUMBER = 3
# ALPHA = 0.5  # learning rate
GAMMA = 0.9  # discount factor
# EPSILON = 0.05
min_explore_rate = 0.01
min_learning_rate = 0.1


## Initialize env
env = gym.make("CartPole-v0")
random.seed(SEED)
env.seed(SEED)

## State preprocessing
# State buckets based on https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# Please take a look at the termination condition
bucket_cart_position = np.linspace(-2.4, 2.4, BUCKET_CART_POSITION_NUMBER)
bucket_cart_velocity = np.linspace(-1, 1, BUCKET_CART_VELOCITY_NUMBER)
bucket_cart_pole_angle = np.linspace(-0.20943, 0.20943, BUCKET_CART_POLE_ANGLE_NUMBER)
bucket_cart_velocity_at_tip = np.linspace(-3.5, 3.5, BUCKET_CART_VELOCITY_AT_TIP_NUMBER)

# All possible states
all_states = {}
all_states_reverse = {}

cartesian_produc = product(
    *[
        range(BUCKET_CART_POSITION_NUMBER + 1),
        range(BUCKET_CART_VELOCITY_NUMBER + 1),
        range(BUCKET_CART_POLE_ANGLE_NUMBER + 1),
        range(BUCKET_CART_VELOCITY_AT_TIP_NUMBER + 1),
    ]
)
for index, item in enumerate(list(cartesian_produc)):
    all_states[item] = index
    all_states_reverse[index] = item


def process_state(raw_state):
    return all_states[
        (
            np.digitize([raw_state[0]], bucket_cart_position)[0],
            np.digitize([raw_state[1]], bucket_cart_velocity)[0],
            np.digitize([raw_state[2]], bucket_cart_pole_angle)[0],
            np.digitize([raw_state[3]], bucket_cart_velocity_at_tip)[0],
        )
    ]


def get_explore_rate(t):
    return max(min_explore_rate, min(1, 1.0 - math.log10((t + 1) / 300)))


def get_learning_rate(t):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((t + 1) / 300)))


## Initialize Q-table
q_table = np.zeros([len(all_states.values()), env.action_space.n])

## Start the env
episode_rewards = []
for i_episode in range(NUM_EPISODES):
    ALPHA = get_learning_rate(i_episode)
    EPSILON = get_explore_rate(i_episode)
    raw_state = env.reset()
    done = False
    episode_reward = 0
    for t in range(MAX_NUM_STEPS):
        # env.render()
        if random.random() < EPSILON:
            action = random.randint(0, 1)
        else:
            # Crucial!!! If there are multiple actions with the same q-value, randomly select one.
            max_q_value_indices = np.where(
                q_table[process_state(raw_state)]
                == q_table[process_state(raw_state)].max()
            )[0]
            action = random.choice(max_q_value_indices)
        old_raw_state = raw_state
        raw_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Update the q-table
        if done:
            reward -= 200
        # print("terminating state", process_state(old_raw_state))
        oldv = q_table[(process_state(old_raw_state), action)]
        q_table[process_state(old_raw_state), action] = (1 - ALPHA) * oldv + ALPHA * (
            reward + GAMMA * np.argmax(q_table[process_state(raw_state)])
        )
        if done:
            break

    if i_episode % 20 == 0:
        print("Episode: ", i_episode, "finished with rewards of ", episode_reward)
        episode_rewards += [episode_reward]

plt.plot(episode_rewards)
