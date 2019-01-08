import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA_MAX = 0.5  # learning rate
ALPHA_MIN = 0.2
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.05
GAMMA = 0.9  # discount factor
MAX_LEARNING_RATE_DECAY_DURATION = 5000
MAX_EXPLORATION_RATE_DECAY_DURATION = 5000

# Training parameters
SEED = 2
NUM_EPISODES = 20000
MAX_NUM_STEPS = 200

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make("Taxi-v2")
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)

# Initialize the q_table
q_table = np.zeros((env.observation_space.n,) + (env.action_space.n,))

# functions for graduatlly decrease learning rate
def get_explore_rate(t):
    return max(
        EPSILON_MIN,
        min(EPSILON_MAX, 1.0 - math.log10((t + 1) / MAX_LEARNING_RATE_DECAY_DURATION)),
    )


def get_learning_rate(t):
    return max(
        ALPHA_MIN,
        min(ALPHA_MAX, 1.0 - math.log10((t + 1) / MAX_EXPLORATION_RATE_DECAY_DURATION)),
    )


# Start the training process
episode_rewards = []
for i_episode in range(NUM_EPISODES):
    alpha = get_learning_rate(i_episode)
    epsilon = get_explore_rate(i_episode)
    raw_state = env.reset()
    done = False
    episode_reward = 0
    for t in range(MAX_NUM_STEPS):
        # env.render()
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            # Crucial!!! If there are multiple actions with the same q-value, randomly select one.
            max_q_value_indices = np.where(
                q_table[raw_state] == q_table[raw_state].max()
            )[0]
            action = random.choice(max_q_value_indices)
        old_raw_state = raw_state
        raw_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Update the q-table
        oldv = q_table[(old_raw_state, action)]
        q_table[old_raw_state, action] = (1 - alpha) * oldv + alpha * (
            reward + GAMMA * np.argmax(q_table[raw_state])
        )
        if done:
            break

    if i_episode % 20 == 0:
        print("Episode: ", i_episode, "finished with rewards of ", episode_reward)
        episode_rewards += [episode_reward]

plt.plot(episode_rewards)
