import gym
import numpy as np
import matplotlib.pyplot as plt
import random

# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.1

# Training parameters
SEED = 2
NUM_EPISODES = 100000
MAX_NUM_STEPS = 200

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make("Taxi-v2").env
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)

# Initialize the q_table
q_table = np.zeros((env.observation_space.n,) + (env.action_space.n,))

# Start the training process
finished_episodes_count = 0
episode_rewards = []
for i_episode in range(NUM_EPISODES):
    raw_state = env.reset()
    done = False
    episode_reward = 0
    for t in range(MAX_NUM_STEPS):
        # env.render()
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[raw_state])
        old_raw_state = raw_state
        raw_state, reward, done, info = env.step(action)

        # Give a super reward
        if reward == 20:
            finished_episodes_count += 1

        episode_reward += reward

        # Update the q-table
        old_value = q_table[old_raw_state, action]
        next_max = np.max(q_table[raw_state])

        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[old_raw_state, action] = new_value
        
        
#        oldv = q_table[(old_raw_state, action)]
#        q_table[old_raw_state, action] = (1 - ALPHA) * oldv + ALPHA * (
#            reward + GAMMA * np.argmax(q_table[raw_state])
#        )
        if done:
            break

    if i_episode % 100 == 0:
        print("Episode: ", i_episode, "finished with rewards of ", episode_reward, "with successful drop-offs of", finished_episodes_count)
        episode_rewards += [episode_reward]

plt.plot(episode_rewards)
