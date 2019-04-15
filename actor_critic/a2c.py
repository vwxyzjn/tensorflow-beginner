"""
Author: Costa Huang
This is an attempted reproduction of the A3C paper's individual thread.
https://arxiv.org/pdf/1602.01783.pdf
"""
import sys
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

# Hyperparameters
learning_rate = 1e-4
learning_rate_state = 1e-4
gamma = 0.98
seed = 1
num_episodes = 1000

# Set up the env
env = gym.make("CartPole-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
env.seed(seed)

# Constant
state_idx = 0
action_idx = 1
next_state_idx = 2
reward_idx = 3
done_int_idx = 4
state_value_idx = 5

# Common placeholders
obs_ph = tf.placeholder(
    shape=(None, ) + env.observation_space.shape, dtype=tf.float64)
R_ph = tf.placeholder(shape=(None), dtype=tf.float64)

# Policy gradient
fc1 = tf.layers.dense(inputs=obs_ph, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=64)
fc3 = tf.layers.dense(inputs=fc2, units=env.action_space.n)
action_probs = tf.nn.softmax(fc3)
action_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
action_probs_chosen = tf.gather_nd(action_probs,
                                   action_probs_chosen_indices_ph)

# state value function
sfc1 = tf.layers.dense(inputs=obs_ph, units=64)
sfc2 = tf.layers.dense(inputs=sfc1, units=64)
state_value = tf.layers.dense(inputs=sfc2, units=1)
state_value_ph = tf.placeholder(tf.float64)

# train
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    tf.reduce_mean(-tf.log(action_probs_chosen) * (R_ph - state_value_ph)))
strain_op = tf.train.GradientDescentOptimizer(learning_rate_state).minimize(
    (R_ph - tf.reduce_mean(state_value)) ** 2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_rewards = []
for i_episode in range(num_episodes):
    state = env.reset()
    episode_replays = []
    rewards = []
    # One step in the environment
    for t in range(200):

        # Take a step
        evaluated_action_probs, evaluated_state_value = sess.run(
            [action_probs, state_value], feed_dict={obs_ph: [state]})
        action = np.random.choice(
            np.arange(len(evaluated_action_probs[0])),
            p=evaluated_action_probs[0])
        next_state, reward, done, _ = env.step(action)

        # Keep track of the transition
        if done:
            done_int = 1
        else:
            done_int = 0
        episode_replays += [[state, action, next_state,
                             reward, done_int, evaluated_state_value]]
        rewards += [reward]

        if done:
            break

        state = next_state

    if i_episode % 10 == 0:
        print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
        episode_rewards += [sum(rewards)]

    # Go through the episode and make policy updates
    if episode_replays[-1][done_int_idx] == 1:
        R = 0
    else:
        R = sess.run(state_value, feed_dict={
                     obs_ph: episode_replays[-1][next_state_idx]})
    for t in range(len(rewards) - 1, -1, -1):
        R = episode_replays[t][reward_idx] + gamma * R
        sess.run(
            [train_op, strain_op],
            feed_dict={
                obs_ph: [episode_replays[t][state_idx]],
                action_probs_chosen_indices_ph: list(
                    enumerate([episode_replays[t][action_idx]])),
                R_ph: R,
                state_value_ph: episode_replays[t][state_idx]
            },
        )

plt.plot(episode_rewards)
