"""
Author: Costa Huang
This is an attempted reproduction of the REINFORCE algorithm in Sutton & Barto's book.
http://incompleteideas.net/book/bookdraft2017nov5.pdf#page=289
"""

import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

# Hyperparameters
learning_rate = 1e-5
learning_rate_state = 1e-4
gamma = 0.99
seed = 0
num_episodes = 5000

# Set up the env
env = gym.make("CartPole-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
env.seed(seed)

# Set up the neural network
obs_ph = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float64)
obs_tp1_ph = tf.placeholder(
    shape=(None,) + env.observation_space.shape, dtype=tf.float64
)
action_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
I_t_ph = tf.placeholder(shape=(None), dtype=tf.float64)
done_ph = tf.placeholder(tf.float64)
rew_ph = tf.placeholder(shape=(None), dtype=tf.float64)

# Set up the policy parameterization
fc1 = tf.layers.dense(inputs=obs_ph, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=64)
fc3 = tf.layers.dense(inputs=fc2, units=env.action_space.n)
action_probs = tf.nn.softmax(fc3)  # This is the pi(a|s, theta)
action_probs_chosen = tf.gather_nd(action_probs, action_probs_chosen_indices_ph)

# Set up the state-value function parameterization
def _state_value(x):
    sfc1 = tf.layers.dense(inputs=x, units=64)
    sfc2 = tf.layers.dense(inputs=sfc1, units=64)
    sfc3 = tf.layers.dense(inputs=sfc2, units=1)
    return sfc3


state_value_func = tf.make_template("state_value_func", _state_value)
state_value_t = state_value_func(obs_ph)
state_value_tp1 = state_value_func(obs_tp1_ph)

# calculate td errors
delta = rew_ph + gamma * state_value_tp1 * (1.0 - done_ph) - state_value_t

# train both
temp = tf.reduce_mean(tf.log(action_probs_chosen) * I_t_ph * delta)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-temp)

temp1 = tf.reduce_mean(state_value_t * I_t_ph * delta)
strain_op = tf.train.GradientDescentOptimizer(learning_rate_state).minimize(
    tf.square(temp1)
)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_rewards = []
with np.errstate(all="raise"):
    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []
        states = []
        actions_taken = []
        rewards = []
        dones = []
        # One step in the environment
        for t in range(200):
#            if i_episode >= 1000:
#                env.render()
            # Take a step
            evaluated_action_probs = sess.run(action_probs, feed_dict={obs_ph: [state]})
            action = np.random.choice(
                np.arange(len(evaluated_action_probs[0])), p=evaluated_action_probs[0]
            )
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -20
            # Keep track of the transition
            states += [state]
            actions_taken += [action]
            rewards += [reward]
            if done:
                dones += [1.0]
            else:
                dones += [0.0]
            if done:
                break

            state = next_state

        if i_episode % 10 == 0:
            print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
            episode_rewards += [sum(rewards)]

        # Go through the episode and make policy updates
        I = 1
        for t, item in enumerate(rewards):
            # The return after this timestep
            if t == len(rewards) - 2:
                break
            sess.run(
                [train_op, strain_op],
                feed_dict={
                    obs_ph: [states[t]],
                    obs_tp1_ph: [states[t + 1]],
                    action_probs_chosen_indices_ph: list(enumerate([actions_taken[t]])),
                    I_t_ph: I,
                    done_ph: dones[t],
                    rew_ph: rewards[t],
                },
            )
            I = gamma * I
            # print(I, gamma ** t)

    plt.plot(episode_rewards)
