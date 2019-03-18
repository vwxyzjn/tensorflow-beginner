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
learning_rate_state = 1e-3
gamma = 0.97
seed = 0
num_episodes = 5000

# Set up the env
env = gym.make("CartPole-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
env.seed(seed)

# Set up the neural network
obs_ph = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float64)
action_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
delta_I_ph = tf.placeholder(tf.float64)

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

# Calculate delta
obs_tp1_ph = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float64)
rew_ph = tf.placeholder(tf.float64)
done_ph = tf.placeholder(tf.float64)
state_value_tp1 = state_value_func(obs_tp1_ph)
delta = tf.reduce_mean(rew_ph + gamma * state_value_tp1 * (1.0 - done_ph) - state_value_t)

# train both
delta_ph = tf.placeholder(tf.float64)
I_ph = tf.placeholder(tf.float64)
temp = tf.reduce_mean(tf.log(action_probs_chosen) * delta_ph * I_ph)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-temp)

temp1 = tf.reduce_mean(state_value_t * delta_ph * I_ph)
strain_op = tf.train.GradientDescentOptimizer(learning_rate_state).minimize(-temp1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_rewards = []
with np.errstate(all="raise"):
    for i_episode in range(num_episodes):
        state = env.reset()
        I = 1
        rewards = []
        for t in range(200):
            # perform action
            evaluated_action_probs = sess.run(action_probs, feed_dict={obs_ph: [state]})
            action = np.random.choice(
                np.arange(len(evaluated_action_probs[0])), p=evaluated_action_probs[0]
            )
            next_state, reward, done, _ = env.step(action)
            rewards += [reward]

            # train
            if done:
                done_int = 1
            else:
                done_int = 0

            evaluated_delta = sess.run(
                delta,
                feed_dict={
                    obs_ph: [state],
                    obs_tp1_ph: [next_state],
                    rew_ph: float(reward),
                    done_ph: float(done_int),
                },
            )
            sess.run(
                [train_op, strain_op],
                feed_dict={
                    obs_ph: [state],
                    action_probs_chosen_indices_ph: list(enumerate([action])),
                    delta_ph: evaluated_delta,
                    I_ph: I
                },
            )
            I = gamma * I
            state = next_state
            if done:
                break

        if i_episode % 10 == 0:
            print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
            episode_rewards += [sum(rewards)]

    plt.plot(episode_rewards)
