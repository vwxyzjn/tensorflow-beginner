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
learning_rate = 1e-4
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
gamma_t_ph = tf.placeholder(shape=(None), dtype=tf.float64)
delta_t_ph = tf.placeholder(tf.float64, [None], name="delta")

# Set up the policy parameterization
fc1 = tf.layers.dense(inputs=obs_ph, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=64)
fc3 = tf.layers.dense(inputs=fc2, units=env.action_space.n)
action_probs = tf.nn.softmax(fc3)  # This is the pi(a|s, theta)
action_probs_chosen = tf.gather_nd(action_probs, action_probs_chosen_indices_ph)

# Set up the state-value function parameterization
sfc1 = tf.layers.dense(inputs=obs_ph, units=64)
sfc2 = tf.layers.dense(inputs=sfc1, units=64)
state_value_t = tf.layers.dense(inputs=sfc2, units=1)
# state_value_tp1 = state_value_func(obs_tp1_ph)

# train both
temp = tf.reduce_mean(tf.log(action_probs_chosen) * gamma_t_ph * delta_t_ph)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-temp)

temp1 = tf.reduce_mean(delta_t_ph)
strain_op = tf.train.GradientDescentOptimizer(learning_rate_state).minimize(tf.square(temp1))

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
    
            # Take a step
            evaluated_action_probs = sess.run(action_probs, feed_dict={obs_ph: [state]})
            action = np.random.choice(
                np.arange(len(evaluated_action_probs[0])), p=evaluated_action_probs[0]
            )
            next_state, reward, done, _ = env.step(action)
            if done: reward = -10
    
            # Keep track of the transition
            states += [state]
            actions_taken += [action]
            rewards += [reward]
            if done:
                dones += [1]
            else:
                dones += [0]
    
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
            if t == len(rewards)-2:
                break
            
            sv_t = sess.run(state_value_t, feed_dict={obs_ph: [states[t]]})
            sv_tp1 = sess.run(state_value_t, feed_dict={obs_ph: [states[t+1]]})
            delta = rewards[t] + gamma * sv_tp1 * (1-dones[t]) - sv_t
            sess.run(
                [train_op, strain_op],
                feed_dict={
                    obs_ph: [states[t]],
                    action_probs_chosen_indices_ph: list(enumerate([actions_taken[t]])),
                    delta_t_ph: delta.reshape(1,),
                    gamma_t_ph: I
                }
            )     
            I = gamma * I
            # print(I, gamma ** t)
    
    
    plt.plot(episode_rewards)
