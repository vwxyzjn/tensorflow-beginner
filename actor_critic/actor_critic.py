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
learning_rate = 0.0005
learning_rate_state = 0.001
gamma = 0.99
seed = 0
num_episodes = 10000

# Set up the env
env = gym.make("CartPole-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
env.seed(seed)

# Set up the neural network
obs_ph = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float64)
action_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)

# Set up the policy parameterization
fc1 = tf.layers.dense(inputs=obs_ph, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=64)
fc3 = tf.layers.dense(inputs=fc2, units=env.action_space.n)
action_probs = tf.nn.softmax(fc3)  # This is the pi(a|s, theta)
neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3, labels=action_probs_chosen_indices_ph)

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
p_loss = tf.reduce_mean(neglogprob * delta_ph * I_ph)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(p_loss)

v_loss = tf.square(delta)
strain_op = tf.train.GradientDescentOptimizer(learning_rate_state).minimize(v_loss)

tf.summary.scalar("v_loss", v_loss)
tf.summary.scalar("p_loss", p_loss)
write_op = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./logs", sess.graph_def)

episode_rewards = []
total_timesteps = 0
with np.errstate(all="raise"):
    for i_episode in range(num_episodes):
        state = env.reset()
        I = 1
        rewards = []
        for t in range(200):
            total_timesteps += 1
            # perform action
            evaluated_action_probs = sess.run(action_probs, feed_dict={obs_ph: [state]})
            action = np.random.choice(
                np.arange(len(evaluated_action_probs[0])), p=evaluated_action_probs[0]
            )
            next_state, reward, done, _ = env.step(action)
            rewards += [reward]
            
            #if i_episode >= 3100:
                #print(sess.run([state_value_t, fc3], feed_dict={obs_ph: [state]}))

            # train
            if done:
                done_int = 1
                if t == 199:
                    reward = 20
                    print("it happened")
                else:
                    reward = -20
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
            _, _, summary = sess.run(
                [train_op, strain_op, write_op],
                feed_dict={
                    action_probs_chosen_indices_ph: [action],
                    delta_ph: evaluated_delta,
                    I_ph: I,
                    obs_ph: [state],
                    obs_tp1_ph: [next_state],
                    rew_ph: float(reward),
                    done_ph: float(done_int),
                },
            )
            writer.add_summary(summary, total_timesteps)
            I = gamma * I
            state = next_state
            if done:
                break

        if i_episode % 10 == 0:
            print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
            episode_rewards += [sum(rewards)]

    # raw plot
    plt.plot(episode_rewards)
    plt.show()
    # smooth plot
    from scipy.ndimage.filters import gaussian_filter1d
    smoothed_rewards = gaussian_filter1d(episode_rewards, sigma=2)
    plt.plot(smoothed_rewards)
    plt.show()
    print(f"average rewards is {np.mean(episode_rewards)}")