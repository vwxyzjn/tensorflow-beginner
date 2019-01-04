import gym
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import random

# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 0.1  # learning rate
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.05
GAMMA = 0.9  # discount factor
MAX_LEARNING_RATE_DECAY_DURATION = 25
MAX_EXPLORATION_RATE_DECAY_DURATION = 25

# Training parameters
SEED = 1
NUM_EPISODES = 3000
MAX_NUM_STEPS = 200

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make("Taxi-v2")
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)

# Build the q-network
with tf.name_scope("q_network"):
    observation = tf.placeholder(tf.float32, [None, 1])
    actions = tf.placeholder(tf.float32, [None, env.action_space.n])
    pred = tf.placeholder(tf.float32)
    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(
            inputs=observation,
            num_outputs=10,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=env.action_space.n,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )

    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(
            inputs=fc2,
            num_outputs=env.action_space.n,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("q_value"):
        q_value = tf.math.reduce_max(action_distribution)

    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(q_value, pred)

    with tf.name_scope("train"):
        train_opt = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)

# functions for graduatlly decrease learning rate
def get_explore_rate(t):
    return max(
        EPSILON_MIN,
        min(EPSILON_MAX, 1.0 - math.log10((t + 1) / MAX_LEARNING_RATE_DECAY_DURATION)),
    )


# Start the training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_rewards = []
    for i_episode in range(NUM_EPISODES):
        epsilon = get_explore_rate(i_episode)
        raw_state = env.reset()
        done = False
        episode_reward = 0
        for t in range(MAX_NUM_STEPS):
            # env.render()
            q_values = sess.run(q_value, feed_dict={observation: [[raw_state]]})
            if random.random() < epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                # Crucial!!! If there are multiple actions with the same q-value, randomly select one.
                action_probability_distribution = sess.run(
                    action_distribution, feed_dict={observation: [[raw_state]]}
                )
                action = np.argmax(action_probability_distribution)
            old_raw_state = raw_state
            raw_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Predict
            if done:
                y = reward
            else:
                target_q_values = sess.run(
                    q_value, feed_dict={observation: [[raw_state]]}
                )
                y = reward + GAMMA * target_q_values

            # Train
            loss_, _ = sess.run(
                [loss, train_opt], feed_dict={observation: [[old_raw_state]], pred: y}
            )

        if i_episode % 20 == 0:
            print("Episode: ", i_episode, "finished with rewards of ", episode_reward)
            episode_rewards += [episode_reward]

    plt.plot(episode_rewards)
