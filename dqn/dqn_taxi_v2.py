import gym
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

tf.reset_default_graph()

# Utility functions
def backup_training_variables(scope: str, sess: tf.Session) -> List[List]:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    vars_values: List[List] = []
    for var in variables:
        vars_values += [sess.run(var)]
    return vars_values


def restore_training_variables(
    scope: str, vars_values: List[List], sess: tf.Session
) -> None:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    for i, var in enumerate(variables):
        sess.run(tf.assign(var, vars_values[i]))


# functions for graduatlly decrease learning rate
def get_explore_rate(t):
    return max(
        EPSILON_MIN,
        min(EPSILON_MAX, 1.0 - math.log10((t + 1) / MAX_LEARNING_RATE_DECAY_DURATION)),
    )


# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 0.04  # learning rate
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.05
GAMMA = 0.4  # discount factor
MAX_LEARNING_RATE_DECAY_DURATION = 3000
MAX_EXPLORATION_RATE_DECAY_DURATION = 3000
TARGET_NETWORK_UPDATE_EPISODE_FREQUENCY = 2

# Training parameters
SEED = 1
NUM_EPISODES = 4000
MAX_NUM_STEPS = 200

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make("Taxi-v2")
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)


def build_neural_network(scope: str) -> Tuple[tf.Variable]:
    with tf.variable_scope(scope):
        observation = tf.placeholder(tf.float32, [None, 1])
        pred = tf.placeholder(tf.float32)
        fc1 = tf.contrib.layers.fully_connected(
            inputs=observation,
            num_outputs=10,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=env.action_space.n,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        fc3 = tf.contrib.layers.fully_connected(
            inputs=fc2,
            num_outputs=env.action_space.n,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        action_distribution = tf.nn.softmax(fc3)
        q_value = tf.math.reduce_max(action_distribution)
        loss = tf.losses.mean_squared_error(q_value, pred)
        train_opt = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)
    return (fc3, observation, pred, action_distribution, q_value, loss, train_opt)


(
    fc3,
    observation,
    pred,
    action_distribution,
    q_value,
    loss,
    train_opt,
) = build_neural_network("q_network")
(
    target_fc3,
    target_observation,
    target_pred,
    target_action_distribution,
    target_q_value,
    target_loss,
    target_train_opt,
) = build_neural_network("target_network")

# Start the training process
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_rewards = []
    target_network_update_counter = 0
    for i_episode in range(NUM_EPISODES):
        target_network_update_counter += 1
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
                    target_q_value, feed_dict={target_observation: [[raw_state]]}
                )
                y = reward + GAMMA * target_q_values

            # Train
            sess.run(train_opt, feed_dict={observation: [[old_raw_state]], pred: y})

        # Update the target network
        if target_network_update_counter > TARGET_NETWORK_UPDATE_EPISODE_FREQUENCY:
            restore_training_variables(
                "target_network", backup_training_variables("q_network", sess), sess
            )
            target_network_update_counter = 0

        if i_episode % 20 == 0:
            print("Episode: ", i_episode, "finished with rewards of ", episode_reward)
            episode_rewards += [episode_reward]

    plt.plot(episode_rewards)
