import gym
import tensorflow as tf
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

tf.reset_default_graph()

# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 0.005  # learning rate
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.02
GAMMA = 0.5  # discount factor
MAX_EXPLORATION_RATE_DECAY_TIMESTEP = 500
TARGET_NETWORK_UPDATE_STEP_FREQUENCY = 500
EXPERIENCER_REPLAY_BATCH_SIZE = 32

# Training parameters
SEED = 3
NUM_EPISODES = 10
MAX_NUM_STEPS = 200
TOTAL_MAX_TIMESTEPS = 5000
# we picked 2000 because on average, the random agent would make a successful drop-off after 2848.14
# timesteps according to https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

## Initialize env
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
env = gym.make(
    "Taxi-v2"
).env  # without the .env, there is gonna be a 200 max num steps.
random.seed(SEED)
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

def build_neural_network(scope: str) -> Tuple[tf.Variable]:
    with tf.variable_scope(scope):
        observation = tf.placeholder(tf.float32, [None, 1])
        pred = tf.placeholder(tf.float32, [None,])
        fc1 = tf.contrib.layers.fully_connected(
            inputs=observation,
            num_outputs=64,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
        )
        fc2 = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=64,
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
        q_value = tf.math.reduce_max(fc3, axis=1)
        loss = tf.losses.huber_loss(q_value, pred)
        train_opt = tf.train.AdamOptimizer(ALPHA).minimize(loss)
        saver = tf.train.Saver()
        tf.summary.scalar("Loss", loss)
        write_op = tf.summary.merge_all()
    return (
        fc3,
        observation,
        pred,
        action_distribution,
        q_value,
        loss,
        train_opt,
        saver,
        write_op,
    )


(
    fc3,
    observation,
    pred,
    action_distribution,
    q_value,
    loss,
    train_opt,
    saver,
    write_op,
) = build_neural_network("q_network")
(
    target_fc3,
    target_observation,
    target_pred,
    target_action_distribution,
    target_q_value,
    target_loss,
    target_train_opt,
    target_saver,
    target_write_op,
) = build_neural_network("target_network")


# Play the game
# Start the training process
with tf.Session() as sess:
    episode_rewards = []
    saver.restore(sess, "./tmp/model.ckpt")
    for i_episode in range(NUM_EPISODES):
        raw_state = env.reset()
        done = False
        episode_reward = 0
        finished_episodes_count = 0
        for t in range(MAX_NUM_STEPS):
            env.render()
            # Crucial!!! If there are multiple actions with the same q-value, randomly select one.
            evaluated_action_probability = sess.run(
                action_distribution, feed_dict={observation: [[raw_state]]}
            )
            action = np.argmax(evaluated_action_probability)
            old_raw_state = raw_state
            raw_state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                finished_episodes_count += 1
                break

        print(
            "Episode: ",
            i_episode,
            "finished with rewards of ",
            episode_reward,
            "with successful drop-offs of",
            finished_episodes_count,
        )
        episode_rewards += [episode_reward]

    plt.plot(episode_rewards)

