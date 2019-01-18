import gym
import tensorflow as tf
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

tf.reset_default_graph()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
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
NUM_EPISODES = 1000
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
# env.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

def build_neural_network(scope: str) -> Tuple[tf.Variable]:
    with tf.variable_scope(scope):
        # Because this is discrete(500) observation space, we actually need to use the one-hot
        # tensor to make training easier.
        # https://github.com/hill-a/stable-baselines/blob/a6f7459a301a7ba3c4bbcebff5829ea054ae802f/stable_baselines/common/input.py#L20
        # So, instead of 
        # observation = tf.placeholder(tf.float32, [None, 1], name="observation")
        # We use
        observation = tf.placeholder(shape=(None,), dtype=tf.int32)
        processed_observations = tf.to_float(tf.one_hot(observation, env.observation_space.n))
        pred = tf.placeholder(tf.float32, [None], name="pred")
        q_value_index = tf.placeholder(tf.int32, [None], name="q_value_index")
        fc1 = tf.contrib.layers.fully_connected(
            inputs=processed_observations,
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
        max_q_value = tf.math.reduce_max(fc3, axis=1)
        # https://github.com/hill-a/stable-baselines/blob/88a5c5d50a7f6ad1f44f6ef0feaa0647ed2f7298/stable_baselines/deepq/build_graph.py#L394
        q_value = tf.reduce_sum(fc3 * tf.one_hot(q_value_index, env.action_space.n), axis=1)
        loss = tf.losses.mean_squared_error(q_value, pred)
        train_opt = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)
        saver = tf.train.Saver()
        tf.summary.scalar("Loss", loss)
        write_op = tf.summary.merge_all()
    return (
        fc3,
        observation,
        pred,
        max_q_value,
        q_value,
        loss,
        train_opt,
        saver,
        write_op,
        q_value_index,
    )


(
    fc3,
    observation,
    pred,
    max_q_value,
    q_value,
    loss,
    train_opt,
    saver,
    write_op,
    q_value_index,
) = build_neural_network("q_network")
(
    target_fc3,
    target_observation,
    target_pred,
    target_max_q_value,
    target_q_value,
    target_loss,
    target_train_opt,
    target_saver,
    target_write_op,
    target_q_value_index,
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
            evaluated_fc3 = sess.run(
                fc3, feed_dict={observation: [raw_state]}
            )
            action = np.argmax(evaluated_fc3[0])
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

