import gym
import tensorflow as tf
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

tf.reset_default_graph()

# Utility functions
def backup_training_variables(scope: str, sess: tf.Session) -> List[List]:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return sess.run(variables)


def restore_training_variables(
    scope: str, vars_values: List[List], sess: tf.Session
) -> None:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    ops = []
    for i, var in enumerate(variables):
        ops.append(tf.assign(var, vars_values[i]))
    sess.run(tf.group(*ops))


def render_env(state, env):
    env.s = state
    env.render()

# https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/tf_util.html#make_session
def make_session(num_cpu=None, make_default=False, graph=None):
    """
    Returns a session that will use <num_cpu> CPU's only

    :param num_cpu: (int) number of CPUs to use for TensorFlow
    :param make_default: (bool) if this should return an InteractiveSession or a normal Session
    :param graph: (TensorFlow Graph) the graph of the session
    :return: (TensorFlow session)
    """
    if num_cpu is None:
        num_cpu = int(os.getenv("RCALL_NUM_CPU", multiprocessing.cpu_count()))
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
    )
    # Prevent tensorflow from taking all the gpu memory
    tf_config.gpu_options.allow_growth = True
    if make_default:
        return tf.InteractiveSession(config=tf_config, graph=graph)
    else:
        return tf.Session(config=tf_config, graph=graph)


# functions for graduatlly decrease learning rate using linear functions
def get_explore_rate(t):
    return max(
        (EPSILON_MIN - EPSILON_MAX) * t / MAX_EXPLORATION_RATE_DECAY_TIMESTEP
        + EPSILON_MAX,
        EPSILON_MIN,
    )


# Replay Memory
class ExperienceReplay:
    def __init__(self, buffer_size=50000):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = np.empty((0, 5), int)
        self.buffer_size = buffer_size

    def add(self, experience):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer = np.append(self.buffer, np.array([experience]), axis=0)
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size :]

    def sample(self, size: int) -> List[List]:
        """ Returns a sample of experiences from the buffer """
        if len(self.buffer) < size:
            size = len(self.buffer)
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        return np.array(sample_output)


# Hypterparameters
# https://en.wikipedia.org/wiki/Q-learning
ALPHA = 1e-3  # learning rate
EPSILON_MAX = 1  # exploration rate
EPSILON_MIN = 0.1
GAMMA = 0.7  # discount factor
MAX_EXPLORATION_RATE_DECAY_TIMESTEP = 5000
TARGET_NETWORK_UPDATE_STEP_FREQUENCY = 500
EXPERIENCER_REPLAY_BATCH_SIZE = 32

# Training parameters
SEED = 1000
NUM_EPISODES = 1000
MAX_NUM_STEPS = 200
TOTAL_MAX_TIMESTEPS = 50000
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

# Start the training process
with make_session(8) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs")
    restore_training_variables(
        "target_network", backup_training_variables("q_network", sess), sess
    )
    er = ExperienceReplay()
    episode_rewards = []
    finished_episodes_count = 0
    target_network_update_counter = 0
    total_timesteps = 0
    for i_episode in range(NUM_EPISODES):
        raw_state = env.reset()
        done = False
        episode_reward = 0
        for t in range(MAX_NUM_STEPS):
            epsilon = get_explore_rate(total_timesteps)
            total_timesteps += 1
            target_network_update_counter += 1
            # env.render()
            if random.random() < epsilon:
                action = random.randint(0, env.action_space.n - 1)
            else:
                evaluated_action_probability = sess.run(
                    action_distribution, feed_dict={observation: [[raw_state]]}
                )
                action = np.argmax(evaluated_action_probability)
            old_raw_state = raw_state
            raw_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store transition in the experience replay
            er.add([old_raw_state, action, reward, raw_state, done])

            # Sample random minibatch of trasitions from the experience replay
            if len(er.buffer) < EXPERIENCER_REPLAY_BATCH_SIZE:
                continue
            batch = er.sample(EXPERIENCER_REPLAY_BATCH_SIZE)

            if done:
                finished_episodes_count += 1

            # Predict
            # use the raw_state from the replay buffer, which is the column at index-3
            # https://stackoverflow.com/questions/4455076/how-to-access-the-ith-column-of-a-numpy-multidimensional-array
            # This is wrong.
            evaluated_target_q_value = sess.run(
                target_q_value, feed_dict={target_observation: er.buffer[:, [3]]}
            )
            y = reward + GAMMA * evaluated_target_q_value
            

            # Train
            _, summary = sess.run(
                [train_opt, write_op],
                feed_dict={observation: er.buffer[:, [0]], pred: y},
            )  # the 0-index column is the old_raw_state

            writer.add_summary(summary, total_timesteps)

            # Update the target network
            if target_network_update_counter > TARGET_NETWORK_UPDATE_STEP_FREQUENCY:
                restore_training_variables(
                    "target_network", backup_training_variables("q_network", sess), sess
                )
                target_network_update_counter = 0

            if done:
                break

            if total_timesteps % 1000 == 0:
                save_path = saver.save(sess, "./tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)

        print(
            "Episode: ",
            i_episode,
            "finished with rewards of ",
            episode_reward,
            "with successful drop-offs of",
            finished_episodes_count,
        )
        episode_rewards += [episode_reward]
        if total_timesteps > TOTAL_MAX_TIMESTEPS:
            break

    plt.plot(episode_rewards)
