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
gamma = 0.98
seed = 1
num_episodes = 1000

# Utility functions
def get_dependent_varialbes(tensor):
    import collections

    op_to_var = {var.op: var for var in tf.trainable_variables()}
    dependent_vars = []
    queue = collections.deque()
    queue.append(tensor.op)
    visited = set([tensor.op])
    while queue:
        op = queue.popleft()
        try:
            dependent_vars.append(op_to_var[op])
        except KeyError:
            # `op` is not a variable, so search its inputs (if any).
            for op_input in op.inputs:
                if op_input.op not in visited:
                    queue.append(op_input.op)
                    visited.add(op_input.op)
    return dependent_vars


# Set up the env
env = gym.make("CartPole-v0")
np.random.seed(seed)
tf.random.set_random_seed(seed)
env.seed(seed)

obs_ph = tf.placeholder(shape=(None,) + env.observation_space.shape, dtype=tf.float64)
fc1 = tf.layers.dense(inputs=obs_ph, units=64)
fc2 = tf.layers.dense(inputs=fc1, units=64)
fc3 = tf.layers.dense(inputs=fc2, units=env.action_space.n)
action_probs = tf.nn.softmax(fc3)
action_probs_chosen_indices_ph = tf.placeholder(shape=(None), dtype=tf.int32)
action_probs_chosen = tf.gather_nd(action_probs, action_probs_chosen_indices_ph)
future_rewards_ph = tf.placeholder(shape=(None), dtype=tf.float64)
temp = tf.reduce_mean(tf.log(action_probs_chosen) * future_rewards_ph)

# Update paramaters
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(-temp)
cost_related_vars = get_dependent_varialbes(temp)
grads = tf.gradients(temp, cost_related_vars)
vars_and_grads = list(zip(cost_related_vars, grads))
ops = []
for item in vars_and_grads:
    ops.append(tf.assign(item[0], item[0] + learning_rate * item[1]))
train_op = tf.group(*ops)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

episode_rewards = []
for i_episode in range(num_episodes):
    state = env.reset()
    episode = []
    states = []
    actions_taken = []
    rewards = []
    # One step in the environment
    for t in range(200):

        # Take a step
        evaluated_action_probs = sess.run(action_probs, feed_dict={obs_ph: [state]})
        action = np.random.choice(
            np.arange(len(evaluated_action_probs[0])), p=evaluated_action_probs[0]
        )
        next_state, reward, done, _ = env.step(action)

        # Keep track of the transition
        states += [state]
        actions_taken += [action]
        rewards += [reward]

        if done:
            break

        state = next_state

    if i_episode % 10 == 0:
        print(f"i_episode = {i_episode}, rewards = {sum(rewards)}")
        episode_rewards += [sum(rewards)]

    # Go through the episode and make policy updates
    for t, item in enumerate(rewards):
        # The return after this timestep
        future_rewards = sum(rewards[t + 1 :])
        sess.run(
            train_op,
            feed_dict={
                obs_ph: [states[t]],
                action_probs_chosen_indices_ph: list(enumerate([actions_taken[t]])),
                future_rewards_ph: future_rewards * gamma ** (t),
            },
        )


plt.plot(episode_rewards)
