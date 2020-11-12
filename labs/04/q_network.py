#!/usr/bin/env python3
import argparse
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import lzma
import pickle
import gym
import numpy as np
import tensorflow as tf
import random

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=70, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=48, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=20, type=int, help="Target update frequency.")
parser.add_argument("--buffer_update_size", default=10, type=int, help="Minimal buffer size for sampling.")
parser.add_argument("--model_path", default="q_network_model.h5", type=str, help="Path to trained model.")
parser.add_argument("--episodes", default=500, type=int, help="Number of training episodes.")


class Network:
    def __init__(self, env, args, model=None):
        if model is None:
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=env.observation_space.shape, batch_size=args.batch_size))
            model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='relu'))
            model.add(tf.keras.layers.Dense(env.action_space.n))
            model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')
        self._model = model

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    # Also note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def train(self, states, q_values):
        self._model.optimizer.minimize(
            lambda: self._model.loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

    def get_q(self, state):
        return self.predict(np.asarray([state], np.float32))[0]

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


def main(env, args):
    if os.path.isfile(args.model_path):
        target_network = Network(env, args, tf.keras.models.load_model(args.model_path))
    else:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

        network = Network(env, args)
        target_network = Network(env, args)
        target_network.copy_weights_from(network)

        # Replay memory; maxlen parameter can be passed to deque for a size limit,
        # which we however do not need in this simple task.
        replay_buffer = collections.deque()
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

        epsilon, step = args.epsilon, 0
        for _ in range(args.episodes):
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                action = env.action_space.sample() if np.random.uniform() < epsilon else np.argmax(network.get_q(state))
                next_state, reward, done, _ = env.step(action)

                replay_buffer.append(Transition(state, action, reward, done, next_state))

                if len(replay_buffer) > args.buffer_update_size:
                    transitions = random.sample(replay_buffer, args.batch_size)
                    states = np.asarray([t.state for t in transitions])
                    q_values = network.predict(states).numpy()
                    targets = [t.reward + (1 - t.done) * args.gamma * np.max(target_network.get_q(t.next_state)) \
                               for t in transitions]
                    actions = [t.action for t in transitions]
                    q_values[np.arange(len(actions)), actions] = targets
                    network.train(states, q_values)

                step = (step + 1) % args.target_update_freq
                if step == 0:
                    target_network.copy_weights_from(network)
                state = next_state

            if args.epsilon_final_at:
                epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        network._model.save(args.model_path)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = np.argmax(target_network.get_q(state))
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
