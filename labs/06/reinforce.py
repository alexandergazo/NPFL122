#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="del.h5", type=str, help="Path to trained model.")

class Network:
    def __init__(self, env, args, model=None):
        if model is None:
            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=env.observation_space.shape, batch_size=args.batch_size))
            model.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='relu'))
            model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._model = model

    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        self._model.optimizer.minimize(
            lambda: tf.reduce_sum(returns * self._model.loss(actions, self._model(states, training=True))) / states.shape[0],
            var_list=self._model.trainable_variables)

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

def main(env, args):
    if os.path.isfile(args.model_path):
        network = Network(env, args, tf.keras.models.load_model(args.model_path))
    else:
        # Fix random seeds and number of threads
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

        # Construct the network
        network = Network(env, args)

        # Training
        for _ in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                while not done:
                    state = state.tolist()
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    action = np.random.choice(env.action_space.n, p=network.predict([state])[0])

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state

                gammas = args.gamma ** np.arange(len(rewards))
                returns = [np.dot(rewards[i:], gammas[:len(gammas) - i]) for i in range(len(rewards))]

                batch_states.append(states)
                batch_actions.append(actions)
                batch_returns.append(returns)

            def equalize_lengths(l, fill=0):
                max_len = max(map(len, l))
                return tf.convert_to_tensor([x + [fill] * (max_len - len(x)) for x in l], dtype=np.float32)

            batch_states = equalize_lengths(batch_states, fill=np.zeros(env.observation_space.shape))
            batch_actions = equalize_lengths(batch_actions)
            batch_returns = equalize_lengths(batch_returns)
            network.train(batch_states, batch_actions, batch_returns)

        network._model.save(args.model_path)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = network.predict([state])[0].argmax()
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
