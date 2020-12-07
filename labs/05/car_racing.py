#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import itertools
import car_racing_environment
import wrappers

from skimage.measure import block_reduce

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=30000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001125, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=1000, type=int, help="Target update frequency.")
parser.add_argument("--buffer_size", default=100000, type=int, help="Buffer maximal size.")
parser.add_argument("--model_path", default='car_model_fast_relu_simple_actions.h5', type=str, help="Path to trained model.")
parser.add_argument("--episodes", default=500000, type=int, help="Number of training episodes.")
parser.add_argument("--omega", default=0.5, type=float, help="Priority distribution exponent.")
parser.add_argument("--beta", default=0.4, type=float, help="Priority distribution exponent.")
parser.add_argument("--beta_final", default=1, type=float, help="Priority distribution exponent.")
parser.add_argument("--beta_final_at", default=30000, type=float, help="Priority distribution exponent.")
parser.add_argument("--replay_start", default=5000, type=int, help='Start training at buffer size.')
parser.add_argument("--downsampling", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--grayscale", default=True, action="store_true", help="Running in ReCodEx")

class Network:
    def __init__(self, env, args, model=None):
        if model is None:
            if args.downsampling:
                inputs = tf.keras.Input(shape=(24, 24, 3))
            elif args.grayscale:
                input_shape = (*env.observation_space.shape[:2], 1)
            else:
                input_shape = env.observation_space.shape
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(32, 8, 4, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.Conv2D(64, 4, 2, padding='same', activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Conv2D(64, 3, 1, padding='same', activation='relu')(x)
            x = tf.keras.layers.Reshape((-1,))(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dense(env.action_space.n)(x)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._model = model

    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def train(self, states, q_values):
        self._model.optimizer.minimize(
            lambda: q_values.shape[1] * self._model.loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

    @tf.function
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)


def main(env, args):
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # actions_dict = list(itertools.product(np.linspace(-1, 1, 5), np.linspace(0, 1, 5), [0, 0.1, 0.4, 0.7]))
    # actions_dict = list(itertools.product([-1, 0, 1], [0, 1], [0, 0.5]))
    actions_dict = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.8], [0, 0, 0]]
    env.action_space.n = len(actions_dict)

    if args.grayscale: env.observation_space.shape = env.observation_space.shape[:2]

    if os.path.isfile(args.model_path):
        target_network = Network(env, args, tf.keras.models.load_model(args.model_path))
    else:
        try:
            network = Network(env, args)
            target_network = Network(env, args)
            target_network.copy_weights_from(network)

            # TODO separate class for replay_buffer
            replay_buffer, weights = [], []
            replay_buffer_idx = 0
            Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

            beta, epsilon, step = args.beta, args.epsilon, 0
            from tqdm import tqdm
            tqdm_log_file = open("tqdm.log", "w")
            pbar = tqdm(total=args.episodes, file=tqdm_log_file)
            while step < args.episodes:
                state, done = env.reset(), False
                if args.downsampling: state = block_reduce(state, (4, 4, 1), np.max)
                if args.grayscale:
                    state = state @ [0.299, 0.587, 0.114]
                    state = state[..., np.newaxis]
                state = 2 * state - 1
                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    action = rng.integers(env.action_space.n) if rng.uniform() < epsilon else \
                        np.argmax(network.predict([state])[0])
                    next_state, reward, done, _ = env.step(actions_dict[action])

                    if args.downsampling: next_state = block_reduce(next_state, (4, 4, 1), np.max)
                    if args.grayscale:
                        next_state = next_state @ [0.299, 0.587, 0.114]
                        next_state = next_state[..., np.newaxis]
                    next_state = next_state * 2 - 1

                    t = Transition(state, action, reward, done, next_state)

                    weight = abs(reward + (~done) * args.gamma * \
                        target_network.predict([next_state])[0, np.argmax(network.predict([next_state])[0])])
                    weight = (1 / (weight ** args.omega)) ** beta
                    if len(replay_buffer) == args.buffer_size - 1:
                        weights[replay_buffer_idx] = weight
                        replay_buffer[replay_buffer_idx] = t
                    else:
                        weights.append(weight)
                        replay_buffer.append(t)

                    if args.beta_final_at:
                        beta = np.interp(step + 1, [0, args.beta_final_at], [args.beta, args.beta_final])
                    if len(replay_buffer) >= args.replay_start:
                        probs = np.array(weights) / np.sum(weights)
                        batch = rng.choice(len(replay_buffer), args.batch_size, p=probs)
                        batch[0] = replay_buffer_idx if len(replay_buffer) == args.buffer_size -1 else len(replay_buffer) - 1

                        states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))

                        targets = rewards + (~dones) * args.gamma * \
                            target_network.predict(next_states)[
                                np.arange(len(next_states)),
                                np.argmax(network.predict(next_states), axis=1)
                            ]
                        q_values = network.predict(states)
                        q_values[np.arange(len(actions)), actions] = targets
                        network.train(states, q_values)

                    pbar.update(1)
                    step += 1
                    if step % args.target_update_freq == 0:
                        target_network.copy_weights_from(network)
                    state = next_state

                if args.epsilon_final_at:
                    epsilon = np.interp(step + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        except KeyboardInterrupt:
            target_network._model.save(args.model_path)

        target_network._model.save(args.model_path)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()
            if args.grayscale:
                state = state @ [0.299, 0.587, 0.114]
                state = state[..., np.newaxis]
            state = 2 * state - 1
            action = np.argmax(target_network.predict([state])[0])
            state, reward, done, _ = env.step(actions_dict[action])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip)), args.seed, evaluate_for=15, report_each=1)

    main(env, args)
