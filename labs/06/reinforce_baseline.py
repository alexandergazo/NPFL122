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
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--grid_search", default=False, action="store_true", help="Use grid search.")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="baseline_model.h5", type=str, help="Path to trained model.")
parser.add_argument("--baseline_path", default="baseline_baseline.h5", type=str, help="Path to trained baseline.")

class Network:
    def __init__(self, env, args, model=None, baseline=None):
        if model is None:
            inputs = tf.keras.Input(shape=env.observation_space.shape)
            model = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
            model = tf.keras.layers.Dense(env.action_space.n, activation='softmax')(model)
            model = tf.keras.Model(inputs=inputs, outputs=model)
            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        if baseline is None:
            inputs = tf.keras.Input(shape=env.observation_space.shape)
            baseline = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
            baseline = tf.keras.layers.Dense(1)(baseline)
            baseline = tf.keras.Model(inputs=inputs, outputs=baseline)
            baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._model = model
        self._baseline = baseline


    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        baseline = tf.squeeze(self._baseline(states, training=True))
        self._model.optimizer.minimize(
            lambda: self._model.loss(actions, self._model(states, training=True), sample_weight= returns - baseline),
            var_list=self._model.trainable_variables)
        self._baseline.optimizer.minimize(
            lambda: self._baseline.loss(returns, tf.squeeze(self._baseline(states, training=True))),
            var_list=self._baseline.trainable_variables)


    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)


def main(env, args):
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

        network.train(np.array(batch_states), np.array(batch_actions), np.array(batch_returns))

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            action = network.predict([state])[0].argmax()
            state, reward, done, _ = env.step(action)
        if env.episode >= args.episodes + 100:
            return


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.grid_search:
        # Env needs not to sys.exit(0) when finished
        from sklearn.model_selection import ParameterGrid
        grid = {'hidden_layer_size': [2,4,50], 'batch_size': [1], 'gamma': [1, 0.99],
                'learning_rate': np.geomspace(0.005, 0.05, 5)}
        grid = list(ParameterGrid(grid))
        results = np.empty(len(grid))
        seeds = np.random.randint(0, 10000, 10, dtype=int).tolist()
        from tqdm import tqdm
        log = open('tqdm.log', 'w')
        log_all = open('all.log', 'w')
        log_passed = open('passed.log', 'w')
        for i, arguments in enumerate(tqdm(grid, file=log)):
            for param in arguments:
                setattr(args, param, arguments[param])
            result = 0
            for seed in tqdm(seeds, file=log):
                args.seed = seed
                env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)
                main(env, args)
                result += np.mean(env._episode_returns[-100:])
            results[i] = result / len(seeds)
            log_all.write(f'{arguments} {results[i]}\n')
            log_all.flush()
            if results[i] >= 490:
                log_passed.write(f'{arguments} {results[i]}\n')
                log_passed.flush()
        for result, arguments in zip(results, grid):
            if result >= 490:
                print(arguments, result)
        print(grid[results.argmax()], results.max())
        exit(0)
    else:
        env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)
        main(env, args)
