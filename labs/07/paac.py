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
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
parser.add_argument("--workers", default=128, type=int, help="Number of parallel workers.")
parser.add_argument("--pass_limit", default=450, type=int, help="Stop evaluation after reaching.")

class Network:
    def __init__(self, env, args, actor=None, critic=None):
        if actor is None:
            actor = tf.keras.Sequential()
            actor.add(tf.keras.Input(shape=env.observation_space.shape))
            actor.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='relu'))
            actor.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
            actor.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        if critic is None:
            critic = tf.keras.Sequential()
            critic.add(tf.keras.Input(shape=env.observation_space.shape))
            critic.add(tf.keras.layers.Dense(args.hidden_layer_size, activation='relu'))
            critic.add(tf.keras.layers.Dense(1))
            critic.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._actor = actor
        self._critic = critic

    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        delta = returns - tf.squeeze(self._critic(states, training=True))
        self._actor.optimizer.minimize(
            lambda: self._actor.loss(actions, self._actor(states, training=True), sample_weight=delta),
            var_list=self._actor.trainable_variables)
        self._critic.optimizer.minimize(
            lambda: self._critic.loss(returns, tf.squeeze(self._critic(states, training=True))),
            var_list=self._critic.trainable_variables)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        return self._critic(states)

def main(env, args):
    # Fix random seeds and number of threads
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            action = np.argmax(network.predict_actions([state])[0])
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv([lambda: gym.make(env.spec.id)] * args.workers)
    states = vector_env.reset()

    training = True
    while training:
        for _ in range(args.evaluate_each):
            actions = [rng.choice(env.action_space.n, p=p) for p in network.predict_actions(states)]

            next_states, rewards, dones, _ = vector_env.step(actions)

            returns = rewards + args.gamma * (~dones) * network.predict_values(next_states).flatten()

            network.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

        if np.mean(env._episode_returns[-args.evaluate_for:]) > args.pass_limit:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
