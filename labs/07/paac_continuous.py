#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=16, type=int, help="Tiles to use.")
parser.add_argument("--workers", default=12, type=int, help="Number of parallel workers.")
parser.add_argument("--pass_limit", default=90, type=int, help="Stop evaluation after reaching.")

def pseudo_one_hot(env, states):
    if isinstance(states, list) or states.ndim == 1:
        states = np.reshape(states, (1, -1))
    states_i = np.zeros((states.shape[0], env.observation_space.nvec[-1]))
    for i, row in enumerate(states):
        states_i[i, row] = 1
    return states_i

class Network:
    def __init__(self, env, args):
        inputs = tf.keras.Input(shape=(env.observation_space.nvec[-1],), batch_size=args.workers)

        actor_base = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        # TODO for different env, change mean activation
        actor_head_mean = tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')(actor_base)
        actor_head_std = tf.keras.layers.Dense(env.action_space.shape[0], activation='softplus')(actor_base)
        self._actor = tf.keras.Model(inputs=inputs, outputs=[actor_head_mean, actor_head_std], name='actor')
        self._actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        # tf.keras.utils.plot_model(self._actor, "actor.png", show_shapes=True)

        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        critic = tf.keras.layers.Dense(1)(critic)
        self._critic = tf.keras.Model(inputs=inputs, outputs=critic)
        self._critic.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        self.args = args

    @tf.function
    def _loss(self, states, actions, returns):
        action_distributions = tfp.distributions.Normal(*self._predict_actions(states))

        entropy_loss = -self.args.entropy_regularization * action_distributions.entropy()

        state_values = self._predict_values(states)
        NLL = tf.reduce_sum(-action_distributions.log_prob(actions), axis=1, keepdims=True)
        NLL = tf.reduce_mean(NLL * (returns - state_values))
        return entropy_loss + NLL


    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        self._actor.optimizer.minimize(
            lambda: self._loss(states, actions, returns),
            var_list = self._actor.trainable_variables
        )
        self._critic.optimizer.minimize(
            lambda: self._critic.loss(returns, self._critic(states)),
            var_list=self._critic.trainable_variables
        )

    @tf.function
    def _predict_actions(self, states):
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self._predict_actions(states)

    @tf.function
    def _predict_values(self, states):
        return self._critic(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        return self._predict_values(states)

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

            state = pseudo_one_hot(env, state)
            action = network.predict_actions([state])[0][0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    env_constructors = [lambda: wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles)] * args.workers
    if args.recodex:
        vector_env = gym.vector.AsyncVectorEnv(env_constructors)
    else:
        vector_env = gym.vector.SyncVectorEnv(env_constructors)
    vector_env.seed(args.seed)
    states = pseudo_one_hot(env, vector_env.reset())

    training = True
    while training:
        for _ in range(args.evaluate_each):
            # print(network.predict_actions(states))
            actions = rng.normal(*network.predict_actions(states))
            actions = actions.clip(env.action_space.low, env.action_space.high)

            next_states, rewards, dones, _ = vector_env.step(actions)
            next_states = pseudo_one_hot(env, next_states)

            returns = rewards + args.gamma * (~dones) * network.predict_values(next_states).flatten()

            network.train(states, actions, returns.reshape(-1, 1))

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
    env = wrappers.EvaluationWrapper(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles), args.seed)

    main(env, args)
