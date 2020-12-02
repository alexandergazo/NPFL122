#!/usr/bin/env python3
import argparse
import collections
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
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.002, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.01, type=float, help="Target network update weight.")
parser.add_argument("--pass_limit", default=-200, type=int, help="Stop evaluation after reaching.")


def get_scaled_tanh(low=-1, high=1):
    from tensorflow.keras.activations import sigmoid, tanh
    if low==-1 and high==1:
        return tanh
    else:
        return lambda x: sigmoid(2 * x) * (high - low) + low

class Network:
    def __init__(self, env, args):
        self.target_tau = args.target_tau
        low, high = env.action_space.low, env.action_space.high

        inputs = tf.keras.Input(shape=env.observation_space.shape)
        actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(actor)
        actor = tf.keras.layers.Dense(env.action_space.shape[0])(actor)
        actor = tf.keras.layers.Lambda(get_scaled_tanh(low, high))(actor)
        self.actor = tf.keras.Model(inputs=inputs, outputs=actor)
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10))
        self.target_actor = tf.keras.models.clone_model(self.actor)

        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        action_inputs = tf.keras.Input(shape=env.action_space.shape)
        critic = tf.keras.layers.concatenate([critic, action_inputs])
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
        critic = tf.keras.layers.Dense(1)(critic)
        self.critic = tf.keras.Model(inputs=[inputs, action_inputs], outputs=critic)
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                            loss=tf.keras.losses.MeanSquaredError())
        self.target_critic = tf.keras.models.clone_model(self.critic)
        # tf.keras.utils.plot_model(self.critic, show_shapes=True)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        self.actor.optimizer.minimize(
            lambda: -self.critic([states, self.actor(states, training=True)], training=True),
            var_list=self.actor.trainable_variables
        )

        self.critic.optimizer.minimize(
            lambda: self.critic.loss(returns, self.critic([states, actions], training=True)),
            var_list=self.critic.trainable_variables
        )

        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_target_actions(self, states):
        return self.target_actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        return self.target_critic([states, self.target_actor(states)])


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            action = network.predict_target_actions([state])[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    while training:
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                action = network.predict_actions([state])[0] + noise.sample()
                action = action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    returns = rewards + args.gamma * (~dones) * network.predict_values(next_states).flatten()
                    network.train(states, actions, returns.reshape(-1, 1))

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
