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
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--pass_limit", default=100, type=int, help="Stop evaluation after reaching.")
parser.add_argument("--max_buffer_size", default=10000, type=int, help="Buffer size limit.")
parser.add_argument("--delay_freq", default=2, type=int, help="Delay parameter.")
parser.add_argument("--exploration_noise_sigma", default=0.1, type=float, help="UB noise sigma.")
parser.add_argument("--policy_noise_sigma", default=0.2, type=float, help='Policy smoothing sigma.')
parser.add_argument("--policy_noise_clip", default=0.5, type=float, help='Policy smoothing clip.')
parser.add_argument("--train", default=False, action="store_true", help="Training.")
parser.add_argument("--model", default="vanilla_tanh_model", type=str, help='Saved model identifier.')


class Network:
    def __init__(self, env, args, rng, actor=None, target_actor=None,
                 critic=None, critic2=None, target_critic=None, target_critic2=None):
        self.target_tau = args.target_tau
        self.rng = rng
        self.c = args.policy_noise_clip
        self.sigma = args.policy_noise_sigma

        low, high = env.action_space.low, env.action_space.high

        inputs = tf.keras.Input(shape=env.observation_space.shape)

        if actor is None:
            actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
            actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(actor)
            actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(actor)

            # TEST ===========================================
            actor = tf.keras.layers.Dense(400, activation='relu')(inputs)
            actor = tf.keras.layers.Dense(300, activation='relu')(actor)
            # END TEST =======================================

            actor = tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')(actor)
            actor = actor * (high - low) / 2 + (high + low) / 2
            self.actor = tf.keras.Model(inputs=inputs, outputs=actor)
            self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        else:
            self.actor = actor

        if (critic is None and critic2 is not None) or (critic is not None and critic2 is None):
            raise NotImplementedError()

        if critic is None:
            critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
            critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
            action_inputs = tf.keras.Input(shape=env.action_space.shape)
            critic = tf.keras.layers.concatenate([critic, action_inputs])
            critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
            critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
            critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
            
            # TEST =============================================
            critic = tf.keras.layers.concatenate([inputs, action_inputs])
            critic = tf.keras.layers.Dense(400, activation='relu')(critic)
            critic = tf.keras.layers.Dense(300, activation='relu')(critic)
            # END TEST ==========================================

            critic = tf.keras.layers.Dense(1)(critic)
            self.critic = tf.keras.Model(inputs=[inputs, action_inputs], outputs=critic)
            self.critic2 = tf.keras.models.clone_model(self.critic)
            self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                loss=tf.keras.losses.MeanSquaredError())
            self.critic2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                                 loss=tf.keras.losses.MeanSquaredError())
        else:
            self.critic = critic
            self.ctitic2 = critic2

        self.target_actor = tf.keras.models.clone_model(self.actor) if target_actor is None else target_actor
        self.target_critic = tf.keras.models.clone_model(self.critic) if target_critic is None else target_critic
        self.target_critic2 = tf.keras.models.clone_model(self.critic2) if target_critic2 is None else target_critic2


    # the functions are split due to unresolved bug in tensorflow #34983
    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train_critics(self, states, actions, returns):
        self.critic.optimizer.minimize(
            lambda: self.critic.loss(returns, self.critic([states, actions], training=True)),
            var_list=self.critic.trainable_variables
        )
        self.critic2.optimizer.minimize(
            lambda: self.critic2.loss(returns, self.critic2([states, actions], training=True)),
            var_list=self.critic2.trainable_variables
        )


    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train_actor_and_targets(self, states, actions, returns):
        self.actor.optimizer.minimize(
            lambda: -self.critic([states, self.actor(states, training=True)], training=True),
            var_list=self.actor.trainable_variables
        )

        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)
        for var, target_var in zip(self.critic2.trainable_variables, self.target_critic2.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)


    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self.actor(states)


    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        actor = self.target_actor(states)
        noise = np.clip(self.rng.normal(scale=self.sigma, size=actor.shape), -self.c, self.c)
        critic = self.target_critic([states, actor + noise])
        critic2 = self.target_critic2([states, actor + noise])
        return tf.minimum(critic, critic2)


    def save(self, file_name):
        self.actor.save(file_name + ".actor.h5")
        self.target_actor.save(file_name + ".target_actor.h5")
        self.critic.save(file_name + ".critic.h5")
        self.critic2.save(file_name + ".critic2.h5")
        self.target_critic.save(file_name + ".target_critc.h5")
        self.target_critic2.save(file_name + ".target_critc2.h5")


    @staticmethod
    def load_from_files(file_name, env, args, rng):
        # TODO check if args env and loaded model are compatible
        actor = tf.keras.models.load_model(file_name + ".actor.h5")
        target_actor = tf.keras.models.load_model(file_name + ".target_actor.h5")
        critic = tf.keras.models.load_model(file_name + ".critic.h5")
        critic2 = tf.keras.models.load_model(file_name + ".critic2.h5")
        target_critic = tf.keras.models.load_model(file_name + ".target_critc.h5")
        target_critic2 = tf.keras.models.load_model(file_name + ".target_critc2.h5")
        return Network(env, args, rng, actor, target_actor,
                       critic, critic2, target_critic, target_critic2)


def main(env, args):
    # Fix random seeds and number of threads
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    if args.model and not args.train:
        network = Network.load_from_files(args.model, env, args, rng)
        while True: evaluate_episode(start_evaluation=True)
        return

    network = Network(env, args, rng)

    replay_buffer = collections.deque(maxlen=args.max_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    training = True
    while training:
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            timestep = 0
            while not done:
                action = network.predict_actions([state])[0] + rng.normal(scale=args.exploration_noise_sigma)
                action = action.clip(env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                # TEST =====================================
                if abs(next_state[2]) < 0.001:
                    reward = -100
                    done = True
                # END TEST ================================

                if len(replay_buffer) >= args.batch_size:
                    batch = rng.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    returns = rewards + args.gamma * (~dones) * network.predict_values(next_states).flatten()
                    network.train_critics(states, actions, returns.reshape(-1, 1))

                    if timestep % args.delay_freq == 0:
                        network.train_actor_and_targets(states, actions, returns.reshape(-1, 1))

                timestep += 1

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

        if np.mean(env._episode_returns[-args.evaluate_for:]) > args.pass_limit:
            training = False

    print(args)
    if args.model:
        network.save(args.model)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    print(args)
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("BipedalWalker-v3"), args.seed)

    main(env, args)
