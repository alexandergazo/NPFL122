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
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="OpenAI Gym environment.")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=400, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--pass_limit", default=290, type=int, help="Stop evaluation after reaching.")
parser.add_argument("--max_buffer_size", default=500000, type=int, help="Buffer size limit.")
parser.add_argument("--delay_freq", default=2, type=int, help="Delay parameter.")
parser.add_argument("--exploration_noise_sigma", default=0.1, type=float, help="Exploration noise sigma.")
parser.add_argument("--policy_noise_sigma", default=0.2, type=float, help='Policy smoothing sigma.')
parser.add_argument("--policy_noise_clip", default=0.5, type=float, help='Policy smoothing clip.')
parser.add_argument("--test", default=False, action="store_true", help="Testing.")
parser.add_argument("--load_model", default=None, type=str, help='Load model identifier.')
parser.add_argument("--save_model", default=None, type=str, help='Save model identifier.')
parser.add_argument("--no_penalty", default=False, action="store_true", help="Don't use penalty on falling (walker env).")
parser.add_argument("--min_buffer_size", default=0, type=int, help="Minimal buffer size needed for training.")
parser.add_argument("--save_limit", default=None, type=float, help="Save every time evaluation performs better than this limit.")
parser.add_argument("--print_episode_length", default=False, action="store_true", help="After each training episode print its length.")


class Network:
    def __init__(self, env, args, rng, actor=None, target_actor=None,
                 critic=None, critic2=None, target_critic=None, target_critic2=None):

        def get_actor(hidden_layer_units=[400, 300]):
            inputs = tf.keras.Input(shape=env.observation_space.shape)

            actor = tf.keras.layers.Dense(hidden_layer_units[0], activation='relu')(inputs)
            for units in hidden_layer_units[1:]:
                actor = tf.keras.layers.Dense(units, activation='relu')(actor)

            actor = tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')(actor)
            actor = actor * (high - low) / 2 + (high + low) / 2
            
            actor = tf.keras.Model(inputs=inputs, outputs=actor)
            actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
            return actor

        def get_critic(hidden_layer_units=[400, 300]):
            inputs = tf.keras.Input(shape=env.observation_space.shape)
            action_inputs = tf.keras.Input(shape=env.action_space.shape)

            critic = tf.keras.layers.concatenate([inputs, action_inputs])

            for units in hidden_layer_units:
                critic = tf.keras.layers.Dense(units, activation='relu')(critic)

            critic = tf.keras.layers.Dense(1)(critic)

            critic = tf.keras.Model(inputs=[inputs, action_inputs], outputs=critic)
            critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())
            return critic

        self.target_tau = args.target_tau
        self.rng = rng
        self.c = args.policy_noise_clip
        self.sigma = args.policy_noise_sigma

        low, high = env.action_space.low, env.action_space.high

        self.actor = get_actor() if actor is None else actor
        self.critic = get_critic() if critic is None else critic
        self.critic2 = get_critic() if critic2 is None else critic2

        self.target_actor = tf.keras.models.clone_model(self.actor) if target_actor is None else target_actor
        self.target_critic = tf.keras.models.clone_model(self.critic) if target_critic is None else target_critic
        self.target_critic2 = tf.keras.models.clone_model(self.critic2) if target_critic2 is None else target_critic2


    # the training functions are split due to unresolved bug in tensorflow #34983
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
        self.target_critic.save(file_name + ".target_critic.h5")
        self.target_critic2.save(file_name + ".target_critic2.h5")


    @staticmethod
    def load_from_files(file_name, env, args, rng):
        # TODO check if args env and loaded model are compatible
        actor = tf.keras.models.load_model(file_name + ".actor.h5")
        target_actor = tf.keras.models.load_model(file_name + ".target_actor.h5")
        critic = tf.keras.models.load_model(file_name + ".critic.h5")
        critic2 = tf.keras.models.load_model(file_name + ".critic2.h5")
        target_critic = tf.keras.models.load_model(file_name + ".target_critic.h5")
        target_critic2 = tf.keras.models.load_model(file_name + ".target_critic2.h5")
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

    if args.load_model:
        network = Network.load_from_files(args.load_model, env, args, rng)
    else:
        network = Network(env, args, rng)
        
    if args.test:
        while True: evaluate_episode(start_evaluation=True)
        return

    # TRAINING

    if not args.save_model:
        args.save_model = input("No save location specified. Do you want to save model anyway? If yes, enter the model name, otherwise press enter.\n")

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

                reward = 0 if reward == -100 and args.no_penalty else reward

                replay_buffer.append(Transition(state, action, reward, done, next_state))
                
                state = next_state
                timestep += 1

            if args.print_episode_length: print(timestep)

            if len(replay_buffer) <= max(args.min_buffer_size, args.batch_size): continue

            for i in range(timestep):
                batch = rng.choice(len(replay_buffer), size=args.batch_size, replace=False)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                returns = rewards + args.gamma * (~dones) * network.predict_values(next_states).flatten()

                network.train_critics(states, actions, returns.reshape(-1, 1))
                if i % args.delay_freq == 0:
                    network.train_actor_and_targets(states, actions, returns.reshape(-1, 1))

        # Periodic evaluation
        performance = np.mean([evaluate_episode() for _ in range(args.evaluate_for)])

        if performance > args.pass_limit:
            training = False
            if args.save_model: 
                network.save(args.save_model)
        elif args.save_model and args.save_limit is not None and performance >= args.save_limit:
            network.save(args.save_model + "{:.2f}".format(performance))

    print(args)
    print("\nThe training is finished.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    print(args)
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
