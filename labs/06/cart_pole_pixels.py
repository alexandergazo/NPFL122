#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=258, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="cart_pole_model.h5", type=str, help="Path to trained model.")

class Network:
    def __init__(self, env, args, model=None):
        if model is not None:
            self.model = model
            return

        inputs = tf.keras.Input(shape=env.observation_space.shape)
        hidden = tf.keras.layers.Conv2D(16, 8, 4)(inputs)
        hidden = tf.keras.layers.Conv2D(32, 4, 2)(hidden)
        hidden = tf.keras.layers.Reshape((-1,))(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(hidden)

        reinforce = tf.keras.layers.Dense(env.action_space.n, activation='softmax')(hidden)
        baseline = tf.keras.layers.Dense(1)(hidden)

        self.model = tf.keras.Model(inputs=inputs, outputs=[reinforce, baseline])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                           loss=[tf.keras.losses.SparseCategoricalCrossentropy(),
                                 tf.keras.losses.MeanSquaredError()])
        # tf.keras.utils.plot_model(self.model, show_shapes=True)


    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        with tf.GradientTape() as tape:
            predict_actions, baseline = self.model(states, training=True)
            baseline = tf.squeeze(baseline)
            loss1 = self.model.loss[0](actions, predict_actions,
                                      sample_weight= returns - tf.stop_gradient(baseline))
            loss2 = self.model.loss[1](returns, baseline)
            loss = 100 * loss1 + loss2

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self.model(states)[0]


def main(env, args):
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if os.path.isfile(args.model_path):
        network = Network(env, args, tf.keras.models.load_model(args.model_path))
    else:
        network = Network(env, args)

        # Fix random seeds and number of threads
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

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

        network.model.save(args.model_path)

    # Final evaluation
    while True:
        state, done, env_start_episode = env.reset(True), False, env.episode
        while not done:
            action = network.predict([state])[0].argmax()
            state, reward, done, _ = env.step(action)
        if env.episode - env_start_episode >= 100:
            return


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
