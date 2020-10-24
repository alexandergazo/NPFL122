#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.05, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.85, type=float, help="Discounting factor.")
parser.add_argument("--q_path", default="Q.npy", type=str, help="Path to learned Q matrix.")
parser.add_argument("--episodes", default=5000, type=int, help="Number of training episodes.")


def main(env, args):
    # Fix random seed
    if args.q_path is None:
        np.random.seed(args.seed)
        env.action_space.seed(args.seed)

        Q = np.zeros((env.observation_space.n, env.action_space.n))

        for episode in range(args.episodes):
            epsilon = args.epsilon * 2 ** (- episode // 1000)
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()
                action = env.action_space.sample() if np.random.uniform() < epsilon else np.argmax(Q[state, :])
                next_state, reward, done, _ = env.step(action)
                Q[state, action] += args.alpha * (reward + args.gamma * np.max(Q[next_state, :]) - Q[state, action])
                state = next_state
        np.save("Q", Q)

    else:
        Q = np.load(args.q_path)

    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0")), args.seed)

    main(env, args)
