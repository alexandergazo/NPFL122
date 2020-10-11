#!/usr/bin/env python3
import argparse

import numpy as np


class MultiArmedBandits():
    def __init__(self, bandits, seed=None):
        self._generator = np.random.RandomState(seed)
        self._bandits = [None] * bandits
        self.reset()

    def reset(self):
        for i in range(len(self._bandits)):
            self._bandits[i] = self._generator.normal(0., 1.)

    def step(self, action):
        return self._generator.normal(self._bandits[action], 1.)

    def greedy(self, epsilon):
        return self._generator.uniform() >= epsilon


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate to use or 0 for averaging.")
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--episodes", default=10000, type=int, help="Episodes to perform.")
parser.add_argument("--epsilon", default=0, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=1, type=float, help="Initial estimation of values.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")


def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    Q = [args.initial for i in range(args.bandits)]
    N = [0 for i in range(args.bandits)]

    rewards = 0
    for step in range(args.episode_length):
        action = np.argmax(Q) if env.greedy(args.epsilon) else np.random.randint(args.bandits)
        N[action] += 1

        reward = env.step(action)
        rewards += reward

        factor = args.alpha if args.alpha != 0 else 1 / N[action]
        Q[action] += factor * (reward - Q[action])

    return rewards / args.episode_length


if __name__ == "__main__":
    args = parser.parse_args()

    # Create the environment
    env = MultiArmedBandits(args.bandits, seed=args.seed)

    returns = []
    for _ in range(args.episodes):
        returns.append(main(env, args))

    # Print the mean and std
    print("{:.2f} {:.2f}".format(np.mean(returns), np.std(returns)))
