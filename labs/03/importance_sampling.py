#!/usr/bin/env python3
import argparse

import gym
import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")


def main(args):
    # Create the environment
    env = gym.make("FrozenLake-v0")
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    # Fix random seed
    np.random.seed(args.seed)

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(env.observation_space.n)
    C = np.zeros(env.observation_space.n)

    for i in range(args.episodes):
        state, done = env.reset(), False

        # Generate episode
        episode = []
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G, W = 0, 1
        for state, action, reward in reversed(episode):
            W *= env.action_space.n / 2 if 1 <= action <= 2 else 0
            if W == 0:
                break
            G = args.gamma * G + reward
            C[state] += W
            V[state] += W / C[state] * (G - V[state])

    return V


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    value_function = main(args)

    # Print the final value function V
    for row in value_function.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
