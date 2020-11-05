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
parser.add_argument("--alpha", default=0.15, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=10000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
parser.add_argument("--weights", default="q_learning_tiles.weights.npy", type=str, help="Path to weights.")


def main(env, args):
    def greedy(state):
        return W[state, :].sum(axis=0).argmax()

    # Fix random seed
    np.random.seed(args.seed)

    if args.weights is None:
        W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
        epsilon, alpha = args.epsilon, args.alpha

        try:
            while True:
                # Perform episode
                state, done = env.reset(), False
                while not done:
                    if args.render_each and env.episode and env.episode % args.render_each == 0:
                        env.render()
                    action = env.action_space.sample() if np.random.uniform() < epsilon else greedy(state)
                    next_state, reward, done, _ = env.step(action)
                    if done:
                        W[state, action] += alpha * (reward - W[state, action].sum())
                    else:
                        W[state, action] += alpha * (
                                reward + args.gamma * W[next_state, greedy(next_state)].sum() - W[state, action].sum())
                        state = next_state
                if args.epsilon_final_at:
                    epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
                    alpha = args.alpha * epsilon
        except KeyboardInterrupt:
            print("Training interrupted.")
            np.save("q_learning_tiles.weights.npy", W)
    else:
        W = np.load(args.weights)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = greedy(state)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles), args.seed)

    main(env, args)
