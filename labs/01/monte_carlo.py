#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", default=0, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")


def _greedy(epsilon):
    return np.random.uniform() >= epsilon


def main(env, args):
    # Fix random seed
    np.random.seed(args.seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    C = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(args.episodes):
        state, done = env.reset(), False
        states, actions, rewards = [], [], []
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            states.append(state)

            action = np.argmax(Q[state]) if _greedy(args.epsilon) else np.random.randint(env.action_space.n)
            actions.append(action)

            state, reward, done, _ = env.step(action)
            rewards.append(reward)

        G = 0
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            G += reward
            C[state, action] += 1
            Q[state, action] += 1 / C[state, action] * (G - Q[state, action])

    Q = np.load("q.npy")
    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args()

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed)

    main(env, args)

