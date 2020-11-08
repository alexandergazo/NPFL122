#!/usr/bin/env python3
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")


# If you add more arguments, ReCodEx will keep them with your default values.
class RemainderList:
    def __init__(self, n):
        self._n = n
        self._list = [None] * (self._n + 1)

    def __getitem__(self, item):
        return self._list[item % (self._n + 1)]

    def __setitem__(self, key, value):
        self._list[key % (self._n + 1)] = value

    def __str__(self):
        return str(self._list)


def acts(env):
    return range(env.action_space.n)


def main(args):
    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("Taxi-v3"), seed=args.seed, report_each=100)

    # Fix random seed and create a generator
    generator = np.random.RandomState(args.seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(args.episodes):
        t, T, S, A, R = 0, np.inf, RemainderList(args.n), RemainderList(args.n), RemainderList(args.n)
        b = RemainderList(args.n)
        S[0], done = env.reset(), False
        A[0] = env.action_space.sample() if generator.uniform() < args.epsilon else Q[S[0]].argmax()
        b[0] = args.epsilon / env.action_space.n + (1 - args.epsilon) * (A[0] == Q[S[0]].argmax())
        while True:
            if t < T:
                S[t + 1], R[t + 1], done, _ = env.step(A[t])
                if not done:
                    A[t + 1] = env.action_space.sample() if generator.uniform() < args.epsilon else Q[S[t + 1]].argmax()
                    b[t + 1] = args.epsilon / env.action_space.n + (1 - args.epsilon) * (
                            A[t + 1] == Q[S[t + 1]].argmax())
                else:
                    T = t + 1
                pi = np.eye(env.action_space.n)[Q.argmax(axis=1)]
                if not args.off_policy:
                    pi = (1 - args.epsilon) * pi + args.epsilon / env.action_space.n * np.ones_like(pi)

            tau = t + 1 - args.n
            if tau >= 0:
                if args.mode == 'tree_backup':
                    if t + 1 >= T:
                        G = R[T]
                    else:
                        G = R[t + 1] + args.gamma * (pi[S[t + 1]].T @ Q[S[t + 1]])
                    k = min(t, T - 1)
                    while k >= tau + 1:
                        G = R[k] + args.gamma * sum(pi[S[k], a] * (G if A[k] == a else Q[S[k], a]) for a in acts(env))
                        k -= 1
                    Q[S[tau], A[tau]] += args.alpha * (G - Q[S[tau], A[tau]])
                else:
                    if args.off_policy:
                        if args.mode == 'sarsa':
                            r = range(tau + 1, min(tau + args.n, T - 1) + 1)
                        elif args.mode == 'expected_sarsa':
                            r = range(tau + 1, min(tau + args.n, T))
                        rho = np.prod([pi[S[i], A[i]] / b[i] for i in r])
                    else:
                        rho = 1
                    G = sum(args.gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, min(tau + args.n, T) + 1))
                    if args.mode == 'sarsa':
                        bootstrapped_value = Q[S[tau + args.n], A[tau + args.n]]
                    elif args.mode == 'expected_sarsa':
                        bootstrapped_value = pi[S[tau + args.n]].T @ Q[S[tau + args.n]]
                    if tau + args.n < T:
                        G += args.gamma ** args.n * bootstrapped_value
                    Q[S[tau], A[tau]] += args.alpha * rho * (G - Q[S[tau], A[tau]])

            if tau == T - 1:
                break
            t += 1
    return Q


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
