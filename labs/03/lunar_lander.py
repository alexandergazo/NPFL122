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
parser.add_argument("--alpha", default=0.008, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.01, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--q_path", default="Q_lunar.npy", type=str, help="Path to learned Q matrix.")
parser.add_argument("--episodes", default=1000, type=int, help="Number of training episodes.")
parser.add_argument("-n", default=4, type=int, help="N for N-step Tree Backup.")


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


def main(env, args):
    def tree_me_pls(Q, expert=False):
        from tqdm import trange
        for _ in trange(args.episodes):
            if expert:
                expert = False
            else:
                expert = True
            # expert = False
            if expert:
                S, A, R = [None], [], []
                S[0], trajectory = env.expert_trajectory()
                for a, r, s in trajectory:
                    S.append(s)
                    A.append(a)
                    R.append(r)
            else:
                S, A, R = RemainderList(args.n), RemainderList(args.n), RemainderList(args.n)
                S[0] = env.reset()
                A[0] = env.action_space.sample() if np.random.uniform() < args.epsilon else np.argmax(Q[S[0], :])
            t, T = 0, np.inf
            while True:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()
                if t < T:
                    if not expert:
                        S[t + 1], R[t + 1], done, _ = env.step(A[t])
                    else:
                        done = t + 1 == (len(R) - 1)
                    if done:
                        T = t + 1
                    elif not expert:
                        A[t + 1] = env.action_space.sample() if np.random.uniform() < args.epsilon else np.argmax(
                            Q[S[t + 1], :])
                tau = t + 1 - args.n
                if tau >= 0:
                    if t + 1 >= T:
                        G = R[T]
                    else:
                        G = R[t + 1] + args.gamma * np.max(Q[S[t + 1], :])  # greedy
                    k = min(t, T - 1)
                    while k >= tau + 1:
                        G = R[k] + args.gamma * (G if np.argmax(Q[S[k], :]) == A[k] else np.max(Q[S[k], :]))  # greedy
                        k -= 1
                    Q[S[tau], A[tau]] += args.alpha * (G - Q[S[tau], A[tau]])
                if tau == T - 1:
                    break
                t += 1

    if args.q_path is None:
        np.random.seed(args.seed)
        env.action_space.seed(args.seed)

        Q = np.zeros((env.observation_space.n, env.action_space.n))
        Q = np.load("Q_lunar_4.npy")

        tree_me_pls(Q, False)

        np.save("Q_lunar_5", Q)
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
    env = wrappers.EvaluationWrapper(wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed)

    main(env, args)
