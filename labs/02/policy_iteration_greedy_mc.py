#!/usr/bin/env python3
import argparse

import numpy as np


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states = 11
    actions = 4
    actions_graphics = ["↑", "→", "↓", "←"]

    def __init__(self, seed):
        self._generator = np.random.RandomState(seed)

    def step(self, state, action):
        probability = self._generator.uniform()
        if probability <= 0.8: return self._step(state, action)
        if probability <= 0.9: return self._step(state, (action + 1) % 4)
        return self._step(state, (action + 3) % 4)

    def epsilon_greedy(self, epsilon, greedy_action):
        if self._generator.uniform() < epsilon:
            return self._generator.randint(len(self.actions_graphics))
        return greedy_action

    @staticmethod
    def _step(state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return (+1 if state == 10 else -100 if state == 6 else 0, state)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
parser.add_argument("--epsilon", default=0.02, type=float, help="Monte Carlo epsilon")
parser.add_argument("--mc_length", default=100, type=int, help="Monte Carlo simulation episode length")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")


# If you add more arguments, ReCodEx will keep them with your default values.


def main(args):
    env = GridWorld(args.seed)

    Q = np.zeros((env.states, env.actions))
    policy = np.zeros(env.states, np.int32)
    returns = [[[] for _ in range(env.actions)] for _ in range(env.states)]

    for _ in range(args.steps):
        for state in range(env.states):
            s = state
            rewards = []
            first_action = None
            for _ in range(args.mc_length):
                action = env.epsilon_greedy(args.epsilon, policy[s])
                if first_action is None:
                    first_action = action
                reward, s = env.step(s, action)
                rewards.append(reward)

            G = 0
            for reward in reversed(rewards):
                G = args.gamma * G + reward
            returns[state][first_action].append(G)
            Q[state, first_action] = np.mean(returns[state][first_action])

        for state in range(env.states):
            policy[state] = np.argmax(Q[state, :])

    value_function = [Q[state, policy[state]] for state in range(env.states)]
    return value_function, policy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    # Print results
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if l == 1 and c == 1 else GridWorld.actions_graphics[policy[state]], end="")
        print()
