#!/usr/bin/env python3
import argparse

import numpy as np


# Print results
def plot(value_function, policy):
    for l in range(3):
        for c in range(4):
            state = l * 4 + c
            if state >= 5: state -= 1
            print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if l == 1 and c == 1 else GridWorld.actions_graphics[policy[state]], end="")
        print()


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
    rewards = [0, 1, -100]
    actions_graphics = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state, action):
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability, state, action):
        if state >= 5: state += 1
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        if state >= 5: state -= 1
        return [probability, +1 if state == 10 else -100 if state == 6 else 0, state]


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.95, type=float, help="Discount factor.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")


# If you add more arguments, ReCodEx will keep them with your default values.

def pi(action, desired_action):
    return 1 if action == desired_action else 0
    # return 0.8 if action == desired_action else 0 if (action + 2) % 4 == desired_action else 0.1


def main(args):
    V = [0] * GridWorld.states
    policy = [0] * GridWorld.states

    for _ in range(args.steps):
        A = -np.eye(GridWorld.states)
        b = np.zeros((GridWorld.states,))

        for state in range(GridWorld.states):
            for p, r, s in GridWorld.step(state, policy[state]):
                b[state] -= p * r
                A[state, s] += args.gamma * p
        V = np.linalg.solve(A, b)

        for state in range(GridWorld.states):
            policy[state] = np.argmax([sum(p * (r + args.gamma * V[s]) for p, r, s in GridWorld.step(state, action))
                                       for action in range(GridWorld.actions)])

    return V, policy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    plot(value_function, policy)
