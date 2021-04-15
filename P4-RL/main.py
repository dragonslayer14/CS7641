import time
import warnings

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from hiive.mdptoolbox import mdp, example


def run_pi(transition, reward, discount, max_iter=1000):
    pi = mdp.PolicyIterationModified(transition, reward, discount, max_iter=max_iter)
    pi.run()
    print(pi.policy)
    print(pi.iter)


def run_vi(transition, reward, discount, epsilon=0.01, max_iter=1000):
    # In verbose mode, at each iteration, displays the variation of V and the condition which stopped iterations:
    # epsilon-optimum policy found or maximum number of iterations reached.
    vi = mdp.ValueIteration(transition, reward, discount,epsilon,max_iter)
    print(vi.verbose)

    vi.run()

    print(vi.policy)

    print(vi.iter)


def run_ql(transition, reward, discount, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1, max_iter=10000):
    ql = mdp.QLearning(transition, reward, discount, epsilon=epsilon, epsilon_decay=epsilon_decay,
                       epsilon_min=epsilon_min, n_iter=max_iter, skip_check=False)
    ql.run()
    # print(ql.Q)
    print(ql.policy)


if __name__ == '__main__':
    
    np.random.seed(0)
    
    # grid world, frozen lake
    random_map = generate_random_map(size=20, p=0.8)

    env = gym.make("FrozenLake-v0", desc=random_map)
    env.reset()
    env.render()
    
    num_states = len(env.env.P)
    num_actions = len(env.env.P[0])
    
    transition = np.zeros((num_actions, num_states, num_states))
    rewards = np.zeros((num_states, num_actions))

    # convert transition matrix dict of dicts of lists to rewards matrix
    # frozen lake has a mostly 0 matrix, might be worth looking at sparse if it gets really big
    for state in env.env.P:
        for action in env.env.P[state]:
            for prob, s_prime, reward, _ in env.env.P[state][action]:
                transition[action][state][s_prime] += prob
                rewards[state][action] += reward
    
    # solve with VI
    run_vi(transition, rewards, 0.96)
    # solve with PI
    run_pi(transition, rewards, 0.96, max_iter=1000)
    # solve with QL
    run_ql(transition, rewards, 0.9, max_iter=100000)

    # non grid world, forest
    P, R = example.forest()
    # solve with VI
    run_vi(P, R, 0.96)
    # solve with PI
    run_pi(P, R, 0.9)
    # solve with QL
    run_ql(P, R, 0.9)

    pass