import time
import warnings

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
from hiive.mdptoolbox import mdp, example
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt


if __name__ == '__main__':
    start = time.time()
    np.random.seed(0)
    
    # TODO probably should make multiple sizes, will make tuning ql a pain because so many iterations will be needed
    # grid world, frozen lake, small 225 states
    random_map = generate_random_map(size=15, p=0.8)

    env = gym.make("FrozenLake-v0", desc=random_map)
    env.reset()
    env.render()

    num_states = len(env.env.P)
    num_actions = len(env.env.P[0])

    transitions = np.zeros((num_actions, num_states, num_states))
    rewards = np.zeros((num_states, num_actions))

    # convert transition matrix dict of dicts of lists to rewards matrix
    # frozen lake has a mostly 0 matrix, might be worth looking at sparse if it gets really big
    for state in env.env.P:
        for action in env.env.P[state]:
            for prob, s_prime, reward, _ in env.env.P[state][action]:
                transitions[action][state][s_prime] += prob
                rewards[state][action] += reward

    # tune PI/VI gamma values
    tune_gamma = False
    if tune_gamma:
        gamma_range = np.linspace(0.01, 0.99, 99)
        vi_iter = []
        pi_iter = []
        vi_time =[]
        pi_time = []
        vi_max_v = []
        pi_max_v = []
        
        for gamma in gamma_range:
            vi = mdp.ValueIteration(transitions, rewards, gamma, max_iter=10000)
            vi.run()
            vi_time.append(vi.time)
            vi_max_v.append(np.max(vi.V))
            vi_iter.append(vi.iter)
    
            pi = mdp.PolicyIterationModified(transitions, rewards, gamma, max_iter=1000)
            pi.run()
            pi_time.append(pi.time)
            pi_max_v.append(np.max(pi.V))
            pi_iter.append(pi.iter)
    
        plt.figure()
        plt.plot(gamma_range, vi_iter, label="VI")
        plt.plot(gamma_range, pi_iter, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('Iterations')
        plt.title('Gamma vs iterations')
        plt.legend()

        plt.figure()
        plt.plot(gamma_range, vi_time, label="VI")
        plt.plot(gamma_range, pi_time, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('time')
        plt.title('Gamma vs time')
        plt.legend()

        plt.figure()
        plt.plot(gamma_range, vi_max_v, label="VI")
        plt.plot(gamma_range, pi_max_v, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('max v')
        plt.title('Gamma vs max v')
        plt.legend()
        
        plt.show()
    
    # tune VI/PI epsilon as stopping value
    tune_epsilon = False
    if tune_epsilon:
        epsilon_range = np.arange(0.0001, 0.05, 0.005)
        vi_iter = []
        pi_iter = []
        vi_time = []
        pi_time = []
        vi_max_v = []
        pi_max_v = []
    
        for epsilon in epsilon_range:
            vi = mdp.ValueIteration(transitions, rewards, gamma=0.99, epsilon=epsilon, max_iter=10000)
            vi.run()
            vi_time.append(vi.time)
            vi_max_v.append(np.max(vi.V))
            vi_iter.append(vi.iter)
        
            pi = mdp.PolicyIterationModified(transitions, rewards, gamma=0.99, epsilon=epsilon, max_iter=1000)
            pi.run()
            pi_time.append(pi.time)
            pi_max_v.append(np.max(pi.V))
            pi_iter.append(pi.iter)
    
        plt.figure()
        plt.plot(epsilon_range, vi_iter, label="VI")
        plt.plot(epsilon_range, pi_iter, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('Iterations')
        plt.title('epsilon vs iterations')
        plt.legend()

        plt.figure()
        plt.plot(epsilon_range, vi_time, label="VI")
        plt.plot(epsilon_range, pi_time, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('time')
        plt.title('epsilon vs time')
        plt.legend()

        plt.figure()
        plt.plot(epsilon_range, vi_max_v, label="VI")
        plt.plot(epsilon_range, pi_max_v, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('max v')
        plt.title('epsilon vs max v')
        plt.legend()

        plt.show()

    tune_ql = False
    if tune_ql:
        
        # max iter
        if False:
            iter_range = [10**4, 5*(10**4), 10**5, 5*(10**5), 10**6, 5*(10**6)]
            ql_time = []
            ql_max_v = []
    
            for iter in iter_range:
                ql = mdp.QLearning(transitions, rewards, gamma=0.99, epsilon=1.0, n_iter=iter)
                ql.run()
                ql_time.append(ql.time)
                ql_max_v.append(np.max(ql.V))
    
            plt.figure()
            plt.plot(iter_range, ql_time, label="QL")
            plt.xlabel('iterations')
            plt.ylabel('time')
            plt.title('iteration vs time')
            plt.legend()
            plt.savefig("charts/lake_ql_iter_time")
    
            plt.figure()
            plt.plot(iter_range, ql_max_v, label="QL")
            plt.xlabel('iterations')
            plt.ylabel('max v')
            plt.title('iterations vs max v')
            plt.legend()
            plt.savefig("charts/lake_ql_iter_max_v")
            plt.show()
        
        if False:
            gamma_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.99]
            ql_time = []
            ql_max_v = []
        
            for gamma in gamma_range:
                ql = mdp.QLearning(transitions, rewards, gamma=gamma, n_iter=10**6)
                ql.run()
                ql_time.append(ql.time)
                ql_max_v.append(np.max(ql.V))
            
            plt.figure()
            plt.plot(gamma_range, ql_time, label="QL")
            plt.xlabel('gamma')
            plt.ylabel('time')
            plt.title('Gamma vs time')
            plt.legend()
            plt.savefig("charts/lake_ql_gamma_time")
    
            plt.figure()
            plt.plot(gamma_range, ql_max_v, label="QL")
            plt.xlabel('gamma')
            plt.ylabel('max v')
            plt.title('Gamma vs max v')
            plt.legend()
            plt.savefig("charts/lake_ql_gamma_max_v")
    
            # plt.show()
        
        # epsilon
        epsilon_range = np.arange(0.1, 1.1, 0.1)
        ql_time = []
        ql_max_v = []
    
        for epsilon in epsilon_range:
            ql = mdp.QLearning(transitions, rewards, gamma=0.6, epsilon=epsilon, n_iter=10**6)
            ql.run()
            ql_time.append(ql.time)
            ql_max_v.append(np.max(ql.V))

        plt.figure()
        plt.plot(epsilon_range, ql_time, label="QL")
        plt.xlabel('epsilon')
        plt.ylabel('time')
        plt.title('epsilon vs time')
        plt.legend()
        plt.savefig("charts/lake_ql_epsilon_time")

        plt.figure()
        plt.plot(epsilon_range, ql_max_v, label="QL")
        plt.xlabel('epsilon')
        plt.ylabel('max v')
        plt.title('epsilon vs max v')
        plt.legend()
        plt.savefig("charts/lake_ql_epsilon_max_v")

        plt.show()

    run_lake = False
    if run_lake:
        # solve with VI
        # run_vi(transition, reward, 0.96)
        vi = mdp.ValueIteration(transitions, rewards, gamma=0.99, epsilon=0.0001)
        vi.run()
        print(vi.policy)
        print(vi.iter)
    
        # solve with PI
        # run_pi(transition, reward, 0.96, )
        pi = mdp.PolicyIterationModified(transitions, rewards, gamma=0.99, max_iter=1000, epsilon=0.0001)
        pi.run()
        print(pi.policy)
        print(pi.iter)
    
        # solve with QL
        # run_ql(transition, reward, 0.96, epsilon=.95, max_iter=100000)
        ql = mdp.QLearning(transitions, rewards, gamma=0.6, epsilon=0.9, n_iter=10**6)
        res = ql.run()
        # print(ql.Q)
        print(ql.time)
        print(ql.policy)
        print(ql.V)
        # print(ql.v_mean)
        # (2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 1, 3, 0)

    # non grid world, forest, large, 5000 states
    transitions, rewards = example.forest(S=5000, r1=10)

    # tune PI/VI gamma values
    tune_gamma = True
    if tune_gamma:
        gamma_range = np.linspace(0.01, 0.99, 99)
        vi_iter = []
        pi_iter = []
        vi_time = []
        pi_time = []
        vi_max_v = []
        pi_max_v = []
    
        for gamma in gamma_range:
            vi = mdp.ValueIteration(transitions, rewards, gamma, max_iter=10000)
            vi.run()
            vi_time.append(vi.time)
            vi_max_v.append(np.max(vi.V))
            vi_iter.append(vi.iter)
        
            pi = mdp.PolicyIterationModified(transitions, rewards, gamma, max_iter=1000)
            pi.run()
            pi_time.append(pi.time)
            pi_max_v.append(np.max(pi.V))
            pi_iter.append(pi.iter)
    
        plt.figure()
        plt.plot(gamma_range, vi_iter, label="VI")
        plt.plot(gamma_range, pi_iter, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('Iterations')
        plt.title('Gamma vs iterations')
        plt.legend()
        plt.savefig("charts/forest_gamma_iter")

        plt.figure()
        plt.plot(gamma_range, vi_time, label="VI")
        plt.plot(gamma_range, pi_time, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('time')
        plt.title('Gamma vs time')
        plt.legend()
        plt.savefig("charts/forest_gamma_time")

        plt.figure()
        plt.plot(gamma_range, vi_max_v, label="VI")
        plt.plot(gamma_range, pi_max_v, label="PI")
        plt.xlabel('gamma')
        plt.ylabel('max v')
        plt.title('Gamma vs max v')
        plt.legend()
        plt.savefig("charts/forest_gamma_max_v")

        # plt.show()

    # tune VI/PI epsilon as stopping value
    tune_epsilon = True
    if tune_epsilon:
        epsilon_range = np.arange(0.0001, 0.05, 0.005)
        vi_iter = []
        pi_iter = []
        vi_time = []
        pi_time = []
        vi_max_v = []
        pi_max_v = []
    
        for epsilon in epsilon_range:
            vi = mdp.ValueIteration(transitions, rewards, gamma=0.99, epsilon=epsilon, max_iter=10000)
            vi.run()
            vi_time.append(vi.time)
            vi_max_v.append(np.max(vi.V))
            vi_iter.append(vi.iter)
        
            pi = mdp.PolicyIterationModified(transitions, rewards, gamma=0.99, epsilon=epsilon, max_iter=1000)
            pi.run()
            pi_time.append(pi.time)
            pi_max_v.append(np.max(pi.V))
            pi_iter.append(pi.iter)
    
        plt.figure()
        plt.plot(epsilon_range, vi_iter, label="VI")
        plt.plot(epsilon_range, pi_iter, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('Iterations')
        plt.title('epsilon vs iterations')
        plt.legend()
        plt.savefig("charts/forest_epsilon_iter")
    
        plt.figure()
        plt.plot(epsilon_range, vi_time, label="VI")
        plt.plot(epsilon_range, pi_time, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('time')
        plt.title('epsilon vs time')
        plt.legend()
        plt.savefig("charts/forest_epsilon_time")

        plt.figure()
        plt.plot(epsilon_range, vi_max_v, label="VI")
        plt.plot(epsilon_range, pi_max_v, label="PI")
        plt.xlabel('epsilon')
        plt.ylabel('max v')
        plt.title('epsilon vs max v')
        plt.legend()
        plt.savefig("charts/forest_epsilon_max_v")

        # plt.show()

    tune_ql = False
    if tune_ql:
    
        # max iter
        iter_range = [10 ** 4, 5 * (10 ** 4), 10 ** 5, 5 * (10 ** 5), 10 ** 6, 5*(10**6)]
        ql_time = []
        ql_max_v = []
    
        for iter in iter_range:
            ql = mdp.QLearning(transitions, rewards, gamma=0.99, epsilon=1.0, n_iter=iter)
            ql.run()
            ql_time.append(ql.time)
            ql_max_v.append(np.max(ql.V))
    
        plt.figure()
        plt.plot(iter_range, ql_time, label="QL")
        plt.xlabel('iterations')
        plt.ylabel('time')
        plt.title('iteration vs time')
        plt.legend()
        plt.savefig("charts/forest_ql_iter_time")
    
        plt.figure()
        plt.plot(iter_range, ql_max_v, label="QL")
        plt.xlabel('iterations')
        plt.ylabel('max v')
        plt.title('iterations vs max v')
        plt.legend()
        plt.savefig("charts/forest_ql_iter_max_v")
        
        gamma_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.99]
        ql_time = []
        ql_max_v = []
    
        for gamma in gamma_range:
            ql = mdp.QLearning(transitions, rewards, gamma=gamma, n_iter=100000)
            ql.run()
            ql_time.append(ql.time)
            ql_max_v.append(np.max(ql.V))
    
        plt.figure()
        plt.plot(gamma_range, ql_time, label="QL")
        plt.xlabel('gamma')
        plt.ylabel('time')
        plt.title('Gamma vs time')
        plt.legend()
        plt.savefig("charts/forest_ql_gamma_time")
    
        plt.figure()
        plt.plot(gamma_range, ql_max_v, label="QL")
        plt.xlabel('gamma')
        plt.ylabel('max v')
        plt.title('Gamma vs max v')
        plt.legend()
        plt.savefig("charts/forest_ql_gamma_max_v")
    
        # plt.show()
    
        # epsilon
        epsilon_range = np.arange(0.1, 1.1, 0.1)
        ql_time = []
        ql_max_v = []
    
        for epsilon in epsilon_range:
            ql = mdp.QLearning(transitions, rewards, gamma=0.99, epsilon=epsilon, n_iter=100000)
            ql.run()
            ql_time.append(ql.time)
            ql_max_v.append(np.max(ql.V))
    
        plt.figure()
        plt.plot(epsilon_range, ql_time, label="QL")
        plt.xlabel('epsilon')
        plt.ylabel('time')
        plt.title('epsilon vs time')
        plt.legend()
        plt.savefig("charts/forest_ql_epsilon_time")
    
        plt.figure()
        plt.plot(epsilon_range, ql_max_v, label="QL")
        plt.xlabel('epsilon')
        plt.ylabel('max v')
        plt.title('epsilon vs max v')
        plt.legend()
        plt.savefig("charts/forest_ql_epsilon_max_v")
    
    plt.show()

    # solve with VI
    vi = mdp.ValueIteration(transitions, rewards, gamma=0.99, epsilon=0.0001, max_iter=10000)
    print(vi.verbose)
    vi.run()
    print(vi.policy)
    print(vi.iter)

    # solve with PI
    pi = mdp.PolicyIterationModified(transitions, rewards, gamma=0.99, epsilon=0.0001, max_iter=1000)
    pi.run()
    print(pi.policy)
    print(pi.iter)

    # solve with QL
    ql = mdp.QLearning(transitions, rewards, gamma=0.99, epsilon=1.0, n_iter=100000)
    res = ql.run()
    # print(ql.Q)
    print(ql.policy)
    print(ql.V)
    print(ql.time)
    
    print(f"took {time.time()-start:.2f}")