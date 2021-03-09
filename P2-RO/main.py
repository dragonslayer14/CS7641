import sys
from datetime import datetime
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import six
sys.modules['sklearn.externals.six'] = six
from mlrose import NeuralNetwork, random_hill_climb, simulated_annealing, genetic_alg, mimic, FlipFlop, SixPeaks, \
    Queens, DiscreteOpt
import numpy as np
import time


random_states = [0,50,800,35]


def average_curves(fit_curves):
    length = max([len(curve) for curve in fit_curves])
    return np.stack([
        np.pad(curve, (0, length - len(curve)), 'edge')
        for curve in fit_curves
    ]).mean(axis=0)


# TODO make 3 versions of each function, 1 for each problem
# just call each set of algos for each problem, each can be tuned
#    need HP tuning charts? think so, double check notes
#    tuning by grid search, mlrose equiv? or just plot fitness over a range of values for an HP that should affect it?
# probably need 3 sizes of problem for each? small, medium, large? check notes
#    would run algo for each size and plot fitness to size, maybe all algos on one plot for space

def run_RHC_1(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = random_hill_climb(problem, random_state=random_state, curve=True, init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"RHC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_GA_1(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = genetic_alg(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"GA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_MIMIC_1(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = mimic(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"MIMIC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_SA_1(problem, init_state):
    start = time.time()
    fit_vals = []
    fit_curves = []
    times = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = simulated_annealing(problem, random_state=random_state, curve=True,
                                                   init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"SA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_RHC_2(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = random_hill_climb(problem, random_state=random_state, curve=True, init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"RHC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_GA_2(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = genetic_alg(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"GA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_MIMIC_2(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = mimic(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"MIMIC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_SA_2(problem, init_state):
    start = time.time()
    fit_vals = []
    fit_curves = []
    times = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = simulated_annealing(problem, random_state=random_state, curve=True,
                                                   init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"SA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_RHC_3(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = random_hill_climb(problem, random_state=random_state, curve=True, init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"RHC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_GA_3(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve = genetic_alg(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)

    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"GA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_MIMIC_3(problem, init_state):
    fit_vals = []
    fit_curves = []
    times = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = mimic(problem, random_state=random_state, curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"MIMIC {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_SA_3(problem, init_state):
    start = time.time()
    fit_vals = []
    fit_curves = []
    times = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve = simulated_annealing(problem, random_state=random_state, curve=True,
                                                   init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)


    # plot average fitness value
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    print(f"SA {problem_name}: {avg_fit}: {np.mean(times):.2f}")


def run_ANN():
    # TODO use multiple random states and average values
    random_state = 0
    _, loss, _, fit_curve = NeuralNetwork(curve=True, random_state=random_state)


def run_ANN_RHC():
    # TODO use multiple random states and average values
    random_state = 0
    _, loss, _, fit_curve = NeuralNetwork(algorithm="random_hill_climb", curve=True, random_state=random_state)


def run_ANN_SA():
    # TODO use multiple random states and average values
    random_state = 0
    _, loss, _, fit_curve = NeuralNetwork(algorithm="simulated_annealing", curve=True, random_state=random_state)


def run_ANN_GA():
    # TODO use multiple random states and average values
    random_state = 0
    _, loss, _, fit_curve = NeuralNetwork(algorithm="genetic_alg", curve=True, random_state=random_state)


if __name__ == "__main__":
    # TODO need to break these and the functions up, since each has to be tuned to problem
    # TODO switch plotting to be only average score per problem, then the chart will be fitness over problem size
    np.random.seed(0)

    # problem 1
    # init_state = np.random.randint(0,2,12)
    init_state = np.random.randint(0,2,50)
    # init_state = np.random.randint(0,2,100)
    plt.figure()
    problem = DiscreteOpt(length = len(init_state), fitness_fn = FlipFlop())
    run_RHC_1(problem, init_state)
    run_SA_1(problem, init_state)
    run_GA_1(problem, init_state)
    run_MIMIC_1(problem, init_state)
    print()
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    plt.title(f"charts/{problem_name} ({len(init_state)})")
    plt.xlabel("step")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(problem_name)
    plt.show()
    plt.close('all')

    # problem 2
    init_state = np.random.randint(0,2,12)
    # init_state = np.random.randint(0, 2, 20)
    plt.figure()
    problem = DiscreteOpt(length = len(init_state), fitness_fn = SixPeaks())
    run_RHC_2(problem, init_state)
    run_SA_2(problem, init_state)
    run_GA_2(problem, init_state)
    run_MIMIC_2(problem, init_state)
    print()
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    plt.title(f"charts/{problem_name} ({len(init_state)})")
    plt.xlabel("step")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(problem_name)
    plt.show()
    plt.close('all')

    # problem 3
    init_state = np.random.randint(0,8,8)
    # init_state = np.random.randint(0, 15, 15)
    plt.figure()
    problem = DiscreteOpt(length = len(init_state), fitness_fn = Queens())
    run_RHC_3(problem, init_state)
    run_SA_3(problem, init_state)
    run_GA_3(problem, init_state)
    run_MIMIC_3(problem, init_state)
    print()
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    plt.title(f"charts/{problem_name} ({len(init_state)})")
    plt.xlabel("step")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(problem_name)
    plt.show()
    plt.close('all')

    # ANN compare
    run_ANN()
    run_ANN_RHC()
    run_ANN_SA()
    run_ANN_GA()
