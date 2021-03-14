import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
from mlrose_hiive import NeuralNetwork, random_hill_climb, simulated_annealing, genetic_alg, mimic, SixPeaks, \
    Queens, DiscreteOpt, MaxKColorGenerator, MaxKColor, GeomDecay
import numpy as np
import time

random_states = [0,50,800,35]


# def average_curves(fit_curves):
#     length = max([len(curve) for curve in fit_curves])
#     return np.stack([
#         np.pad(curve, (0, length - len(curve)), 'edge')
#         for curve in fit_curves
#     ]).mean(axis=0)


# TODO make 3 versions of each function, 1 for each problem
# just call each set of algos for each problem, each can be tuned
#    need HP tuning charts? think so, double check notes
#    tuning by grid search, mlrose equiv? or just plot fitness over a range of values for an HP that should affect it?
# probably need 3 sizes of problem for each? small, medium, large? check notes
#    would run algo for each size and plot fitness to size, maybe all algos on one plot for space

def run_RHC_1(problem, init_state, **kwargs):
    fit_vals = []
    # fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, _, evals = random_hill_climb(problem, random_state=random_state, restarts=60, max_attempts=12,
                                                  **kwargs, curve=True, fevals=True, init_state=init_state)

        fit_vals.append(best_fit)
        # fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"
    #
    # plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"RHC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_GA_1(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve, evals = genetic_alg(problem, random_state=random_state, curve=True,fevals=True,
                                                    pop_size=120, mutation_prob=0.12, **kwargs)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"GA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_MIMIC_1(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = mimic(problem, random_state=random_state, pop_size=150, keep_pct=0.15,
                                              **kwargs, curve=True, fevals=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"MIMIC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_SA_1(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = simulated_annealing(problem, random_state=random_state, curve=True,
                                                   init_state=init_state, fevals=True, **kwargs)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"SA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_RHC_2(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve, evals = random_hill_climb(problem, random_state=random_state, restarts=35,
                                                          **kwargs, curve=True,fevals=True,  init_state=init_state)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"RHC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_GA_2(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve, evals = genetic_alg(problem, random_state=random_state, pop_size=210, mutation_prob=0.23,
                                                    **kwargs,fevals=True,  curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"GA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_MIMIC_2(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = mimic(problem, random_state=random_state, pop_size=300, keep_pct=0.2,
                                              **kwargs,fevals=True,  curve=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"MIMIC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_SA_2(problem, init_state, **kwargs):
    start = time.time()
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = simulated_annealing(problem, random_state=random_state, **kwargs, curve=True,fevals=True,
                                                   init_state=init_state, schedule=GeomDecay(init_temp=11, decay=0.96))

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"SA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_RHC_3(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve, evals = random_hill_climb(problem, random_state=random_state, restarts=39,
                                                          **kwargs, curve=True, init_state=init_state, fevals=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/rhc_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="rhc")
    # plt.title(f"RHC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"RHC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_GA_3(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()
        _, best_fit, fit_curve, evals = genetic_alg(problem, random_state=random_state, pop_size=170, mutation_prob=0.82,
                                                    **kwargs, curve=True, fevals=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))

    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/ga_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="ga")
    # plt.title(f"GA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"GA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_MIMIC_3(problem, init_state, **kwargs):
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []

    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = mimic(problem, random_state=random_state, pop_size=275,
                                              **kwargs, curve=True, fevals=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/mimic_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="mimic")
    # plt.title(f"MIMIC {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"MIMIC {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_SA_3(problem, init_state, **kwargs):
    start = time.time()
    fit_vals = []
    fit_curves = []
    times = []
    fevals = []


    # run multiple times to get average
    for random_state in random_states:
        start = time.time()

        _, best_fit, fit_curve, evals = simulated_annealing(problem, random_state=random_state, schedule=GeomDecay(decay=0.94),
                                                            **kwargs, curve=True, init_state=init_state, fevals=True)

        fit_vals.append(best_fit)
        fit_curves.append(fit_curve)
        times.append(time.time() - start)
        fevals.append(sum(evals.values()))


    # plot average fitness value
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    # hack for ease of naming
    problem_name = str(problem.fitness_fn).split('.')[-1].split(' ')[0]
    # chart_name = f"charts/sa_{problem_name}_{len(init_state)}_{dt_string}"

    # plt.plot(average_curves(fit_curves), label="sa")
    # plt.title(f"SA {problem_name} ({len(init_state)})")
    # plt.xlabel("step")
    # plt.ylabel("fitness")
    # plt.savefig(chart_name)
    # plt.show()

    avg_fit = np.average(fit_vals)
    avg_time = round(np.mean(times), 2)
    avg_evals = np.mean(fevals)
    print(f"SA {problem_name}: {avg_fit}: {avg_time}: {avg_evals}")
    return avg_fit, avg_time, avg_evals


def run_ANN():
    with open("data/wine.csv", 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    random_state = 0
    net = NeuralNetwork(hidden_nodes=[290], max_iters=110, algorithm="gradient_descent", curve=True,
                        learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True,
                        early_stopping=True)

    # Normalize feature data
    scaler = MinMaxScaler()

    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # this shouldn't be necessary, but following the example and breaks otherwise
    # One hot encode target values
    one_hot = OneHotEncoder()

    label_train_hot = one_hot.fit_transform(label_train.reshape(-1, 1)).todense()
    label_test_hot = one_hot.transform(label_test.reshape(-1, 1)).todense()

    start = time.time()

    net.fit(data_train_scaled, label_train_hot)

    fit_time = time.time() - start

    print(f"Gradient Descent: {fit_time}")

    # plot fitness curve

    plt.plot(net.fitness_curve, label="gradient descent")
    # plt.show()


def run_ANN_RHC():
    with open("data/wine.csv", 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    random_state = 0
    net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="random_hill_climb", curve=True,
                        learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True,
                        restarts=5, max_iters=110)

    # Normalize feature data
    scaler = MinMaxScaler()

    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # this shouldn't be necessary, but following the example and breaks otherwise
    # One hot encode target values
    one_hot = OneHotEncoder()

    label_train_hot = one_hot.fit_transform(label_train.reshape(-1, 1)).todense()
    label_test_hot = one_hot.transform(label_test.reshape(-1, 1)).todense()

    start = time.time()
    net.fit(data_train_scaled, label_train_hot)
    fit_time = time.time() - start

    print(f"RHC: {fit_time}")

    # train_vals = []
    #
    # plt.figure()
    # plt.title("rhc")
    # plt.xlabel("allowed restarts")
    # plt.ylabel("score")
    #
    # # create MC curve over HP for tuning
    # for restarts in range(0,100,5):
    #     vals = []
    #     for random_state in random_states:
    #         net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="random_hill_climb", restarts=restarts,
    #                             curve=True, learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True)
    #         net.fit(data_train_scaled, label_train_hot)
    #         label_pred = net.predict(data_train_scaled)
    #         train_accuracy = accuracy_score(label_train_hot, label_pred)
    #         vals.append(train_accuracy)
    #     avg = np.mean(vals)
    #     print(f"rhc {restarts}: {avg}")
    #     train_vals.append(avg)
    #
    # plt.plot(range(0,100,5), train_vals)
    # plt.savefig("charts/ann_rhc_restarts")


    # plot fitness curve
    # plt.figure()
    # plt.title("rhc")
    # plt.xlabel("iterations")
    # plt.ylabel("fitness")
    plt.plot(net.fitness_curve, label="rhc")
    # plt.show()


def run_ANN_SA():
    with open("data/wine.csv", 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    random_state = 0
    net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="simulated_annealing", curve=True,
                        learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True,
                        schedule=GeomDecay(init_temp=1, decay=0.99), max_iters=110)

    # Normalize feature data
    scaler = MinMaxScaler()

    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # this shouldn't be necessary, but following the example and breaks otherwise
    # One hot encode target values
    one_hot = OneHotEncoder()

    label_train_hot = one_hot.fit_transform(label_train.reshape(-1, 1)).todense()
    label_test_hot = one_hot.transform(label_test.reshape(-1, 1)).todense()

    start = time.time()
    net.fit(data_train_scaled, label_train_hot)
    fit_time = time.time() -start

    print(f"SA: {fit_time}")

    # train_vals = []
    #
    # plt.figure()
    # plt.title("sa")
    # plt.xlabel("temp")
    # plt.ylabel("score")
    #
    #
    # # create MC curve over HP for tuning
    # for temp in range(1,20):
    #     vals = []
    #     for random_state in random_states:
    #         net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="simulated_annealing",
    #                             schedule=GeomDecay(init_temp=temp),
    #                             curve=True, learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True)
    #         net.fit(data_train_scaled, label_train_hot)
    #         label_pred = net.predict(data_train_scaled)
    #         train_accuracy = accuracy_score(label_train_hot, label_pred)
    #         vals.append(train_accuracy)
    #     avg = np.mean(vals)
    #     print(f"sa {temp}: {avg}")
    #     train_vals.append(avg)
    #
    #
    # plt.plot(range(1,20), train_vals, label="temp")
    # plt.savefig("charts/ann_sa_temp")

    # train_vals = []
    # plt.figure()
    # plt.title("sa")
    # plt.ylabel("score")
    # plt.xlabel("decay")
    #
    # # create MC curve over HP for tuning
    # for decay in np.linspace(0.01, 1, 21):
    #     vals = []
    #     for random_state in random_states:
    #         net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="simulated_annealing",
    #                             schedule=GeomDecay(decay=decay),
    #                             curve=True, learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True)
    #         net.fit(data_train_scaled, label_train_hot)
    #         label_pred = net.predict(data_train_scaled)
    #         train_accuracy = accuracy_score(label_train_hot, label_pred)
    #         vals.append(train_accuracy)
    #     avg = np.mean(vals)
    #     print(f"sa {decay}: {avg}")
    #     train_vals.append(avg)
    #
    # plt.plot(np.linspace(0.01, 1, 21), train_vals, label="decay")
    # plt.savefig("charts/ann_sa_decay")

    # plot fitness curve
    # plt.figure()
    # plt.title("SA")
    # plt.xlabel("iterations")
    # plt.ylabel("fitness")
    plt.plot(net.fitness_curve, label="sa")
    # plt.show()


def run_ANN_GA():
    with open("data/wine.csv", 'r') as f:
        data = np.genfromtxt(f, delimiter=',')

    data, labels = data[:, :-1], data[:, -1]

    # split for training and testing
    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.4, random_state=0, stratify=labels
    )

    random_state = 0
    net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="genetic_alg", curve=True,
                        learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True,
                        pop_size=160, max_iters=110)

    # Normalize feature data
    scaler = MinMaxScaler()

    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # this shouldn't be necessary, but following the example and breaks otherwise
    # One hot encode target values
    one_hot = OneHotEncoder()

    label_train_hot = one_hot.fit_transform(label_train.reshape(-1, 1)).todense()
    label_test_hot = one_hot.transform(label_test.reshape(-1, 1)).todense()

    start = time.time()
    net.fit(data_train_scaled, label_train_hot)
    fit_time = time.time()-start

    print(f"GA: {fit_time}")

    # train_vals = []
    #
    # plt.figure()
    # plt.title("ga")
    # plt.xlabel("pop size")
    # plt.ylabel("score")
    #
    #
    # # create MC curve over HP for tuning
    # for temp in range(100, 310,20):
    #     vals = []
    #     for random_state in random_states:
    #         net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="genetic_alg",
    #                             pop_size=temp,
    #                             curve=True, learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True)
    #         net.fit(data_train_scaled, label_train_hot)
    #         label_pred = net.predict(data_train_scaled)
    #         train_accuracy = accuracy_score(label_train_hot, label_pred)
    #         vals.append(train_accuracy)
    #     avg = np.mean(vals)
    #     print(f"ga {temp}: {avg}")
    #     train_vals.append(avg)
    #
    # plt.plot(range(100, 310,20), train_vals, label="pop_size")
    # plt.savefig("charts/ann_ga_pop_size")
    #
    # train_vals = []
    # plt.figure()
    # plt.title("ga")
    # plt.ylabel("score")
    # plt.xlabel("mutation_prob")
    #
    #
    # # create MC curve over HP for tuning
    # for decay in np.linspace(0.01, 1, 21):
    #     vals = []
    #     for random_state in random_states:
    #         net = NeuralNetwork(hidden_nodes=[290], early_stopping=True, algorithm="genetic_alg",
    #                             mutation_prob=decay,
    #                             curve=True, learning_rate=0.001, random_state=random_state, is_classifier=True, bias=True)
    #         net.fit(data_train_scaled, label_train_hot)
    #         label_pred = net.predict(data_train_scaled)
    #         train_accuracy = accuracy_score(label_train_hot, label_pred)
    #         vals.append(train_accuracy)
    #     avg = np.mean(vals)
    #     print(f"ga {decay}: {avg}")
    #     train_vals.append(avg)
    #
    # plt.plot(np.linspace(0.01, 1, 21), train_vals, label="mutation_prob")
    # plt.savefig("charts/ann_ga_mutation_prob")

    # plot fitness curve
    # plt.figure()
    # plt.title("GA")
    # plt.xlabel("iterations")
    # plt.ylabel("fitness")
    plt.plot(net.fitness_curve, label="ga")
    # plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    # problem 1
    init_states = [
        np.random.randint(0, 2, 20),
        np.random.randint(0, 2, 50),
        np.random.randint(0, 2, 100)
    ]

    # plot fevals and fitness over problem sizes
    lens = [len(init_state) for init_state in init_states]
    RHC_vals = []
    SA_vals = []
    GA_vals = []
    MIMIC_vals = []
    RHC_fevals = []
    SA_fevals = []
    GA_fevals = []
    MIMIC_fevals = []
    fit_func=MaxKColor([])
    problem_name = str(fit_func).split('.')[-1].split(' ')[0]
    plt.figure()
    plt.title(problem_name)
    plt.xlabel("problem size")
    plt.ylabel("fitness")
    for init_state in init_states:
        problem = MaxKColorGenerator().generate(seed=123, number_of_nodes=len(init_state),
                                                max_connections_per_node=4, max_colors=None)
        fit,_,evals = run_RHC_1(problem, init_state)
        RHC_vals.append(fit)
        RHC_fevals.append(evals)

        fit,_,evals = run_SA_1(problem, init_state)
        SA_vals.append(fit)
        SA_fevals.append(evals)

        fit,_,evals = run_GA_1(problem, init_state)
        GA_vals.append(fit)
        GA_fevals.append(evals)

        fit,_,evals = run_MIMIC_1(problem, init_state)
        MIMIC_vals.append(fit)
        MIMIC_fevals.append(evals)
    plt.plot(lens, RHC_vals, label="rhc")
    plt.plot(lens, SA_vals, label="sa")
    plt.plot(lens, GA_vals, label="ga")
    plt.plot(lens, MIMIC_vals, label="mimic")

    problem_name = str(fit_func).split('.')[-1].split(' ')[0]
    plt.title(problem_name)
    plt.xlabel("problem size")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(f"charts/{problem_name}")

    # chart fevals
    plt.figure()
    plt.title(f"{problem_name} fevals")
    plt.xlabel("problem size")
    plt.ylabel("fevals")
    plt.plot(lens, RHC_fevals, label="rhc")
    plt.plot(lens, SA_fevals, label="sa")
    plt.plot(lens, GA_fevals, label="ga")
    plt.plot(lens, MIMIC_fevals, label="mimic")
    plt.legend()
    plt.savefig(f"charts/{problem_name}")

    # problem 2
    init_state = init_states = [
        np.random.randint(0, 2, 10),
        np.random.randint(0, 2, 30),
        np.random.randint(0, 2, 50)
    ]
    # plt.figure()
    fit_func = SixPeaks()

    # plot time over problem sizes
    lens = [len(init_state) for init_state in init_states]
    RHC_vals = []
    SA_vals = []
    GA_vals = []
    MIMIC_vals = []
    RHC_times = []
    SA_times = []
    GA_times = []
    MIMIC_times = []
    for init_state in init_states:
        problem = DiscreteOpt(length=len(init_state), fitness_fn=fit_func)
        fit, duration,_ = run_RHC_2(problem, init_state)
        RHC_vals.append(fit)
        RHC_times.append(duration)
        fit, duration,_ = run_SA_2(problem, init_state)
        SA_vals.append(fit)
        SA_times.append(duration)
        fit, duration,_ = run_GA_2(problem, init_state)
        GA_vals.append(fit)
        GA_times.append(duration)
        fit, duration,_ = run_MIMIC_2(problem, init_state)
        MIMIC_vals.append(fit)
        MIMIC_times.append(duration)
    plt.plot(lens, RHC_vals, label="rhc")
    plt.plot(lens, SA_vals, label="sa")
    plt.plot(lens, GA_vals, label="ga")
    plt.plot(lens, MIMIC_vals, label="mimic")

    print()
    problem_name = str(fit_func).split('.')[-1].split(' ')[0]
    plt.title(problem_name)
    plt.xlabel("problem size")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(f"charts/{problem_name}")

    plt.figure()
    plt.title(f"{problem_name} times")
    plt.xlabel("problem size")
    plt.ylabel("time (seconds)")
    plt.plot(lens, RHC_times, label="rhc")
    plt.plot(lens, SA_times, label="sa")
    plt.plot(lens, GA_times, label="ga")
    plt.plot(lens, MIMIC_times, label="mimic")
    plt.legend()
    plt.savefig(f"charts/{problem_name}_time")

    # problem 3

    init_states = [
        np.random.randint(0, 8, 8),
        np.random.randint(0, 36, 36),
        np.random.randint(0, 50, 50)
    ]
    # plt.figure()
    fit_func = Queens()

    # plot fitness over problem sizes
    lens = [len(init_state) for init_state in init_states]
    RHC_vals = []
    SA_vals = []
    GA_vals = []
    MIMIC_vals = []
    for init_state in init_states:
        problem = DiscreteOpt(length=len(init_state), fitness_fn=fit_func, maximize=False)
        fit, _,_ = run_RHC_3(problem, init_state)
        RHC_vals.append(fit)
        fit, _,_ = run_SA_3(problem, init_state)
        SA_vals.append(fit)
        fit, _,_ = run_GA_3(problem, init_state)
        GA_vals.append(fit)
        fit, _,_ = run_MIMIC_3(problem, init_state)
        MIMIC_vals.append(fit)
    plt.plot(lens, RHC_vals, label="rhc")
    plt.plot(lens, SA_vals, label="sa")
    plt.plot(lens, GA_vals, label="ga")
    plt.plot(lens, MIMIC_vals, label="mimic")

    print()
    problem_name = str(fit_func).split('.')[-1].split(' ')[0]
    plt.title(problem_name)
    plt.xlabel("problem size")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(f"charts/{problem_name}")

    # ANN compare
    plt.figure()
    plt.title("ANN Comparison")
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    run_ANN()
    run_ANN_RHC()
    run_ANN_SA()
    run_ANN_GA()
    plt.legend()
    plt.savefig("charts/ann_compare")
    plt.close('all')
    print()
