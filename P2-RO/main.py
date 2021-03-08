from mlrose import NeuralNetwork, random_hill_climb, simulated_annealing, genetic_alg, mimic


def run_RHC(problem):
    # TODO use multiple random states and average values
    random_state=0
    best_state, best_fit, fit_vals = random_hill_climb(problem, random_state=random_state, curve=True)


def run_GA(problem):
    # TODO use multiple random states and average values
    random_state=0
    best_state, best_fit, fit_vals = genetic_alg(problem, curve=True, random_state=random_state)


def run_MIMIC(problem):
    # TODO use multiple random states and average values
    random_state=0
    best_state, best_fit, fit_vals = mimic(problem, curve=True, random_state=random_state)


def run_SA(problem):
    # TODO use multiple random states and average values
    random_state=0
    best_state, best_fit, fit_vals = simulated_annealing(problem, curve=True, random_state=random_state)


def run_ANN(algo = "gradient_descent"):
    # TODO use multiple random states and average values
    random_state = 0
    _, loss, _, fit_curve = NeuralNetwork(algorithm=algo, curve=True, random_state=random_state)


if __name__ == "__main__":
    problems = []
    for problem in problems:
        run_RHC(problem)
        run_GA(problem)
        run_SA(problem)
        run_MIMIC(problem)

    # ANN compare
    run_ANN()
    run_ANN("random_hill_climb")
    run_ANN("simulated_annealing")
    run_ANN("genetic_alg")
