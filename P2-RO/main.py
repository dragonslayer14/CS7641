from mlrose import NeuralNetwork, random_hill_climb, simulated_annealing, genetic_alg, mimic


def run_RHC():
    random_hill_climb()
    pass


def run_GA():
    genetic_alg()
    pass


def run_MIMIC():
    mimic()
    pass


def run_SA():
    simulated_annealing()
    pass


def run_ANN(algo = "gradient_descent"):
    NeuralNetwork(algorithm=algo)
    pass


if __name__ == "__main__":
    run_RHC()
    run_GA()
    run_SA()
    run_MIMIC()
    run_ANN()
    run_ANN("random_hill_climb")
    run_ANN("simulated_annealing")
    run_ANN("genetic_alg")
