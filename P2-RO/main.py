import mlrose

def run_RHC():
    pass


def run_GA():
    pass


def run_MIMIC():
    pass


def run_SA():
    pass


def run_ANN(algo = "backprop"):
    pass


if __name__ == "__main__":
    run_RHC()
    run_GA()
    run_SA()
    run_MIMIC()
    run_ANN()
    run_ANN("random_hill_climbing")
    run_ANN("genetic")
    run_ANN("mimic")
