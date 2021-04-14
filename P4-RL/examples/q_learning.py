# These examples are reproducible only if random seed is set to 0 in
# both the random and numpy.random modules.
import numpy as np
import mdptoolbox, mdptoolbox.example
np.random.seed(0)
P, R = mdptoolbox.example.forest()
ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
ql.run()
print(ql.Q)
expected = (11.198908998901134, 11.741057920409865, 12.259732864170232)
all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
print(ql.policy)

import mdptoolbox
import numpy as np
P = np.array([[[0.5, 0.5],[0.8, 0.2]],[[0, 1],[0.1, 0.9]]])
R = np.array([[5, 10], [-1, 2]])
np.random.seed(0)
ql = mdptoolbox.mdp.QLearning(P, R, 0.9)
ql.run()
print(ql.Q)
expected = (40.82109564847122, 34.37431040682546)
all(expected[k] - ql.V[k] < 1e-12 for k in range(len(expected)))
print(ql.policy)