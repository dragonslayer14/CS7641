from hiive.mdptoolbox import example, mdp

P, R = example.forest()
vi = mdp.ValueIteration(P, R, 0.96)
print(vi.verbose)

vi.run()
expected = (5.93215488, 9.38815488, 13.38815488)
all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))

print(vi.policy)

print(vi.iter)

from hiive import mdptoolbox
import numpy as np
P = np.array([[[0.5, 0.5], [0.8, 0.2]], [[0, 1], [0.1, 0.9]]])
R = np.array([[5, 10], [-1, 2]])
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()
expected = (40.048625392716815, 33.65371175967546)
all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))

print(vi.policy)

print(vi.iter)

from hiive import mdptoolbox
import numpy as np
from scipy.sparse import csr_matrix as sparse
P = [None] * 2
P[0] = sparse([[0.5, 0.5], [0.8, 0.2]])
P[1] = sparse([[0, 1], [0.1, 0.9]])
R = np.array([[5, 10], [-1, 2]])
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
vi.run()
expected = (40.048625392716815, 33.65371175967546)
all(expected[k] - vi.V[k] < 1e-12 for k in range(len(expected)))

print(vi.policy)
