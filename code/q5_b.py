import numpy as np
import matplotlib.pyplot as plt
from itertools import product as product
from scipy.linalg import expm

# Parameters
N = 100
I_0 = 5
Rsi = 2/N
Rir = 1/N

allStates = [(x, y) for x in range(N + 1) for y in range(N + 1) if x + y <= N]
states = {
    (x, y) : ind for ind, (x, y) in enumerate(allStates, 0)
    }
biStates = {ind : state for state, ind in states.items()}

# Generate Q matrix
Q = np.zeros((len(states), len(states)))

for (x, y), ind in states.items():
    if y > 0:
        transitionTo = states[(x, y - 1)]
        Q[ind, transitionTo] = Rir * y
    if x > 0:
        transitionTo = states[(x - 1, y + 1)]
        Q[ind, transitionTo] = Rsi * y * x
    Q[ind, ind] = - np.sum(Q[ind, :])

# Probability transition matrices
Ps = [expm(Q * t) for t in [5, 10, 20, 50]]

# Checking validity
for P in Ps:
    assert np.allclose(np.sum(P, axis=1), np.ones((P.shape[0],)))

# estimate expected value of I_t
Is = []
for P in Ps:
    assert np.allclose(np.sum(P, axis=1), np.ones((P.shape[0],)))
    start = states[(N - I_0, I_0)]
    Is.append(sum(biStates[col][1] * P[start, col] for col in range(P.shape[1])))

print(Is)