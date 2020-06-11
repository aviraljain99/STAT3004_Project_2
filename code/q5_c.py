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

# Create Q-matrix
Q = np.zeros((len(states), len(states)))

for (x, y), ind in states.items():
    if y > 0:
        transitionTo = states[(x, y - 1)]
        Q[ind, transitionTo] = Rir * y
    if x > 0:
        transitionTo = states[(x - 1, y + 1)]
        Q[ind, transitionTo] = Rsi * y * x
    Q[ind, ind] = - np.sum(Q[ind, :])

def doobGillespie(diagonal, currState, Pjump, biStates):
    """ Performs one simulation of the CTMC """
    times = [(0, biStates[currState])]
    state = currState
    while biStates[state][1] != 0: # While the number of infectives isn't zero
        timeSpent = np.random.exponential(scale=1/(diagonal[state]))
        state = np.random.choice(range(Pjump.shape[1]), p=Pjump[state, :])
        times.append((times[-1][0] + timeSpent, biStates[state]))
    return times

def binSearch(searchIn, goal):
    """ Performs binary search to search for a goal """
    if goal >= searchIn[-1][0]:
        return (len(searchIn) - 1, None)
    elif goal <= searchIn[0][0]:
        return (None, 0)
    else:
        low = 0
        upper = len(searchIn) - 1
        middle = int((upper + low)/2)
        while low < upper: # replace this
            # print("searching between " + str(low) + " and " + str(upper))
            if middle < len(searchIn) - 1:
                if searchIn[middle][0] <= goal and searchIn[middle + 1][0] >= goal:
                    return (middle, middle + 1)
            elif middle > 0:
                if searchIn[middle - 1][0] >= goal and searchIn[middle][0] <= goal:
                    return (middle, middle - 1)
            
            if goal > searchIn[middle][0]:
                low = middle + 1
            else: # goal < middle
                upper = middle - 1
            middle = int((upper + low)/2)
        return (middle, middle + 1)

# Times
ts = [5, 10, 20, 50]

t_values = np.asarray([0 for t in ts])
trials = 10000

diagonal = - np.diag(Q)

for trial in range(trials):
    trace = doobGillespie(diagonal, states[(N - I_0, I_0)], Pjump, biStates)
    for ind_t, t in enumerate(ts, 0):
        lower, _ = binSearch(trace, t)
        t_values[ind_t] += trace[lower][1][1]

# Expected values
print([val/trials for val in t_values])