import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Parameters
N = 5 # total population
Rsi = 1

I_0 = 1
S_0 = 4

# Q matrix
Q = np.zeros((N + 1, N + 1))
for i in range(1, N): # number of infectives
    Q[i, i - 1] = Rsi * i * (N - i)
    Q[i, i] = - (Rsi * i * (N - i))


def doobGillespie(diagonal, currState):
	"""Performs a single simulation of the CTMC"""
    times = [0]
    state = currState
    # print(- np.diag(Q)[state])
    timeSpent = np.random.exponential(scale=1/diagonal[state])
    state -= 1
    times.append(timeSpent)
    # print(state)
    while state != 0 and (state != Q.shape[0] - 1):
        # print(- np.diag(Q)[state])
        timeSpent = np.random.exponential(scale=1/(diagonal[state]))
        state -= 1
        times.append((times[-1] + timeSpent))
        # print(state)
    return times


def binSearch(searchIn, goal):
	"""Returns the lower and upper index between which the 
	element that is being searched for is"""
    if goal >= searchIn[-1]:
        return (len(searchIn) - 1, None)
    elif goal <= searchIn[0]:
        return (None, 0)
    else:
        low = 0
        upper = len(searchIn) - 1
        middle = int((upper + low)/2)
        while low < upper: # replace this
            # print("searching between " + str(low) + " and " + str(upper))
            if middle < len(searchIn) - 1:
                if searchIn[middle] <= goal and searchIn[middle + 1] >= goal:
                    return (middle, middle + 1)
            elif middle > 0:
                if searchIn[middle - 1] >= goal and searchIn[middle] <= goal:
                    return (middle, middle - 1)
            
            if goal > searchIn[middle]:
                low = middle + 1
            else: # goal < middle
                upper = middle - 1
            middle = int((upper + low)/2)
        return (middle, middle + 1)


# Range of time
ts = np.linspace(0.001, 2, num=20)

t_values = np.asarray([[0 for i in range(S_0 + 1)] for t in ts])
trials = 1000000
timings = np.zeros((trials, S_0 + 1))

diagonal = - np.diag(Q)

for trial in range(trials):
    timings[trial, :] = doobGillespie(diagonal, S_0)

for i, t in enumerate(ts, 0):
    for row in range(timings.shape[0]):
        low, upper = binSearch(timings[row, :], t)
        # So at time t, the number of susceptible is between index low and upper
        # 0 -> 4
        # 1 -> 3
        # 2 -> 2
        # 3 -> 1
        # 4 -> 0
        # if upper != None
        if low == None:
            print(timings[row, :]) # This should never happen
        else:
            t_values[i, 4 - low] += 1

# Divide by the sum of each row
probs = t_values / np.sum(t_values, axis=1)[:, np.newaxis]

# Calculating the formulaic versions
four = lambda t : np.exp(-4 * t)
six = lambda t : np.exp(-6 * t) 
form = lambda t : [four(t), 
                    2 * (four(t) - six(t)), 
                    6 * (four(t) - six(t) - (2 * t * six(t))),
                    36 * ((t * four(t)) - four(t) + six(t) + (t * six(t))),
                    1 + (27 * four(t)) - (36 * t * four(t)) - (28 * six(t)) - (24 * t * six(t))
                   ]

# Calculating the probability transition matrix at each time t
Ps = [expm(t * Q) for t in ts]

plt.figure(figsize=(7,4))

maxDiff = []

for index, (t, P) in enumerate(zip(ts, Ps), 0):
    monte = probs[index, :]
    calc = list(reversed(form(t)))
    maxDiff.append(max([abs(a - b) for a, b in zip(monte, calc)]))

# Plotting the absolute maximum difference
plt.scatter(ts, maxDiff)
plt.ylim((-0.005, 0.005))
plt.title("Max difference between the formula and Monte Carlo")
plt.ylabel("Difference in probability")
plt.xlabel("Times")
plt.savefig("q3_b")