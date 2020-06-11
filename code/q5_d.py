import numpy as np
import matplotlib.pyplot as plt
from itertools import product as product
from scipy.linalg import expm

def scoobydoo(currState, Pjump, biStates):
    """ Does a single simulation of the CTMC and returns the final state """
    state = currState
    while biStates[state][1] != 0:
        state = np.random.choice(Pjump.shape[1], p=Pjump[state, :])
    return biStates[state]

# Parameters
N = 100
Rirs = [1/N, 10/N, 30/N, 50/N, 75/N, 100/N, 150/N, 2]
I_0 = 5
Rsi = 2/N

# Matrix to record how many susceptibles were remaining 
# for each R_ir value
distribution = np.zeros((len(Rirs), N - I_0 + 1))

allStates = [(x, y) for x in range(N + 1) for y in range(N + 1) if x + y <= N]
states = {(x, y) : ind for ind, (x, y) in enumerate(allStates, 0)}
biStates = {ind : state for state, ind in states.items()}

# Perform simulation for R_ir values
for R_ind, Rir in enumerate(Rirs, 0):    
    Q = np.zeros((len(states), len(states)))

    for (x, y), ind in states.items():
        if y > 0:
            transitionTo = states[(x, y - 1)]
            Q[ind, transitionTo] = Rir * y
        if x > 0:
            transitionTo = states[(x - 1, y + 1)]
            Q[ind, transitionTo] = Rsi * y * x
        Q[ind, ind] = - np.sum(Q[ind, :])
    
    Pjump = Q / (- np.diagonal(Q))[:, np.newaxis]
    np.fill_diagonal(Pjump, 0)
    np.nan_to_num(Pjump, copy=False)
    
    trials = int(5e4)
    infected = [0 for i in range(N - I_0 + 1)]

    for trial in range(trials):
        endState = scoobydoo(states[(N - I_0, I_0)], Pjump, biStates)
        suscepInfected = (N - I_0) - endState[0]
        infected[suscepInfected] += 1
    
    distribution[R_ind, :] = np.asarray([i/trials for i in infected])

 # Plotting the distributions
fig, ax = plt.subplots(nrows=len(Rirs), sharex=True, figsize=(7,10))

for i, Rir in enumerate(Rirs, 0):
    ax[i].bar(range(len(infected)), distribution[i, :], label="Rir = {}".format(Rirs[i]))
    if x == 4:
        ax[i].set_xlabel("Number infected")
    ax[i].set_ylabel("Probability")
    ax[i].set_xlim((0, 100))
    ax[i].legend()

plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.1)
plt.savefig("images//q5_d")