import numpy as np
import matplotlib.pyplot as plt
from numpy.random import exponential as exp

# Parameters
I_0 = 5
S_0 = 5
N = 10
Ris = 2

def getQMatrix(Rsi):
    """ Create Q matrix for SI model with fixed Rsi value"""
    Q = np.zeros((N + 1, N + 1))
    for i in range(N):
        Q[i, i + 1] = Ris * (N - i) # infective becomes a susceptible again
        if i != 0:
            Q[i, i - 1] = Rsi * i * (N - i) # susceptible becomes an infective
        Q[i, i] = - np.sum(Q[i, :])
    return Q

# Run simulation for a range of Rsi values initially
Rs = np.linspace(0, 1, num=40)
probs = []
trials = int(5000)

for Rsi in Rs:
    # Get Q matrix
    Q = getQMatrix(Rsi)
    diagonal = (- np.diag(Q))
    jumpUp = [Q[i, i + 1]/(-Q[i, i]) for i in range(N)]    
    # Get probability matrix
    Pjump = Q / diagonal[:, np.newaxis]
    np.fill_diagonal(Pjump, 0)
    np.nan_to_num(Pjump, copy=False)
    event = 0
    for trial in range(trials):
        time = 0
        state = S_0
        # print(state)
        while state != N:
            time += exp(scale=1/diagonal[state])
            val = np.random.rand()
            if val <= jumpUp[state]:
                state += 1
            else:
                state -= 1
        if time > 15:
            event += 1
    probs.append(event/trials)

# Create plot and observe which values give probability close to 0.9
plt.figure()
plt.scatter(Rs, probs, label="Probability")
plt.plot(Rs, np.ones(len(probs)) * 0.9, label="P = {}".format(0.9))
plt.xlabel("Rsi value")
plt.ylabel("Probability")
plt.legend()
plt.savefig("images\\q4_a")

# After finding that Rsi values close to 0.8 give probability close to 0.9
# choose Rsi values in the proximity of 0.8 and run a larger number of trials
trials = int(10000)

finer = []
finerRsi = [0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.832, 0.835, 0.84]

for Rsi in finerRsi:
    print(Rsi)
    # Get Q matrix
    Q = getQMatrix(Rsi)
    diagonal = (- np.diag(Q))
    jumpUp = [Q[i, i + 1]/(-Q[i, i]) for i in range(N)]    
    # Get probability matrix
    Pjump = Q / diagonal[:, np.newaxis]
    np.fill_diagonal(Pjump, 0)
    np.nan_to_num(Pjump, copy=False)
    
    event = 0
    for trial in range(trials):
        # print(trial)
        time = 0
        state = S_0
        # print(state)
        while state != N:
            time += exp(scale=1/diagonal[state])
            val = np.random.rand()
            if val <= jumpUp[state]:
                state += 1
            else:
                state -= 1
            # state = np.random.choice(range(0, N + 1), p=Pjump[state, :])
            # print(state)
        if time > 15:
            event += 1
    finer.append(event/trials)

# Plot probability values
plt.figure()
plt.scatter(finerRsi, finer, label="Probability")
plt.plot(finerRsi, np.ones(len(finerRsi)) * 0.9, label="P = {}".format(0.9))
plt.xlabel("Rsi value")
plt.ylabel("Probability")
plt.legend()
plt.savefig("images\\q4_a_ii")