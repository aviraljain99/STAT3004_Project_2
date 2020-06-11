# STAT3004 Project 2

A project involving simulating epidemics and determining the behaviour of different models for epidemics. The notation used is:

* S - Susceptibles
* I - Infectives
* R - Removed/Recovered 

The main models investigated were:

* SI - A simple model where once infected, an individual remains infected
* SIS - Once infected, an individual can again become susceptible, i.e. an individual can transition between the infected and susceptible states
* SIR - A more realistic model where after becoming infected, an individual transitions to a Recovered/Removed state. After recovering, an individual is assumed to have become immune to the infection and cannot become susceptible again
* SI(SR) - In comparison to the previous model, once an individual has recovered, they do not become susceptible again. This assumption is removed and once recovered, an individual may lose immunity and become a susceptible once again.

These were modelled using Continuous Time Markov Chains (CTMCs). Simulations were done using Python and Jupyter Notebooks. A copy of the task-sheet for this project can be found at this [link](https://courses.smp.uq.edu.au/STAT3004/Project2.pdf)
