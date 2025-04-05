#(Macro-Economics) Simple Implementation of the Schumpeterian endogenous model (ROMER 1990 modified) by K.Tomov

#Vertical innovation unlike ROMER which which is horizontal innovation shock with additional risk and destruction rate

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Model Parameters
CAPITAL_SHARE = 1/3             # alpha in production function
CAPITAL_DEPRECIATION = 0.025   # Annual capital depreciation rate
INNOVATION_EFFICIENCY = 0.05    # Effectiveness of R&D in improving technology
TIME_PREFERENCE = 0.010        # Discount rate
T_PERIODS = 100                 # Number of periods for the simulation
RISK_FACTOR = 0.2             # Determines how sensitive the risk is to R&D effort
DESTRUCTION_RATE = 0.4         # Fraction of capital lost if creative destruction occurs
INITIAL_TECH = 1.0            # Initial technology level [A]
INITIAL_CAPITAL = 10.0        # Initial capital stock [k]

#Objective function

def production(k, A):
    return A * (k ** CAPITAL_SHARE)

def expected_NP_output(E, k, A, Y):
    A_next = A * (1 + INNOVATION_EFFICIENCY * E)
    k_next = (1 - CAPITAL_DEPRECIATION) * k + (1 - E) * Y - (RISK_FACTOR * E * DESTRUCTION_RATE) * k
    k_next = max(k_next, 1e-8)# bounds to avoid negative capital
    return A_next * (k_next ** CAPITAL_SHARE)

#Global minima function for RnD effort

def optimize_RnD(k, A, Y):
    obj = lambda E: -expected_NP_output(E, k, A, Y)
    result = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    return result.x

#simulation of paths of economic variables of T_PERIODS

def run_simulation():   
    k_path = np.zeros(T_PERIODS) #capital stock
    A_path = np.zeros(T_PERIODS) #technology Level
    Y_path = np.zeros(T_PERIODS) #production output
    E_path = np.zeros(T_PERIODS)  # chosen R&D effort each period
    k_path[0] = INITIAL_CAPITAL #starting capital
    A_path[0] = INITIAL_TECH #starting capital stock

    for t in range(T_PERIODS - 1):
        Y = production(k_path[t], A_path[t])
        Y_path[t] = Y
        E_opt = optimize_RnD(k_path[t], A_path[t], Y)
        E_path[t] = E_opt #R&D effort E in [0, 1]
        A_path[t+1] = A_path[t] * (1 + INNOVATION_EFFICIENCY * E_opt)
        k_path[t+1] = ((1 - CAPITAL_DEPRECIATION) * k_path[t] + (1 - E_opt) * Y - (RISK_FACTOR * E_opt * DESTRUCTION_RATE) * k_path[t]) #risk of creative destruction
        
    Y_path[-1] = production(k_path[-1], A_path[-1])
    E_path[-1] = optimize_RnD(k_path[-1], A_path[-1], Y_path[-1])
    
    return k_path, A_path, Y_path, E_path

if __name__ == "__main__":
    k, A, Y, E = run_simulation()
    results = pd.DataFrame({ "C": k, "T": A, "O": Y, "RnD": E})
    
    fig, ax = plt.subplots(4, 1, figsize=(10, 13))
    results.plot(y="C", ax=ax[0], title="Capital Dynamics")
    results.plot(y="T", ax=ax[1], title="Technology Dynamics")
    results.plot(y="O", ax=ax[2], title="Output Dynamics")
    results.plot(y="RnD", ax=ax[3], title="R&D Effort Dynamics")
    
    plt.tight_layout()
    plt.show()
