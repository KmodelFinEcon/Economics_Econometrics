
###Bond duration incorporating Vasicek yield curve###
#by             Kaloi Tomov

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Bond Assumptions
ytm = 0.04  # starting YTM
years = np.arange(1, 6, 1) #year1to5
cf = np.array([7, 7, 7, 7, 107])  #CF including coupons + principal)

# Vasicek model parameters
a = 0.2  #mean reversion
b = ytm # LTM interest rate.
sigma = 0.01 # expected Volatility
r0 = ytm  # Initial interest rate
T = 5  #time in years
N = 2500  #time steps in simulation
dt = T / N  # size of timestep

# Simulate Vasicek interest rate paths
np.random.seed(42) 
t = np.linspace(0, T, N) 
r = np.zeros(N)
r[0] = r0

for i in range(1, N):
    dW = np.random.normal(0, np.sqrt(dt))
    dr = a * (b - r[i-1]) * dt + sigma * dW  # Vasicek SDE
    r[i] = r[i-1] + dr  #new IR

# interest rate simulated
def present_value_cash_flow(fv, rates, t):
    pvcf = fv * np.exp(-rates[t-1] * t)
    return pvcf

# PV of cashflows
data = pd.DataFrame({'CF': cf}, index=years)
data.index.name = 'Year'
data['PVCF'] = data.apply(lambda row: present_value_cash_flow(row['CF'], r, row.name), axis=1)
PVTCF = np.sum(data['PVCF'])

def macaulay(t, PVTCF, PVCF):
    return (t * PVCF) / PVTCF

data['Macaulay'] = data.apply(lambda row: macaulay(row.name, PVTCF, row['PVCF']), axis=1)

# Modified duration
def modified(macaulay, ytm):
    return macaulay / (1 + ytm)

data['Modified'] = data.apply(lambda row: modified(row['Macaulay'], ytm), axis=1)

macaulay_total = np.sum(data['Macaulay'])
modified_duration = macaulay_total / (1 + ytm)

# Output results
print("Simulated Interest Rates at Each Year:")
print(r[:5])  # Print first 5 simulated rates for illustration
print("\nBond Cash Flows and Durations:")
print(data)
print(f"\nTotal Macaulay Duration: {macaulay_total}")
print(f"Modified Duration: {modified_duration}")

# Simulated IR
plt.figure(figsize=(10, 6))
plt.plot(t, r, label="Simulated Interest Rate Path", color="blue")
plt.axhline(b, color="red", linestyle="--", label="Long-term Mean (b)")
plt.title("Vasicek Model Simulation for Bond Duration")
plt.xlabel("Time (Years)")
plt.ylabel("Interest Rate")
plt.legend()
plt.grid(True)
plt.show()