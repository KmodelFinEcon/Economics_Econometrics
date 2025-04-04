
#(Macro-Economics)Implementation of the ROMER endogenous growth model by K.Tomov

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt

#gobal parameters 

CAPITAL_SHARE = 1/3                # Capital's share in production (Cobb-Douglas) [a]
LABOR_SHARE = 1 - CAPITAL_SHARE     # Labor's share 
TECH_GROWTH = 0.013                #Long-term technology growth rate
POP_GROWTH = 0.0014                  # Population growth rate
CAPITAL_DEPRECIATION = 0.025     # Annual capital depreciation rate
TIME_PREFERENCE = 0.015          # Real interest rate from rfr
TECH_SHOCK_PERSIST = 0.75        # Technology shock persistence
GOV_SHOCK_PERSIST = 0.65         # Government spending shock persistence
STEADY_STATE_EMPLOYMENT = 1/3    # Employment rate
GOV_SPENDING_SHARE = 0.25        # Government spending share of output
INIT_TECH_LEVEL = 1              # Initial technology level 
INIT_LABOR_SUPPLY = 0            # Initial labor supply 

#objective function 

EFFECTIVE_DISCOUNT = np.log(1 + TIME_PREFERENCE) - TECH_GROWTH
OUTPUT_CAP_RATIO = (TIME_PREFERENCE + CAPITAL_DEPRECIATION) / CAPITAL_SHARE
STEADY_STATE_CAPITAL = OUTPUT_CAP_RATIO ** (1 / (CAPITAL_SHARE - 1))
STEADY_OUTPUT = OUTPUT_CAP_RATIO * STEADY_STATE_CAPITAL
STEADY_WAGE = LABOR_SHARE * STEADY_OUTPUT
GOV_STEADY_SPENDING = GOV_SPENDING_SHARE * STEADY_OUTPUT
GOV_NORM_FACTOR = np.log(GOV_STEADY_SPENDING * STEADY_STATE_EMPLOYMENT) + INIT_TECH_LEVEL + INIT_LABOR_SUPPLY

# Consumption and Investment Ratios
consumption_steady = ((1 - CAPITAL_DEPRECIATION - np.exp(TECH_GROWTH + POP_GROWTH)) * STEADY_STATE_CAPITAL 
                      + STEADY_OUTPUT - GOV_STEADY_SPENDING)
CONSUMPTION_CAP_RATIO = consumption_steady / STEADY_STATE_CAPITAL
LABOR_LEISURE_RATIO = (1 - STEADY_STATE_EMPLOYMENT) / STEADY_STATE_EMPLOYMENT * STEADY_WAGE / consumption_steady

#computation of main model and calculation of transition matrix

def compute_b_coefficients(coeffs):
    beta_ck, beta_ca, beta_cg, beta_lk, beta_la, beta_lg = coeffs
    b_kk = (1 - CAPITAL_DEPRECIATION + OUTPUT_CAP_RATIO * (CAPITAL_SHARE + LABOR_SHARE * beta_lk) - CONSUMPTION_CAP_RATIO * beta_ck)
    b_ka = OUTPUT_CAP_RATIO * LABOR_SHARE * (1 + beta_la) - CONSUMPTION_CAP_RATIO * beta_ca
    b_kg = (OUTPUT_CAP_RATIO * LABOR_SHARE * beta_lg - CONSUMPTION_CAP_RATIO * beta_cg - (GOV_STEADY_SPENDING / STEADY_STATE_CAPITAL))
    growth_adj = np.exp(TECH_GROWTH + POP_GROWTH)# Growth adjustment
    
    return np.array([b_kk / growth_adj, b_ka / growth_adj, b_kg / growth_adj])

#defining the equilibrium conditions

def model_equations(coeffs): 
    beta = compute_b_coefficients(coeffs)
    beta_ck, beta_ca, beta_cg, beta_lk, beta_la, beta_lg = coeffs
    labor_elasticity = STEADY_STATE_EMPLOYMENT / (1 - STEADY_STATE_EMPLOYMENT) + CAPITAL_SHARE
    arbitrage_adj = LABOR_SHARE * (TIME_PREFERENCE + CAPITAL_DEPRECIATION) / (TIME_PREFERENCE + 1)

    return [

        beta_ck + labor_elasticity * beta_lk - CAPITAL_SHARE,
        beta_ca + labor_elasticity * beta_la - LABOR_SHARE,
        beta_cg + labor_elasticity * beta_lg,
        beta_ck - (beta_ck * beta[0] - arbitrage_adj * ((beta_lk - 1) * beta[0])),# Euler residuals
        beta_ca - (beta_ck * beta[1] + beta_ca * TECH_SHOCK_PERSIST
                   - arbitrage_adj * ((beta_lk - 1) * beta[1] + (1 + beta_la) * TECH_SHOCK_PERSIST)),
        beta_cg - (beta_ck * beta[2] + beta_cg * GOV_SHOCK_PERSIST
                   - arbitrage_adj * ((beta_lk - 1) * beta[2] + beta_lg * GOV_SHOCK_PERSIST))
    ]

#Economic state in a single period

def compute_period(state, beta, shock=None):
    k, a, g = state
    shock = shock if shock is not None else [0, 0]
    beta_ck, beta_ca, beta_cg, beta_lk, beta_la, beta_lg = beta
    b = compute_b_coefficients(beta)
    labor_supply = beta_lk * k + beta_la * a + beta_lg * g
    output = CAPITAL_SHARE * k + LABOR_SHARE * (a + labor_supply)
    consumption = beta_ck * k + beta_ca * a + beta_cg * g
    rfr = (1 + (TIME_PREFERENCE + CAPITAL_DEPRECIATION) * (output - k))**4 - 1  # on annual basis
    wage_rate = CAPITAL_SHARE * k + LABOR_SHARE * a - CAPITAL_SHARE * labor_supply
    k_next = b[0] * k + b[1] * a + b[2] * g
    a_next = TECH_SHOCK_PERSIST * a + shock[0]
    g_next = GOV_SHOCK_PERSIST * g + shock[1]

    flows = np.array([consumption, labor_supply, output, rfr, wage_rate])
    next_state = np.array([k_next, a_next, g_next])
    return flows, next_state

#Main simulation

def run_simulation(shock_size=1, periods=60):
    beta_solution = fsolve(model_equations, x0=[0.6, 0.3, 0.2, 0, 1, 0])
    economic_flows = np.zeros((periods, 5))# Initialize storage: economic flows (consumption, labor, output, interest rate, wage)
    capital_stocks = np.zeros((periods, 3))# Capital stocks: k, a, g
    economic_flows[0], capital_stocks[1] = compute_period(capital_stocks[0], beta=beta_solution, shock=[shock_size, 0])

    for t in range(1, periods - 1):
        economic_flows[t], capital_stocks[t + 1] = compute_period(capital_stocks[t], beta=beta_solution)

    return economic_flows, capital_stocks

#Execution

if __name__ == "__main__":
    flows, stocks = run_simulation()
    results = pd.DataFrame(np.c_[flows, stocks],columns=["consumption", "labor", "output", "rfr", "wage","capital", "technology", "gov_spending"])

    fig, ax = plt.subplots(3, 1, figsize=(10, 13))
    
    results.plot(y=["capital", "technology", "labor"], ax=ax[0],
                 title="Production Factors Dynamics")
    ax[0].axhline(0, ls='--', c='k', lw=0.3)

    results.plot(y=["consumption", "output"], ax=ax[1],
                 title="Output and Consumption Response")
    ax[1].axhline(0, ls='--', c='k', lw=0.3)

    results.plot(y=["wage", "rfr"], ax=ax[2],
                 title="Wage and Interest Rate Dynamics")
    ax[2].axhline(0, ls='--', c='k', lw=0.3)

    plt.tight_layout()
    plt.show()