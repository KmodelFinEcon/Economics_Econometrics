# ODE implementation of the Semi-Endogenous Growth Model of Jones (1995) by K.Tomov from Dr. B.Moll 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#global starting parameters/ Timeframe

A0 = 1.0
K0 = 10.0
L0 = 10.0
u0 = [A0, K0, L0]
t_end = 10.0
t_eval = np.arange(0, t_end + 1, 1)

params = {
    'α': 0.3,
    'δ': 0.05,
    'ϕ': 0.9,
    'δ_K': 0.1,
    's': 0.2,
    'β': 0.15,
    'n': 0.01,
    'λ': 0.9
}

# Neoclassical Production Function.
def F(x, alpha, beta):
    A, K, L = x
    return A * K**(alpha) * (beta * L)**(1 - alpha)

def jones1995(t, u, params):
    alpha = params['α']
    delta = params['δ']
    beta  = params['β']
    phi   = params['ϕ']
    s     = params['s']
    delta_K = params['δ_K']
    lam   = params['λ']  
    n     = params['n']
    
    A, K, L = u
    dA = delta * (beta * L)**lam * A**phi
    dK = s * A * K**alpha * ((1 - beta) * L)**(1 - alpha) - delta_K * K
    dL = n
    return [dA, dK, dL]

def create_df(sol, params):
    df = pd.DataFrame(sol.y.T, columns=['A', 'K', 'L'])
    df['period'] = sol.t
    df['Y'] = df.apply(lambda row: F([row['A'], row['K'], row['L']], params['α'], params['β']), axis=1)
    for key, value in params.items():
        df[key] = value
    return df

def line_plot_group(df, variable, group_variable, title=None, show_legend=True):
    if title is None:
        title = str(variable)
        title = title.replace("balance_sheet", "BS")
        title = title.replace("profit_and_loss_statement", "PaL")
        title = title.replace("intermediate_inputs", "int inputs")
        title = title.replace("aggregate", "agg")
        title = title.replace("producer price index", "PPI")
        title = title.replace("__", "/")
        title = title.replace("_", " ")
        title = title.capitalize()
        if 'aggregation' in df.columns:
            aggregation_method = df['aggregation'].unique()[0]
            if aggregation_method != "mean":
                title = f"{title} ({aggregation_method})"
    
    fig, ax = plt.subplots()
    groups = df.groupby(group_variable)
    for name, group in groups:
        ax.plot(group['period'], group[variable], label=str(name), linewidth=2)
    if show_legend:
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    return ax

def multiple_line_plot_group(df, list_of_variables, group_variable):
    axes = []
    show_legend = True
    for variable in list_of_variables:
        col = df[variable]
        if not (np.all(col == 0) or col.isna().all()):
            ax = line_plot_group(df, variable, group_variable, show_legend=show_legend)
            axes.append(ax)
        show_legend = False
    return axes

#ODE simuluation


# Solving the ODE for the basic model.
sol = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)

fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label='A', linewidth=2)
ax.plot(sol.t, sol.y[1], label='K', linewidth=2)
ax.plot(sol.t, sol.y[2], label='L', linewidth=2)
Y = np.array([F(u, params['α'], params['β']) for u in sol.y.T])
ax.plot(sol.t, Y, label='Y', linewidth=2)
ax.set_title("Jones (1995) model", fontsize=9)
ax.set_xlabel("Time (t)")
ax.legend(fontsize=7)
plt.savefig("plots/jones1995_basicModel.png", dpi=400)
plt.close()


# Testing different lambdas.

# First simulation 
params['λ'] = 0.9 #(lambda = 0.9)
sol1 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df1 = create_df(sol1, params)

# Second simulation 
params['λ'] = 0.6 #(lambda = 0.6)
sol2 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df2 = create_df(sol2, params)

df_lambda = pd.concat([df1, df2], ignore_index=True)

variables = ['A', 'K', 'L', 'Y']

axes = multiple_line_plot_group(df_lambda, variables, 'λ')

n_plots = len(axes)
fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), sharex=True)
if n_plots == 1:
    axs = [axs] 

for i, variable in enumerate(variables):
    for key, group in df_lambda.groupby('λ'):
        axs[i].plot(group['period'], group[variable], label=f"λ={key}", linewidth=2)
    axs[i].set_title(variable, fontsize=9)
    axs[i].tick_params(axis='both', labelsize=7)
    if i == 0:
        axs[i].legend(fontsize=7)
axs[-1].set_xlabel("Time (t)", fontsize=9)
plt.tight_layout()
plt.savefig("plots/jones_lambdas.png", dpi=400)
plt.close()


# Test different phi.

params = {
    'α': 0.3,
    'δ': 0.05,
    'ϕ': 0.9,
    'δ_K': 0.1,
    's': 0.2,
    'β': 0.15,
    'n': 0.01,
    'λ': 0.9
}
sol1 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df1 = create_df(sol1, params)

# Second simulation (phi = 0.6)
params['ϕ'] = 0.6
sol2 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df2 = create_df(sol2, params)

df_phi = pd.concat([df1, df2], ignore_index=True)

# Plot by grouping with phi.
variables = ['A', 'K', 'L', 'Y']
n_plots = len(variables)
fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), sharex=True)
if n_plots == 1:
    axs = [axs]
for i, variable in enumerate(variables):
    for key, group in df_phi.groupby('ϕ'):
        axs[i].plot(group['period'], group[variable], label=f"ϕ={key}", linewidth=2)
    axs[i].set_title(variable, fontsize=9)
    axs[i].tick_params(axis='both', labelsize=7)
    if i == 0:
        axs[i].legend(fontsize=7)
axs[-1].set_xlabel("Time (t)", fontsize=9)
plt.tight_layout()
plt.savefig("plots/jones_phis.png", dpi=400)
plt.close()

# Testing different betas.

params = {
    'α': 0.3,
    'δ': 0.05,
    'ϕ': 0.9,
    'δ_K': 0.1,
    's': 0.2,
    'β': 0.15,
    'n': 0.01,
    'λ': 0.9
}
sol1 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df1 = create_df(sol1, params)

params['β'] = 0.3 #(beta = 0.3)
sol2 = solve_ivp(lambda t, u: jones1995(t, u, params), [0, t_end], u0, t_eval=t_eval)
df2 = create_df(sol2, params)

df_beta = pd.concat([df1, df2], ignore_index=True)

n_plots = len(variables)
fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), sharex=True)
if n_plots == 1:
    axs = [axs]
for i, variable in enumerate(variables):
    for key, group in df_beta.groupby('β'):
        axs[i].plot(group['period'], group[variable], label=f"β={key}", linewidth=2)
    axs[i].set_title(variable, fontsize=9)
    axs[i].tick_params(axis='both', labelsize=7)
    if i == 0:
        axs[i].legend(fontsize=7)
axs[-1].set_xlabel("Time (t)", fontsize=9)
plt.tight_layout()
plt.savefig("plots/jones_betas.png", dpi=400)
plt.close()


# Test change of lambda (simulate in two halves).

params = {
    'α': 0.3,
    'δ': 0.05,
    'ϕ': 0.9,
    'δ_K': 0.1,
    's': 0.2,
    'β': 0.15,
    'n': 0.01,
    'λ': 0.9
}
tspan_first = (0, t_end/2)
tspan_second = (t_end/2, t_end)
t_eval_first = np.arange(tspan_first[0], tspan_first[1] + 1, 1)
t_eval_second = np.arange(tspan_second[0], tspan_second[1] + 1, 1)

sol_first = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_first, u0, t_eval=t_eval_first)
df_both = create_df(sol_first, params)

u0_second = sol_first.y[:, -1]
sol_second = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_second, u0_second, t_eval=t_eval_second)
df_both = pd.concat([df_both, create_df(sol_second, params)], ignore_index=True)

params['λ'] = 0.1
u0_second = sol_first.y[:, -1]
sol_second_alt = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_second, u0_second, t_eval=t_eval_second)
df_both = pd.concat([df_both, create_df(sol_second_alt, params)], ignore_index=True)

fig, axs = plt.subplots(len(variables), 1, figsize=(6, 4 * len(variables)), sharex=True)
if len(variables) == 1:
    axs = [axs]
for i, variable in enumerate(variables):
    for key, group in df_both.groupby('λ'):
        axs[i].plot(group['period'], group[variable], label=f"λ={key}", linewidth=2)
    axs[i].set_title(variable, fontsize=9)
    axs[i].tick_params(axis='both', labelsize=7)
    if i == 0:
        axs[i].legend(fontsize=7)
axs[-1].set_xlabel("Time (t)", fontsize=9)
plt.tight_layout()
plt.savefig("plots/jones_change_of_lambda.png", dpi=400)
plt.close()


# Test change of saving rate.

params = {
    'α': 0.3,
    'δ': 0.05,
    'ϕ': 0.9,
    'δ_K': 0.1,
    's': 0.2,
    'β': 0.15,
    'n': 0.01,
    'λ': 0.9
}
tspan_first = (0, t_end/2)
tspan_second = (t_end/2, t_end)
t_eval_first = np.arange(tspan_first[0], tspan_first[1] + 1, 1)
t_eval_second = np.arange(tspan_second[0], tspan_second[1] + 1, 1)

sol_first = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_first, u0, t_eval=t_eval_first)
df_both = create_df(sol_first, params)

u0_second = sol_first.y[:, -1]
sol_second = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_second, u0_second, t_eval=t_eval_second)
df_both = pd.concat([df_both, create_df(sol_second, params)], ignore_index=True)

params['s'] = 0.1
u0_second = sol_first.y[:, -1]
sol_second_alt = solve_ivp(lambda t, u: jones1995(t, u, params), tspan_second, u0_second, t_eval=t_eval_second)
df_both = pd.concat([df_both, create_df(sol_second_alt, params)], ignore_index=True)

fig, axs = plt.subplots(len(variables), 1, figsize=(6, 4 * len(variables)), sharex=True)
if len(variables) == 1:
    axs = [axs]
for i, variable in enumerate(variables):
    for key, group in df_both.groupby('s'):
        axs[i].plot(group['period'], group[variable], label=f"s={key}", linewidth=2)
    axs[i].set_title(variable, fontsize=9)
    axs[i].tick_params(axis='both', labelsize=7)
    if i == 0:
        axs[i].legend(fontsize=7)
axs[-1].set_xlabel("Time (t)", fontsize=9)
plt.tight_layout()
plt.savefig("plots/jones_change_of_savingRate.png", dpi=400)
plt.close()
