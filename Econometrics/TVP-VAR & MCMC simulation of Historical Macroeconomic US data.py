# TVP-VAR & MCMC of US macro-economic data simulation (from 1973 to 2024) implementation by K.Tomov

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import invwishart, invgamma
from statsmodels.tsa.tsatools import lagmat
import warnings
warnings.filterwarnings("ignore")

# Fetch the macro data from stats model api

dta = sm.datasets.macrodata.load_pandas().data
dta.index = pd.date_range('1973Q3', '2024Q1', freq='QS')
mod = sm.tsa.UnobservedComponents(dta.infl, 'llevel')
res = mod.fit()
print("Maximum likelihood param:", res.params)

sim_kfs = mod.simulation_smoother()  
sim_cfa = mod.simulation_smoother(method='cfa')  

nsimulations = 60
simulated_state_kfs = pd.DataFrame(
    np.zeros((mod.nobs, nsimulations)), index=dta.index)
simulated_state_cfa = pd.DataFrame(
    np.zeros((mod.nobs, nsimulations)), index=dta.index)

for i in range(nsimulations):
    sim_kfs.simulate()
    simulated_state_kfs.iloc[:, i] = sim_kfs.simulated_state[0]
    sim_cfa.simulate()
    simulated_state_cfa.iloc[:, i] = sim_cfa.simulated_state[0]
    
fig, axes = plt.subplots(2, figsize=(15, 6))

dta.infl.plot(ax=axes[0], color='k')
axes[0].set_title('Simulations from KFS method, MLE parameters')
simulated_state_kfs.plot(ax=axes[0], color='C0', alpha=0.25, legend=False)

dta.infl.plot(ax=axes[1], color='k')
axes[1].set_title('Simulations from CFA method, MLE parameters')
simulated_state_cfa.plot(ax=axes[1], color='C0', alpha=0.25, legend=False)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:2], ['Dataset', 'Simulated state'])
fig.tight_layout()


fig, ax = plt.subplots(figsize=(15, 3))

mod.update([4, 0.05])

for i in range(nsimulations):
    sim_kfs.simulate()
    ax.plot(dta.index, sim_kfs.simulated_state[0],
            color='C0', alpha=0.25, label='Simulated state')

dta.infl.plot(ax=ax, color='k', label='Dataset', zorder=-1)
ax.set_title('Simulations with different params')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-2:], labels[-2:])
fig.tight_layout()

# Bayesian approach extension

y = dta[['realgdp', 'cpi', 'unemp', 'tbilrate']].copy()
y.columns = ['gdp', 'inf', 'unemp', 'int']

# Converting to real GDP growth and CPI inflation rates
y[['gdp', 'inf']] = np.log(y[['gdp', 'inf']]).diff() * 100
y = y.iloc[1:]

fig, ax = plt.subplots(figsize=(15, 5))
y.plot(ax=ax)
ax.set_title('Evolution of variables under TVP-VAR')

#objective function of TVP_VAR

class TVPVAR(sm.tsa.statespace.MLEModel):
    def __init__(self, y):
        augmented = lagmat(y, 1, trim='both', original='in', use_pandas=True)
        p = y.shape[1]
        y_t = augmented.iloc[:, :p]
        z_t = sm.add_constant(augmented.iloc[:, p:])

        k_states = p * (p + 1)
        super().__init__(y_t, exog=z_t, k_states=k_states)
        
        self._index = y_t.index

        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = z_t.T

        self['transition'] = np.eye(k_states)
        self['selection'] = np.eye(k_states)
        self.ssm.initialize('known', stationary_cov=5 * np.eye(self.k_states))

    def update_variances(self, obs_cov, state_cov_diag):
        self['obs_cov'] = obs_cov
        self['state_cov'] = np.diag(state_cov_diag)

    @property
    def state_names(self):
        state_names = np.empty((self.k_endog, self.k_endog + 1), dtype=object)
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names[i] = (
                [f'intercept.{endog_name}'] +
                [f'L1.{other_name}->{endog_name}' for other_name in self.endog_names]
            )
        return state_names.ravel().tolist()

mod_tvp = TVPVAR(y)
initial_obs_cov = np.cov(y.T)
initial_state_cov_diag = [0.01] * mod_tvp.k_states
mod_tvp.update_variances(initial_obs_cov, initial_state_cov_diag)
initial_res = mod_tvp.smooth([])

def plot_coefficients_by_equation(states):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes[0, 0].plot(states.iloc[:, :5])
    axes[0, 0].set_title('GDP growth')
    axes[0, 0].legend()
    
    axes[0, 1].plot(states.iloc[:, 5:10])
    axes[0, 1].set_title('Inflation rate')
    axes[0, 1].legend()
    
    axes[1, 0].plot(states.iloc[:, 10:15])
    axes[1, 0].set_title('Unemployment equation')
    axes[1, 0].legend()

    axes[1, 1].plot(states.iloc[:, 15:20])
    axes[1, 1].set_title('Interest rate equation')
    axes[1, 1].legend()
    
    fig.tight_layout()
    return fig

plot_coefficients_by_equation(pd.DataFrame(
    initial_res.states.smoothed, 
    index=mod_tvp._index, 
    columns=mod_tvp.state_names)
)

# Bayesian estimation via Monte-Carlo-Markov-Chain

v10 = mod_tvp.k_endog + 3
S10 = np.eye(mod_tvp.k_endog)
vi20 = 6
Si20 = 0.01

# sampler setup
niter = 15000
nburn = 5000

store_states = np.zeros((niter + 1, mod_tvp.nobs, mod_tvp.k_states))
store_obs_cov = np.zeros((niter + 1, mod_tvp.k_endog, mod_tvp.k_endog))
store_state_cov = np.zeros((niter + 1, mod_tvp.k_states))
store_obs_cov[0] = initial_obs_cov
store_state_cov[0] = initial_state_cov_diag
mod_tvp.update_variances(store_obs_cov[0], store_state_cov[0])
sim = mod_tvp.simulation_smoother(method='cfa')

for i in range(niter):
    mod_tvp.update_variances(store_obs_cov[i], store_state_cov[i])
    sim.simulate()
    store_states[i + 1] = sim.simulated_state.T
    design_T = mod_tvp['design'].transpose(2, 0, 1)
    fitted = np.matmul(design_T, store_states[i + 1][..., None])[..., 0]
    resid = mod_tvp.endog - fitted
    store_obs_cov[i + 1] = invwishart.rvs(v10 + mod_tvp.nobs, S10 + resid.T @ resid)

    resid_states = store_states[i + 1, 1:] - store_states[i + 1, :-1]
    sse = np.sum(resid_states**2, axis=0)
    
    for j in range(mod_tvp.k_states):
        rv = invgamma.rvs((vi20 + mod_tvp.nobs - 1) / 2, scale=(Si20 + sse[j]) / 2)
        store_state_cov[i + 1, j] = rv

states_posterior_mean = pd.DataFrame(np.mean(store_states[nburn + 1:], axis=0), index=mod_tvp._index, columns=mod_tvp.state_names)
plot_coefficients_by_equation(states_posterior_mean)
plt.show()

