 ##DSGE Business Cycle estimator model with cobb-douglas production function/ AR(1) process #### data from fred [1980-2025]
#               Implemented by K.Tomov, Original model created by Prof. F. Ruge-Murcia (2007) 

import numpy as np
from scipy import optimize, signal
import pandas as pd
from pandas_datareader.data import DataReader
import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_fprime, approx_fprime_cs
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sn
pd.set_option('float_format', lambda x: '%.3g' % x, )

#####Initial model parameters#####

# Save the names of the equations, variables, and parameters
equation_names = ['static FOC', 'euler equation', 'production','aggregate resource constraint', 'capital accumulation','labor-leisure', 'technology shock transition']
variable_names = ['output', 'consumption', 'investment','labor', 'leisure', 'capital', 'technology']
parameter_names = ['discount rate', 'marginal disutility of labor','depreciation rate', 'capital share','technology shock persistence','technology shock standard deviation']
variable_symbols = [ r"y", r"c", r"i", r"n", r"l", r"k", r"z"]
contemporaneous_variable_symbols = [r"$%s_t$" % symbol for symbol in variable_symbols]
lead_variable_symbols = [r"$%s_{t+1}$" % symbol for symbol in variable_symbols]
parameter_symbols = [r"$\beta$", r"$\psi$", r"$\delta$", r"$\alpha$", r"$\rho$", r"$\sigma^2$"]

# Stationary Parameters estimation (fixed)
parameters = pd.DataFrame({
    'name': parameter_names,
    'value': [0.80, 2, 0.025, 0.25, 0.75, 0.02] #discount rate, Marginal disutility of labor, Depreciation rate, capital share, technology shock, tech shock STD
})

class RBC1(object):
    def __init__(self, params=None):
        # Model dimensions
        self.k_params = 6
        self.k_variables = 7
        
        # Initialize parameters
        if params is not None:
            self.update(params)
    
    def update(self, params):
        # Save deep parameters
        self.discount_rate = params[0]
        self.disutility_labor = params[1]
        self.depreciation_rate = params[2]
        self.capital_share = params[3]
        self.technology_shock_persistence = params[4]
        self.technology_shock_std = params[5]
        
    def eval_logged(self, log_lead, log_contemporaneous):
        (log_lead_output, log_lead_consumption, log_lead_investment,
         log_lead_labor, log_lead_leisure, log_lead_capital,
         log_lead_technology_shock) = log_lead
        
        (log_output, log_consumption, log_investment, log_labor,
         log_leisure, log_capital, log_technology_shock) = log_contemporaneous
        
        return np.r_[
            self.log_static_foc(
                log_lead_consumption, log_lead_labor,
                log_lead_capital, log_lead_technology_shock
            ),
            self.log_euler_equation(
                log_lead_consumption, log_lead_labor,
                log_lead_capital, log_lead_technology_shock,
                log_consumption
            ),
            self.log_production(
                log_lead_output, log_lead_labor, log_lead_capital,
                log_lead_technology_shock
            ),
            self.log_aggregate_resource_constraint(
                log_lead_output, log_lead_consumption,
                log_lead_investment
            ),
            self.log_capital_accumulation(
                log_lead_capital, log_investment, log_capital
            ),
            self.log_labor_leisure_constraint(
                log_lead_labor, log_lead_leisure
            ),
            self.log_technology_shock_transition(
                log_lead_technology_shock, log_technology_shock
            )
        ]
    
    def log_static_foc(self, log_lead_consumption, log_lead_labor,
                       log_lead_capital, log_lead_technology_shock):
        return (
            np.log(self.disutility_labor) +
            log_lead_consumption -
            np.log(1 - self.capital_share) -
            log_lead_technology_shock -
            self.capital_share * (log_lead_capital - log_lead_labor)
        )
        
    def log_euler_equation(self, log_lead_consumption, log_lead_labor,
                           log_lead_capital, log_lead_technology_shock,
                           log_consumption):
        return (
            -log_consumption -
            np.log(self.discount_rate) +
            log_lead_consumption -
            np.log(
                (self.capital_share *
                 np.exp(log_lead_technology_shock) * 
                 np.exp((1 - self.capital_share) * log_lead_labor) /
                 np.exp((1 - self.capital_share) * log_lead_capital)) +
                (1 - self.depreciation_rate)
            )
        )
        
    def log_production(self, log_lead_output, log_lead_labor, log_lead_capital,
                       log_lead_technology_shock):
        return (
            log_lead_output -
            log_lead_technology_shock -
            self.capital_share * log_lead_capital -
            (1 - self.capital_share) * log_lead_labor
        )
        
    def log_aggregate_resource_constraint(self, log_lead_output, log_lead_consumption,
                                          log_lead_investment):
        return (
            log_lead_output -
            np.log(np.exp(log_lead_consumption) + np.exp(log_lead_investment))
        )
    
    def log_capital_accumulation(self, log_lead_capital, log_investment, log_capital):
        return (
            log_lead_capital -
            np.log(np.exp(log_investment) + (1 - self.depreciation_rate) * np.exp(log_capital))
        )
    
    def log_labor_leisure_constraint(self, log_lead_labor, log_lead_leisure):
        return (
            -np.log(np.exp(log_lead_labor) + np.exp(log_lead_leisure))
        )
    
    def log_technology_shock_transition(self, log_lead_technology_shock, log_technology_shock):
        return (
            log_lead_technology_shock -
            self.technology_shock_persistence * log_technology_shock
        )
        
class RBC2(RBC1):
    def steady_state_numeric(self):
        log_start_vars = [0.5] * self.k_variables
        eval_logged = lambda log_vars: self.eval_logged(log_vars, log_vars)
        result = optimize.root(eval_logged, log_start_vars)

        return np.exp(result.x)

mod2 = RBC2(parameters['value'])

steady_state = pd.DataFrame({
    'value': mod2.steady_state_numeric()
}, index=variable_names)

class RBC3(RBC2):
    
    def update(self, params):
        super(RBC3, self).update(params)
        
        self.theta = (self.capital_share / (
            1 / self.discount_rate -
            (1 - self.depreciation_rate)
        ))**(1 / (1 - self.capital_share))
        
        self.eta = self.theta**self.capital_share
    
    def steady_state_analytic(self):
        steady_state = np.zeros(7)

        numer = (1 - self.capital_share) / self.disutility_labor
        denom = (1 - self.depreciation_rate * self.theta**(1 - self.capital_share))
        steady_state[3] = numer / denom
        # Output
        steady_state[0] = self.eta * steady_state[3]
        # Consumption
        steady_state[1] = (1 - self.capital_share) * self.eta / self.disutility_labor
        # Investment
        steady_state[2] = self.depreciation_rate * self.theta * steady_state[3]
        # Labor (computed already)
        # Leisure
        steady_state[4] = 1 - steady_state[3]
        # Capital
        steady_state[5] = self.theta * steady_state[3]
        # Technology shock
        steady_state[6] = 1
        
        return steady_state
    
mod3 = RBC3(parameters['value'])

steady_state = pd.DataFrame({
    'numeric': mod3.steady_state_numeric(),
    'analytic': mod3.steady_state_analytic()
}, index=variable_names)


class RBC4(RBC3):
    
    def A_numeric(self):
        log_steady_state = np.log(self.steady_state_analytic())

        eval_logged_lead = lambda log_lead: self.eval_logged(log_lead, log_steady_state)
        
        return approx_fprime_cs(log_steady_state, eval_logged_lead)

    def B_numeric(self):
        log_steady_state = np.log(self.steady_state_analytic())
        
        eval_logged_contemporaneous = lambda log_contemp: self.eval_logged(log_steady_state, log_contemp)
        
        return -approx_fprime_cs(log_steady_state, eval_logged_contemporaneous)
    
    def C(self):
        return np.r_[[0]*(self.k_variables-1), 1]

mod4 = RBC4(parameters['value'])
        
display(pd.DataFrame(mod4.A_numeric(), index=equation_names, columns=lead_variable_symbols))
display(pd.DataFrame(mod4.B_numeric(), index=equation_names, columns=contemporaneous_variable_symbols))
display(pd.DataFrame(mod4.C(), index=equation_names, columns=[r'$\varepsilon_t$']))

class RBC5(RBC4):
    
    def update(self, params):
        super(RBC5, self).update(params)
        
        self.gamma = 1 - self.depreciation_rate * self.theta**(1 - self.capital_share)
        self.zeta = self.capital_share * self.discount_rate * self.theta**(self.capital_share - 1)
    
    def A_analytic(self):
        steady_state = self.steady_state_analytic()
        
        A = np.array([
            [0, 1, 0, self.capital_share, 0, -self.capital_share, -1],
            [0, 1, 0, self.zeta * (self.capital_share - 1), 0, self.zeta * (1 - self.capital_share), -self.zeta],
            [1, 0, 0, (self.capital_share - 1), 0, -self.capital_share, -1],
            [1, -self.gamma, (self.gamma - 1), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, -steady_state[3], -steady_state[4], 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        
        return A

    def B_analytic(self):
        
        B = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, self.depreciation_rate, 0, 0, 1 - self.depreciation_rate, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, self.technology_shock_persistence],
        ])
        
        return B

mod5 = RBC5(parameters['value'])

display(pd.DataFrame(mod5.A_analytic(), index=equation_names, columns=lead_variable_symbols))
assert(np.all(np.abs(mod5.A_numeric() - mod5.A_analytic()) < 1e-10))

display(pd.DataFrame(mod5.B_analytic(), index=equation_names, columns=lead_variable_symbols))
assert(np.all(np.abs(mod5.B_numeric() - mod5.B_analytic()) < 1e-10))

reduced_equation_names = [
    'euler equation', 'capital accumulation'
]
reduced_variable_names = [
    'consumption', 'capital'
]
reduced_parameter_names = parameter_names

reduced_variable_symbols = [
    r"c", r"k"
]
reduced_contemporaneous_variable_symbols = [
    r"$%s_t$" % symbol for symbol in reduced_variable_symbols
]
reduced_lead_variable_symbols = [
    r"$%s_{t+1}$" % symbol for symbol in reduced_variable_symbols
]

reduced_parameter_symbols = parameter_symbols

class ReducedRBC1(RBC5):
    def __init__(self, params=None):
        # Model dimensions
        self.k_params = 6
        self.k_variables = 2
        self.reduced_idx = [1, -2]
    
        if params is not None:
            self.update(params)

    def steady_state_numeric(self):
        return super(ReducedRBC1, self).steady_state_numeric()[self.reduced_idx]
        
    def steady_state_analytic(self):
        return super(ReducedRBC1, self).steady_state_analytic()[self.reduced_idx]
    
    def A(self):
        return np.eye(self.k_variables)
    
    def B(self):
        B11 = 1 + self.depreciation_rate * (self.gamma / (1 - self.gamma))
        B12 = (
            -self.depreciation_rate *
            (1 - self.capital_share + self.gamma * self.capital_share) /
            (self.capital_share * (1 - self.gamma))
        )
        B21 = 0
        B22 = self.capital_share / (self.zeta + self.capital_share*(1 - self.zeta))
        
        return np.array([[B11, B12],
                         [B21, B22]])
        
    def C(self):
        C1 = self.depreciation_rate / (self.capital_share * (1 - self.gamma))
        C2 = (
            self.zeta * self.technology_shock_persistence /
            (self.zeta + self.capital_share*(1 - self.zeta))
        )
        return np.array([C1, C2])[:,np.newaxis]
        
reduced_mod1 = ReducedRBC1(parameters['value'])

reduced_steady_state = pd.DataFrame({
    'steady state': reduced_mod1.steady_state_analytic()
}, index=reduced_variable_names)
display(reduced_steady_state.T)

display(pd.DataFrame(reduced_mod1.A(), index=reduced_equation_names, columns=reduced_lead_variable_symbols))
display(pd.DataFrame(reduced_mod1.B(), index=reduced_equation_names, columns=reduced_contemporaneous_variable_symbols))
display(pd.DataFrame(reduced_mod1.C(), index=reduced_equation_names, columns=[r'$z_t$']))

def ordered_jordan_decomposition(matrix):
    eigenvalues, right_eigenvectors = np.linalg.eig(matrix.transpose())
    left_eigenvectors = right_eigenvectors.transpose()
    
    idx = np.argsort(eigenvalues)
    
    return np.diag(eigenvalues[idx]), left_eigenvectors[idx, :]

def solve_blanchard_kahn(B, C, rho, k_predetermined):

    eigenvalues, left_eigenvectors = ordered_jordan_decomposition(B)
    left_eigenvectors = left_eigenvectors

    k_variables = len(B)
    k_nonpredetermined = k_variables - k_predetermined

    k_stable = len(np.where(eigenvalues.diagonal() < 1)[0])
    k_unstable = k_variables - k_stable

    if not k_unstable == k_nonpredetermined:
        raise RuntimeError('BK condition not met')

    decoupled_C = np.dot(left_eigenvectors, C)

    p1 = np.s_[:k_predetermined]
    p2 = np.s_[k_predetermined:]
    p11 = np.s_[:k_predetermined, :k_predetermined]
    p12 = np.s_[:k_predetermined, k_predetermined:]
    p21 = np.s_[k_predetermined:, :k_predetermined]
    p22 = np.s_[k_predetermined:, k_predetermined:]

    tmp = np.linalg.inv(left_eigenvectors[p22])
    policy_state = - np.dot(tmp, left_eigenvectors[p21])
    policy_shock = -(
        np.dot(tmp, 1. / eigenvalues[p22]).dot(
            np.linalg.inv(
                np.eye(k_nonpredetermined) -
                rho / eigenvalues[p22]
            )
        ).dot(decoupled_C[p2])
    )

    transition_state = B[p11] + np.dot(B[p12], policy_state)
    transition_shock = np.dot(B[p12], policy_shock) + C[p1]
    
    return policy_state, policy_shock, transition_state, transition_shock

class ReducedRBC2(ReducedRBC1):
    def solve(self, params=None):
        if params is not None:
            self.update(params)
        
        phi_ck, phi_cz, T_kk, T_kz = solve_blanchard_kahn(
            self.B(), self.C(),
            self.technology_shock_persistence, 1
        )
        
        inv_capital_share = 1. / self.capital_share
        tmp1 = (1 - self.capital_share) * inv_capital_share
        phi_yk = 1 - tmp1 * phi_ck
        phi_yz = inv_capital_share - tmp1 * phi_cz
        phi_nk = 1 - inv_capital_share * phi_ck
        phi_nz = inv_capital_share * (1 - phi_cz)
        design = np.r_[
            phi_yk, phi_yz, phi_nk, phi_nz, phi_ck, phi_cz
        ].reshape((3,2))
        
        # Create the transition matrix
        transition = np.r_[
            T_kk[0,0], T_kz[0,0], 0, self.technology_shock_persistence
        ].reshape((2,2))
        
        return design, transition
    

reduced_mod2 = ReducedRBC2(parameters['value'])

np.random.seed(12345)


######simulated data###### Stochastic element


# Parameters
T = 250   # number of periods to simulate
T0 = 100  # number of initial periods to "burn"

gen_eps = np.random.normal(0, reduced_mod1.technology_shock_std, size=(T+T0+1))
eps = gen_eps

reduced_mod2 = ReducedRBC2(parameters['value'])
design, transition = reduced_mod2.solve()
selection = np.array([0, 1])

raw_observed = np.zeros((T+T0+1,3))
raw_state = np.zeros((T+T0+2,2))

for t in range(T+T0+1):
    raw_observed[t] = np.dot(design, raw_state[t])
    raw_state[t+1] = np.dot(transition, raw_state[t]) + selection * eps[t]

sim_observed = raw_observed[T0+1:,:]
sim_state = raw_state[T0+1:-1,:]

fig, ax = plt.subplots(figsize=(13,4))
ax.plot(sim_observed[:,0], label='Output')
ax.plot(sim_observed[:,1], label='Labor')
ax.plot(sim_observed[:,2], label='Consumption')
ax.set_title('Simulated observed series')
ax.xaxis.grid()
ax.legend(loc='lower left')
fig.tight_layout();
plt.show()

fig, ax = plt.subplots(figsize=(13,4))
ax.plot(sim_state[:,0], label='Capital')
ax.plot(sim_state[:,1], label='Technology shock')
ax.set_title('Simulated unobserved states')
ax.xaxis.grid()
ax.legend(loc='lower left')
fig.tight_layout();
plt.show()

start='1995-01'
end = '2025-01'
labor = DataReader('HOANBS', 'fred', start=start, end=end)        # hours
consumption = DataReader('PCECC96', 'fred', start=start, end=end) # billions of dollars
investment = DataReader('GPDI', 'fred', start=start, end=end)     # billions of dollars
population = DataReader('CNP16OV', 'fred', start=start, end=end)  # population in thousands
recessions = DataReader('USRECQ', 'fred', start=start, end=end)

# Collect the raw values
raw = pd.concat((labor, consumption, investment, population.resample('QS').mean()), axis=1)
raw.columns = ['labor', 'consumption', 'investment', 'population']
raw['output'] = raw['consumption'] + raw['investment']

y = np.log(raw.output * 10**(9-3) / raw.population)
n = np.log(raw.labor * (1e3 * 40) / raw.population)
c = np.log(raw.consumption * 10**(9-3) / raw.population)
y = y.diff()[1:]
n = n.diff()[1:]
c = c.diff()[1:]

econ_observed = pd.concat((y, n, c), axis=1)
econ_observed.columns = ['output','labor','consumption']

fig, ax = plt.subplots(figsize=(13,4))
dates = econ_observed.index._mpl_repr()
ax.plot(dates, econ_observed.output, label='Output')
ax.plot(dates, econ_observed.labor, label='Labor')
ax.plot(dates, econ_observed.consumption, label='Consumption')
rec = recessions.resample('QS').last().loc[econ_observed.index[0]:].iloc[:, 0].values
ylim = ax.get_ylim()
ax.xaxis.grid()
ax.legend(loc='lower left');
plt.show()

class EstimateRBC1(sm.tsa.statespace.MLEModel):
    def __init__(self, output=None, labor=None, consumption=None,
                 measurement_errors=True,
                 disutility_labor=3, depreciation_rate=0.025,
                 capital_share=0.36, **kwargs):

        # determ the observed data
        self.output = output is not None
        self.labor = labor is not None
        self.consumption = consumption is not None
        self.observed_mask = (
            np.array([self.output, self.labor, self.consumption], dtype=bool)
        )
        
        observed_variables = np.r_[['output', 'labor', 'consumption']]
        self.observed_variables = observed_variables[self.observed_mask]
        
        self.measurement_errors = measurement_errors
        
        #Endogenous shock array
        endog = []
        if self.output:
            endog.append(np.array(output))
        if self.labor:
            endog.append(np.array(labor))
        if self.consumption:
            endog.append(np.array(consumption))
        endog = np.c_[endog].transpose()
        
        super(EstimateRBC1, self).__init__(endog, k_states=2, k_posdef=1, **kwargs)
        self.initialize_stationary()
        self.data.ynames = self.observed_variables
        
        if self.k_endog > 1 and not measurement_errors:
            raise ValueError('Stochastic singularity encountered')
        
        self.disutility_labor = disutility_labor
        self.depreciation_rate = depreciation_rate
        self.capital_share = capital_share
        
        self.structural = ReducedRBC2()
        
        self['selection', 1, 0] = 1
        
        idx = np.diag_indices(self.k_endog)
        self._idx_obs_cov = ('obs_cov', idx[0], idx[1])
        
    @property
    def start_params(self):
        start_params = [0.99, 0.5, 0.01]
        if self.measurement_errors:
            start_meas_error = np.r_[[0.1]*3]
            start_params += start_meas_error[self.observed_mask].tolist()
        
        return start_params

    @property
    def param_names(self):
        param_names = ['beta', 'rho', 'sigma.vareps']
        if self.measurement_errors:
            meas_error_names = np.r_[['sigma2.y', 'sigma2.n', 'sigma2.c']]
            param_names += meas_error_names[self.observed_mask].tolist()
        
        return param_names
    
#additional constraints for precise computation
    
    def transform_params(self, unconstrained):
        constrained = np.zeros(unconstrained.shape, unconstrained.dtype)
        constrained[0] = max(1 / (1 + np.exp(unconstrained[0])) - 1e-4, 1e-4)# Discount rate is between 0 and 1
        constrained[1] = unconstrained[1] / (1 + np.abs(unconstrained[1]))# Technology shock persistence is between -1 and 1
        constrained[2] = np.abs(unconstrained[2])# Technology shock std. dev. is positive
        if self.measurement_errors:
            constrained[3:3+self.k_endog] = unconstrained[3:3+self.k_endog]**2# Measurement error variances must be positive
        
        return constrained
    
    def untransform_params(self, constrained):
        unconstrained = np.zeros(constrained.shape, constrained.dtype)
        
        unconstrained[0] = np.log((1 - constrained[0] + 1e-4) / (constrained[0] + 1e-4))# Discount rate is between 0 and 1
        unconstrained[1] = constrained[1] / (1 + constrained[1])# Technology shock persistence is between -1 and 1
        unconstrained[2] = constrained[2] # Technology shock std. dev. is positive
        if self.measurement_errors:
            unconstrained[3:3+self.k_endog] = constrained[3:3+self.k_endog]**0.5# Measurement error variances must be positive
        
        return unconstrained
    
    def update(self, params, **kwargs):
        params = super(EstimateRBC1, self).update(params, **kwargs)
        
        structural_params = np.r_[
            params[0],
            self.disutility_labor,
            self.depreciation_rate,
            self.capital_share,
            params[1:3]
        ]
        
        # Solve the model
        design, transition = self.structural.solve(structural_params)
        
        # Update the statespace representation
        self['design'] = design[self.observed_mask, :]
        if self.measurement_errors:
            self[self._idx_obs_cov] = params[3:3+self.k_endog]
        self['transition'] = transition
        self['state_cov', 0, 0] = self.structural.technology_shock_std**2
        
sim_mod = EstimateRBC1(
    output=sim_observed[:,0],
    labor=sim_observed[:,1],
    consumption=sim_observed[:,2],
    measurement_errors=True
)

sim_res = sim_mod.fit(maxiter=1000)

print(sim_res.summary())

fig, axes = plt.subplots(2, 1, figsize=(13,7))

# Filtered states confidence intervals
states_cov = np.diagonal(sim_res.filtered_state_cov).T
states_upper = sim_res.filtered_state + 1.96 * states_cov**0.5
states_lower = sim_res.filtered_state - 1.96 * states_cov**0.5

ax = axes[0]
lines, = ax.plot(sim_res.filtered_state[0], label='Capital')
ax.fill_between(states_lower[0], states_upper[0], color=lines.get_color(), alpha=0.2)

lines, = ax.plot(sim_res.filtered_state[1], label='Technology shock')
ax.fill_between(states_lower[1], states_upper[1], color=lines.get_color(), alpha=0.2)

ax.set_xlim((0, 200))
ax.hlines(0, 0, 200)
ax.set_title('Filtered states (simulated data)')
ax.legend(loc='lower left')
ax.xaxis.grid()
ax = axes[1]

forecasts_cov = np.diagonal(sim_res.forecasts_error_cov).T
forecasts_upper = sim_res.forecasts + 1.96 * forecasts_cov**0.5
forecasts_lower = sim_res.forecasts - 1.96 * forecasts_cov**0.5

for i in range(sim_mod.k_endog):
    lines, = ax.plot(sim_res.forecasts[i], label=sim_mod.endog_names[i].title())
    ax.fill_between(forecasts_lower[i], forecasts_upper[i], color=lines.get_color(), alpha=0.1)

ax.set_xlim((0, 200))
ax.hlines(0, 0, 200)
ax.set_title('step ahead (simulated data)')
ax.legend(loc='lower left')
ax.xaxis.grid()
fig.tight_layout();
plt.show()

# Setup the statespace model
econ_mod = EstimateRBC1(
    output=econ_observed['output'],
    labor=econ_observed['labor'],
    consumption=econ_observed['consumption'],
    measurement_errors=True,
    dates=econ_observed.index
)

econ_res = econ_mod.fit(maxiter=1000, information_matrix_type='oim')

print(econ_res.summary())

fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [3, 1]})

ax_main = axes[0]
states_cov = np.diagonal(econ_res.filtered_state_cov).T
states_upper = econ_res.filtered_state + 1.96 * states_cov**0.5
states_lower = econ_res.filtered_state - 1.96 * states_cov**0.5
lines_capital, = ax_main.plot(dates, econ_res.filtered_state[0], label='Capital')
ax_main.fill_between(dates, states_lower[0], states_upper[0], color=lines_capital.get_color(), alpha=0.2)
lines_tech, = ax_main.plot(dates, econ_res.filtered_state[1], label='Technology Shock')
ax_main.fill_between(dates, states_lower[1], states_upper[1], color=lines_tech.get_color(), alpha=0.2)
ax_main.hlines(0, dates[0], dates[-1], linestyle='--', color='gray', alpha=0.5)
ax_main.set_title('Filtered States with Confidence Intervals (Full Time Range)')
ax_main.legend(loc='lower left')
ax_main.grid(True, axis='x')

ax_zoom = axes[1]
zoom_period = int(0.4 * len(dates))  # Last 40% of the data
zoom_dates = dates[-zoom_period:]

ax_zoom.plot(zoom_dates, econ_res.filtered_state[0][-zoom_period:], label='Capital (Zoomed)')
ax_zoom.fill_between(zoom_dates, states_lower[0][-zoom_period:], states_upper[0][-zoom_period:], 
                     color=lines_capital.get_color(), alpha=0.2)

ax_zoom.plot(zoom_dates, econ_res.filtered_state[1][-zoom_period:], label='Technology Shock (Zoomed)')
ax_zoom.fill_between(zoom_dates, states_lower[1][-zoom_period:], states_upper[1][-zoom_period:], 
                     color=lines_tech.get_color(), alpha=0.2)

ax_zoom.hlines(0, zoom_dates[0], zoom_dates[-1], linestyle='--', color='gray', alpha=0.5)
ax_zoom.set_title('precised confidence interval')
ax_zoom.legend(loc='lower left')
ax_zoom.grid(True, axis='x')

plt.tight_layout()
plt.show()