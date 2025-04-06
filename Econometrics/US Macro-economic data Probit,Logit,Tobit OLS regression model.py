#US macro-economic data multivariate-regression implementation by K.Tomov

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.simplefilter('ignore')
from scipy.stats import norm


start = '1980-01-01'
end = '2025-12-31'

#fetch the dat from fred

series_info = {
    'GDP': ('GDP', 'last'),                # Nominal GDP (Q4 value)
    'Import': ('IMPGS', 'last'),           # Imports of Goods and Services (Q4 value)
    'Export': ('EXPGS', 'last'),           # Exports of Goods and Services (Q4 value)
    'CPI': ('CPIAUCSL', 'last'),           # Consumer Price Index (for inflation)
    'Unemployment': ('UNRATE', 'mean')     # Unemployment Rate (annual average)
}

def fetch_series(series_id, method, start, end):
    s = web.DataReader(series_id, 'fred', start, end)
    return s.resample('A').last() if method == 'last' else s.resample('A').mean()

data = {}
for key, (series_id, method) in series_info.items():
    data[key] = fetch_series(series_id, method, start, end)

data['Inflation Rate'] = data['CPI'].pct_change() * 100
df = pd.DataFrame({
    'GDP': data['GDP'].squeeze(),
    'Import': data['Import'].squeeze(),
    'Export': data['Export'].squeeze(),
    'Inflation_Rate': data['Inflation Rate'].squeeze(),
    'Unemployment': data['Unemployment'].squeeze()
}).dropna().loc['1980':]

df.index = df.index.year
df.index.name = 'Year'

print("US Macroeconomic Data (Annual):")
print(df.head())

#log transform variables
df['log_GDP'] = np.log(df['GDP'])
df['log_Import'] = np.log(df['Import'])
df['log_Export'] = np.log(df['Export'])
df['log_Inflation_Rate'] = np.log(df['Inflation_Rate'].replace(0, np.nan))
df['log_Unemployment'] = np.log(df['Unemployment'])
df_ols = df.dropna(subset=['log_GDP', 'log_Import', 'log_Export', 'log_Inflation_Rate', 'log_Unemployment'])


ols_formula = "log_GDP ~ log_Import + log_Export + log_Inflation_Rate + log_Unemployment"
ols_model = smf.ols(formula=ols_formula, data=df_ols)
ols_results = ols_model.fit()
print("\nPooled OLS Regression Summary:")
print(ols_results.summary())


df['GDP_growth'] = df['GDP'].pct_change()
df['Downturn'] = (df['GDP_growth'] < 0).astype(int)

df_model = df.dropna(subset=['GDP_growth', 'Downturn'])

X_vars = ['Import', 'Export', 'Inflation_Rate', 'Unemployment']
X_model = sm.add_constant(df_model[X_vars])
y = df_model['Downturn']

X_fit = np.linspace(0.01, 3, 100)
Y_fit = np.log(X_fit)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(X_fit, Y_fit)
ax.grid()
ax.set_title("y is \\log of x")
plt.show()

probit_model = sm.Probit(y, X_model)
probit_results = probit_model.fit(disp=False)
print("\nProbit Model Summary:")
print(probit_results.summary())

logit_model = sm.Logit(y, X_model)
logit_results = logit_model.fit(disp=False)
print("\nLogit Model Summary:")
print(logit_results.summary())

# Probit and Logit CDFs ---
Z = np.linspace(-5, 5, 100)
Y_probit = norm.cdf(Z)
Y_logit = 1 / (1 + np.exp(-Z))
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(Z, Y_probit, lw=3, label="Probit")
ax.plot(Z, Y_logit, lw=3, label="Logit")
ax.legend()
plt.show()

df_model['probit_prob'] = probit_results.predict(X_model)
df_model['logit_prob'] = logit_results.predict(X_model)

plt.figure(figsize=(10, 6))
plt.plot(df_model.index, df_model['probit_prob'], marker='o', label='Probit Predicted Probability')
plt.plot(df_model.index, df_model['logit_prob'], marker='s', label='Logit Predicted Probability')
plt.xlabel('Year')
plt.ylabel('Predicted Probability of Downturn')
plt.title('Predicted Probabilities from Probit and Logit Models')
plt.legend()
plt.grid(True)
plt.show()

beta1, beta2 = -40, 5
n = 400
X_tobit = np.linspace(0, 30, n)
u = 10 * np.random.randn(n)
Y_tobit = beta1 + beta2 * X_tobit + u
Y_censored = Y_tobit.clip(min=0)
fig, ax = plt.subplots(figsize=(18, 9))
ax.scatter(X_tobit, Y_tobit, s=50, edgecolors="b", facecolors="none", label="Original")
ax.scatter(X_tobit, Y_censored, s=10, color="r", label="from census")
ax.axhline(y=0, color="k", alpha=0.7)
ax.legend()
plt.show()
