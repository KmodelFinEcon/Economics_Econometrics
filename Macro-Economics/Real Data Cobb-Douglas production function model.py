# Cobb-Douglas Production Function Model from US FRED Macro-economic data 
#               by. K.Tomov

import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

##### PART 1: Data Collection + OLS regression

#data fetch
start = datetime.datetime(1980, 1, 1)
end = datetime.datetime(2024, 12, 31)
gdp = web.DataReader('GDPC1', 'fred', start, end) #Quaterly GDP figures
capital = web.DataReader('RKNANPUSA666NRUG', 'fred', start, end)# Real Capital Stock
labor = web.DataReader('CE16OV', 'fred', start, end) #civilian Employment

# Year-end convertion
gdp_annual = gdp.resample('YE').last()
labor_annual = labor.resample('YE').mean()
capital_annual = capital.resample('YE').last()

data = pd.concat([gdp_annual, capital_annual, labor_annual], axis=1, join='inner')
data.columns = ['GDP', 'Capital', 'Labor']
data.dropna(inplace=True)

#log construction
data['ln_GDP'] = np.log(data['GDP'])
data['ln_Capital'] = np.log(data['Capital'])
data['ln_Labor'] = np.log(data['Labor'])

X = data[['ln_Capital', 'ln_Labor']]
X = sm.add_constant(X) #intercept
y = data['ln_GDP']

# OLS fitting!!
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Get parrameters 
alpha = results.params['ln_Capital']
beta = results.params['ln_Labor']
A = np.exp(results.params['const'])

print(f"Estimated Parameters")
print(f"alpha [Capital Elasticity]: {alpha:.3f}")
print(f"beta [Labor Elasticity]: {beta:.3f}")
print(f"A [Total Factor Productivity]: {A:.3f}")
print(f"RTS [alpha & beta]: {alpha + beta:.3f}")

# GDP prediction with Cobb-Douglas
data['Predicted_GDP'] = A * (data['Capital'] ** alpha) * (data['Labor'] ** beta)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['GDP'], label='Actual GDP')
plt.plot(data.index, data['Predicted_GDP'], label='Predicted GDP', linestyle='--')
plt.title('CD fit vs actual GDP')
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.legend()
plt.show()

#### PART 2: 3D Plot of the Production Function

K = np.linspace(data['Capital'].min(), data['Capital'].max(), 100)
L = np.linspace(data['Labor'].min(), data['Labor'].max(), 100)
K_grid, L_grid = np.meshgrid(K, L)
GDP_surface = A * (K_grid ** alpha) * (L_grid ** beta)
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Capital'], data['Labor'], data['GDP'],
           c='red', s=50, label='Real Observation', depthshade=False)
surf = ax.plot_surface(K_grid, L_grid, GDP_surface, cmap='magma', alpha=0.7)

ax.plot(data['Capital'], data['Labor'], data['GDP'],
        c='black', lw=2, label='Path')
surface_patch = mpatches.Patch(color=plt.cm.magma(0.5), label='Estimated Production Function')

ax.set_xlabel('Capital Stock', fontsize=10)
ax.set_ylabel('Labor Force', fontsize=10)
ax.set_zlabel('Real GDP', fontsize=10)
ax.set_title('Cobb-Douglas Production Function', fontsize=13)

handles, labels = ax.get_legend_handles_labels()
handles.append(surface_patch)
labels.append('Estimated Production Function')
ax.legend(handles, labels)

for xi, yi, zi in zip(data['Capital'], data['Labor'], data['GDP']):
    ax.plot([xi, xi], [yi, yi], [0, zi], c='gray', alpha=0.3)
    ax.plot([xi, xi], [data['Labor'].min(), yi], [zi, zi], c='gray', alpha=0.3)
    ax.plot([data['Capital'].min(), xi], [yi, yi], [zi, zi], c='gray', alpha=0.3)

plt.tight_layout()
plt.show()