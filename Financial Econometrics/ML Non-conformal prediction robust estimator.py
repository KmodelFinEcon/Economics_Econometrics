
#non-conformal prediction robust estimator vs normal distribution residual plot by Kt.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from scipy.stats import norm
from nonconformist.cp import IcpRegressor
from nonconformist.nc import RegressorNc, AbsErrorErrFunc

# Generate data
np.random.seed(60)
n_samples = 1000
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 2 * X.squeeze() + np.random.normal(0, 1, n_samples)

# training and calibration sets
train_size = 200
X_train, X_cal = X[:train_size], X[train_size:]
y_train, y_cal = y[:train_size], y[train_size:]

# Robust Model 95% confidence interval
model = HuberRegressor(epsilon=1.35)
model.fit(X_train, y_train)

# Conformal prediction initiation 
nc = RegressorNc(model, AbsErrorErrFunc())
icp = IcpRegressor(nc)
icp.fit(X_train, y_train)
icp.calibrate(X_cal, y_cal)

# Generate predictions and conformal intervals for test set
X_test = np.linspace(0, 10, 1000).reshape(-1, 1)
predictions = icp.predict(X_test, significance=0.05)  # 95% prediction interval
y_pred = model.predict(X_test)

# normal distribution confidence intervals based on training residuals
residuals_train = y_train - model.predict(X_train)
std_dev = np.std(residuals_train)
normal_conf_interval = norm.interval(0.90, loc=y_pred, scale=std_dev)

# Plot regression predictions with conformal and normal prediction intervals
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data', alpha=0.5)
plt.plot(X_test, y_pred, color='red', label='Linear Regression Prediction')
plt.fill_between(X_test.squeeze(), predictions[:, 0], predictions[:, 1],
                 color='orange', alpha=0.3, label='Conformal Prediction Interval (90%)')
plt.fill_between(X_test.squeeze(), normal_conf_interval[0], normal_conf_interval[1],
                 color='green', alpha=0.3, label='Normal Distribution Confidence Interval (90%)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Conformal Prediction vs Normal Distribution Prediction Intervals')
plt.legend()
plt.show()

#  histogram of residuals and the fitted normal distribution
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(residuals_train, bins=20, density=True, alpha=0.6, color='g',
                            label='Training Residuals')

mu, sigma = np.mean(residuals_train), np.std(residuals_train)
x_values = np.linspace(bins[0], bins[-1], 100)
pdf = norm.pdf(x_values, mu, sigma)
plt.plot(x_values, pdf, 'k', linewidth=2, label='Fitted Normal PDF')

normal_interval = norm.interval(0.90, loc=mu, scale=sigma)
plt.axvline(normal_interval[0], color='r', linestyle='--', label='Normal 90% Interval')
plt.axvline(normal_interval[1], color='r', linestyle='--')

plt.xlabel('Residual')
plt.ylabel('Density')
plt.title('Residuals with Fitted Normal Distribution')
plt.legend()
plt.show()