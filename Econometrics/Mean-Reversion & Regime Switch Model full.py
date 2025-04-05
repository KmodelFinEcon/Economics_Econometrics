# Mean-reversion & regime switches model

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import pandas as pd
import random
import numpy as np
import scipy.stats as scs
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from collections import Counter, deque
from imblearn.under_sampling import RandomUnderSampler


df = yf.download('MSFT', start="2015-01-01", end="2025-03-15", interval="1d")
df.head()

plt.title(f'MSFT Price History')
plt.plot(
    list(i for i in df.index),
    list(i[4] for i in df.values))
plt.grid(False)

plt.show()

df['r-1'] = df['Close'].pct_change().dropna()

ex_ret = df['r-1']
ex_ret.plot(title='Excess returns', figsize=(12, 3), grid=False)

# Fit the markov regression model
mod_kns = sm.tsa.MarkovRegression(ex_ret.dropna(), k_regimes=2, trend='n', switching_variance=True)
res_kns = mod_kns.fit()
res_kns.summary()

res_kns.smoothed_marginal_probabilities.head()

print(res_kns.smoothed_marginal_probabilities.head())

fig, axes = plt.subplots(2, figsize=(10,7))
ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.grid(False)
ax.set(title='Smoothed probability of a low-variance regime returns')
ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a high-variance regime returns')
fig.tight_layout()
ax.grid(False)

plt.show()

low_var = list(res_kns.smoothed_marginal_probabilities[0])
high_var = list(res_kns.smoothed_marginal_probabilities[1])

regime_list = []
for i in range(0, len(low_var)):
    if low_var[i] > high_var[i]:
        regime_list.append(0)
    else:
        regime_list.append(1)
        
regime_df = pd.DataFrame()
regime_df['regimes'] = regime_list

sns.set(font_scale=1.5)

if isinstance(df.columns, pd.MultiIndex):# Flatten the df columns if they are a MultiIndex (this fixes the merge error)
    df.columns = df.columns.get_level_values(0)
df = df.iloc[len(df)-len(regime_list):]
regimes = (pd.DataFrame(regime_list, columns=['regimes'], index=df.index)
          .join(df, how='inner')
          .reset_index(drop=False)
          .rename(columns={'index':'Date'}))
regimes.head()

warnings.filterwarnings("ignore")
colors = 'green', 'red', 'yellow'
sns.set_style("whitegrid")
order = [0, 1]
fg = sns.FacetGrid(data=regimes, hue='regimes', hue_order=order,
                   palette=colors, aspect=1.31, height=12)
fg.map(plt.scatter, 'Date', "Close", alpha=0.8).add_legend()
sns.despine(offset=10)
fg.fig.suptitle('Historical MSFT code', fontsize=24, fontweight='demi')

plt.show()

#random forest classifier (ML)

ml_df = yf.download('MSFT', start="2015-01-01", end="2025-03-15", interval="1d")

# price and volume returns
for i in [1, 2, 3, 5, 7, 14, 21]:
    ml_df[f'Close_{i}_Value'] = ml_df['Close'].pct_change(i)
    ml_df[f'Volume_{i}_Value'] = ml_df['Volume'].pct_change(i)
ml_df.dropna(inplace=True)

# probabilities
low_var_prob = list(res_kns.smoothed_marginal_probabilities[0])
high_var_prob = list(res_kns.smoothed_marginal_probabilities[1])
ml_df['Low_Var_Prob'] = low_var_prob[len(low_var_prob)-len(ml_df):] # adjust length
ml_df['High_Var_Prob'] = high_var_prob[len(high_var_prob)-len(ml_df):]

# volatility     
for i in [3, 7, 14, 21]:
    ml_df[f'Volt_{i}_Value'] = np.log(1 + ml_df['Close']).rolling(i).std()
    
ml_df.dropna(inplace=True)

# states changes
ml_df['regimes'] = regime_list[len(regimes)-len(ml_df):] # adjust length

ml_df.head()

# modify

ml_df['regimes'] = ml_df['regimes'].shift(-1) 
ml_df.dropna(inplace=True)
ml_df

rf_df = ml_df.copy()
labels = rf_df.pop('regimes') # get target states into a variable
labels = labels.astype('int')

X_train, X_test, y_train, y_test = train_test_split(rf_df, labels, test_size=0.2)
Counter(y_train)

# undersample our low_variance regime examples
under_sampler = RandomUnderSampler(random_state=40)
X_rs, y_rs = under_sampler.fit_resample(X_train, y_train)

Counter(y_rs)

rf = RandomForestClassifier(n_estimators=40) #number of trees 
rf.fit(X_rs, y_rs)

y_pred = rf.predict(X_test)
y_prob_pred = rf.predict_proba(X_test)

acc_score = accuracy_score(y_test, y_pred)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('accuracy_score=', acc_score,'roc=', roc_auc, 'FPR=', false_positive_rate[1],  'TPR=', true_positive_rate[1])

n_estimators = [1, 2, 4, 8, 10, 20, 30, 40, 50, 100]
train_results = [2,4,50]
test_results = [2,4,20]

for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator)
    rf.fit(X_rs, y_rs)
    train_pred = rf.predict(X_rs)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_rs, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
line2, = plt.plot(n_estimators, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.grid(False)
plt.show()

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_rs, y_rs)

results = permutation_importance(rf, X_rs, y_rs, scoring='accuracy')

importance = results.importances_mean

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.grid(False)
plt.show()

ml_df.columns[25]