
import pandas as pd
import numpy as np

df = pd.read_csv('/content/breast-cancer.csv')

df.head()

X = df[['texture_mean', 'area_mean']]
df['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
y = df['diagnosis']

y

import seaborn as sns
import matplotlib.pyplot as plt

# colors = {'M': 'red', 'B': 'blue'}
# plt.scatter(df['texture_mean'], df['area_mean'], c=df['diagnosis'].map(colors))
# colors = {'1': 'red', '0': 'blue'}
plt.xlabel('Texture Mean')
plt.ylabel('Area Mean')
plt.scatter(df['texture_mean'], df['area_mean'], c=df['diagnosis'])

from sklearn.linear_model import Perceptron
p = Perceptron()

p.fit(X,y)

p.coef_, p.intercept_ # values of w1, w2 and b

from mlxtend.plotting import plot_decision_regions

# df['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
plot_decision_regions(X.values, y.values, clf=p, legend=3)
