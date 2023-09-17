import numpy as np
import pickle

from sklearn.datasets import load_iris
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = load_iris()

X = data.data
y = data.target
features = data.feature_names

sc = StandardScaler()
X = sc.fit_transform(X)

clf = LogisticRegression()
clf.fit(X,y)

dump(features, 'features.pkl')
dump(clf, 'model.pkl')

