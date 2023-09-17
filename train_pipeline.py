import numpy as np
import pickle

from sklearn.datasets import load_iris
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

data = load_iris()

X = data.data
y = data.target
features = data.feature_names

pipe = make_pipeline(StandardScaler(), SimpleImputer(),  LogisticRegression())
pipe.fit(X,y)

dump(features, 'features.pkl')
dump(pipe, 'pipe.pkl')
