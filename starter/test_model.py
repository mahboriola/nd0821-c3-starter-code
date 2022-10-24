import pytest
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb

import sys
sys.path.append('./starter/starter/ml')

from data import process_data
from model import train_model, inference, compute_model_metrics

data = pd.read_csv('starter/data/census.csv')

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

with open('starter/model/model.pkl', 'rb') as f:
    model = pickle.load(f)

def test_train_model():
    trained_model = train_model(X_train, y_train)
    assert isinstance(trained_model, xgb.XGBClassifier)

def test_inference():
    y_pred = inference(model, X_test)
    assert y_pred.shape == y_test.shape

def test_compute_model_metrics():
    precision, recall, fbeta = compute_model_metrics(y_test, y_test)
    assert precision == 1 and recall == 1 and fbeta == 1

test_train_model()
test_inference()
test_compute_model_metrics()