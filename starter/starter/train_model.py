# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
import json

import sys
sys.path.append('./starter/starter/ml')

from data import process_data
from model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv('starter/data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

scores = {
    'precision': round(precision, ndigits=3),
    'recall': round(recall, ndigits=3),
    'fbeta': round(fbeta, ndigits=3)
}

with open('starter/screenshots/test_set_scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

with open('starter/model/model.pkl', 'wb') as model_file, \
     open('starter/model/encoder.pkl', 'wb') as encoder_file, \
     open('starter/model/lb.pkl', 'wb') as lb_file:
    pickle.dump(model, model_file)
    pickle.dump(encoder, encoder_file)
    pickle.dump(lb, lb_file)
