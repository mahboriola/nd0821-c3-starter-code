from sklearn.metrics import fbeta_score, precision_score, recall_score
import xgboost as xgb
import json

import sys
sys.path.append('./starter/starter/ml')
from data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = xgb.XGBClassifier(n_estimators=1000)
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : xgboost.XGBClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred


def compute_data_slices_metrics(data, model, encoder, lb):
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

    scores = {}

    for feat in cat_features:
        scores[feat] = {}
        for feat_value in data[feat].unique():
            df_temp = data[data[feat] == feat_value]

            X_test, y_test, _, _ = process_data(
                df_temp, categorical_features=cat_features,
                label="salary", training=False,
                encoder=encoder, lb=lb
            )

            y_pred = inference(model, X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            scores[feat][feat_value] = {
                'precision': round(precision, ndigits=3),
                'recall': round(recall, ndigits=3),
                'fbeta': round(fbeta, ndigits=3)
            }

    return scores
