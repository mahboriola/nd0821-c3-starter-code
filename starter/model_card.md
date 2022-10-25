# Model Card

## Model Details
A XGBoost Classifier model that predicts the salary based on census data.

Model: [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
Hyperparameters:
- `n_estimators=1000`

## Intended Use
The model developed in this project is intended to classify salary based on the Census dataset (more details in the next section).

## Training Data
The data used to train the model was the Census data from UCI.
More precisely, 80% of the provided data was used to train the model.

The training data were preprocessed before being used in model training.
For the categorical features, the OneHot encoding was used.
The Continuous/Numerical features were kept the way they were in the dataset.
The target feature (`salary`) was processed using a LabelBinarizer encoder.

Source: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
20% of the data provided was used to test the model.
The same data processing used in training data was used in the evaluation data.

## Metrics
- Precision: 0.944
- Recall: 0.907
- Fbeta: 0.925

## Ethical Considerations
The used dataset contains some sensitive data because the dataset is extracted from the 1994 Census database.
This project was intended to be just an educational project and not be deployed in production.

## Caveats and Recommendations
The model was trained in a "single run", it was not cross-validated, and neither any hyperparameter tuning was applied. So, maybe with another model or with the procedures mentioned before it should get better scores.
