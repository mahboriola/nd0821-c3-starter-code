from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import json

import sys
sys.path.append('./starter/starter/ml')

from model import compute_data_slices_metrics

with open('starter/model/model.pkl', 'rb') as model_file, \
     open('starter/model/encoder.pkl', 'rb') as encoder_file, \
     open('starter/model/lb.pkl', 'rb') as lb_file:
    model = pickle.load(model_file)
    encoder = pickle.load(encoder_file)
    lb = pickle.load(lb_file)

data = pd.read_csv('starter/data/census.csv')
_, test = train_test_split(data, test_size=0.20)

scores = compute_data_slices_metrics(test, model, encoder, lb)

with open('starter/screenshots/slice_output.json', 'w') as f:
    json.dump(scores, f, indent=4)