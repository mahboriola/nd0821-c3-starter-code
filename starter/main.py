# Put the code for your API here.
from doctest import Example
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data

app = FastAPI()

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

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
            }
        }

with open('starter/model/model.pkl', 'rb') as m, \
     open('starter/model/encoder.pkl', 'rb') as e, \
     open('starter/model/lb.pkl', 'rb') as l:
    model = pickle.load(m)
    encoder = pickle.load(e)
    lb = pickle.load(l)

@app.get('/')
async def hello_message():
    return {'msg': 'Hello! Welcome to my deploy project!'}

@app.post('/predict')
async def predict_data(data: Data): 
    data = pd.DataFrame(data.dict(), index=[0])
    
    data.columns = [col_name.replace('_', '-') for col_name in data.columns]
    
    data, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None, training=False,
        encoder=encoder, lb=lb
    )
    
    preds = inference(model, data)
    return {'prediction': int(preds[0])}

