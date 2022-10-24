from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_hello_message():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'msg': 'Hello! Welcome to my deploy project!'}


def test_predict_without_body():
    r = client.post('/predict')
    assert r.status_code != 200


def test_predict():
    body = {
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
    r = client.post('/predict', json=body)
    assert r.status_code == 200
    assert r.json()['prediction'] == 0
