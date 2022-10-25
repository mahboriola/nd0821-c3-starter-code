import requests

data = {
    "age": 31,
    "workclass": "Private",
    "fnlgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 14084,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States"
}

r = requests.post(
    "https://ml-model-w-fastapi.herokuapp.com/predict",
    json=data
)

print("POST Request Response")
print(f"Status code: {r.status_code}")
print(f"Response data: {r.json()}")