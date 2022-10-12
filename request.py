import json
import requests

url = 'http://127.0.0.1:8000/diabetes_prediction'

input_data_for_model = {

    'Glucose' : 166,
    'BloodPressure' : 72,
    'Insulin' : 175,
    'BMI' : 25.8
    
    }

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)


