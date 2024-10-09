import requests
import pandas as pd
import json
import os

# Define the API endpoint URL
api_url = "http://localhost:8000/predict"

# Get the absolute path to the parent directory (sound_realty)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Construct the path to the future_unseen_examples.csv file
csv_file_path = os.path.join(repo_root, 'app', 'data', 'adult.csv')

# Load test data
data = pd.read_csv(csv_file_path).head(5)

# Convert the DataFrame to a list of dictionaries (JSON)
data_json = data.to_dict(orient="records")

# Print the JSON payload
print("JSON Payload:")
print(json.dumps(data_json, indent=4))

# Send the POST request to the API with the data
response = requests.post(api_url, json=data_json)

# Check the response status code
if response.status_code == 200:
    print("Success!")
    print("API Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Failed to connect to API. Status code: {response.status_code}")
    print("Response Text:")
    print(response.text)
