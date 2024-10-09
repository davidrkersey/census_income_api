import datetime
import os
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

router = APIRouter()

# The model and model info are now stored in the Docker volume at /app/model/
model_dir = "/app/model"

# Construct the full path to the model.pkl file and model params JSON
model_file_path = os.path.join(model_dir, 'model.pkl')
model_info_file_path = os.path.join(model_dir, 'model_info.json')

# Function to load the model and model info
def load_model():
    with open(model_file_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    with open(model_info_file_path, "r") as model_info_file:
        loaded_model_info = json.load(model_info_file)
    return loaded_model, loaded_model_info

def load_transform_data(json_data: dict) -> pd.DataFrame:
    """Load dataset from JSON and transform"""
    # Convert JSON data to a DataFrame
    data = pd.DataFrame(json_data)
    
    # Handle missing values
    dataset = data.fillna(np.nan)

    # Map income values to binary
    dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

    # Feature Engineering
    # Convert 'sex' to binary values
    dataset["sex"] = dataset["sex"].map({"Male": 0, "Female": 1})

    # Create 'marital.status' as binary (Married = 1, Single = 0)
    dataset["marital.status"] = dataset["marital.status"].replace(
        ['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')
    dataset["marital.status"] = dataset["marital.status"].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
    dataset["marital.status"] = dataset["marital.status"].map({"Married": 1, "Single": 0})
    dataset["marital.status"] = dataset["marital.status"].astype(int)

    # Ensure that only the relevant features are kept
    relevant_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'sex', 'marital.status']
    dataset = dataset[relevant_features]

    return dataset

# Initial load of the model and model info
model, model_info = load_model()
current_version = model_info['version']

@router.post("/predict")
async def predict(input_data: list[dict]):
    """
    API endpoint to make predictions. Accepts a list of dictionaries, filters out demographic data,
    adds relevant demographic data based on zipcodes, and returns predictions.

    :param input_data: A list of dictionaries containing the input data.
    :return: A JSON response with predictions and metadata.
    """
    global model, model_info, current_version

    try:
        # Load the current model info to check the model version
        with open(model_info_file_path, "r") as model_info_file:
            latest_model_info = json.load(model_info_file)
            latest_version = latest_model_info['version']
            print(latest_version)
            print(current_version)

        # Check if the model version has changed
        if latest_version != current_version:
            model, model_info = load_model()
            current_version = latest_version
            print(f"Model updated to version {current_version} at {datetime.datetime.now()}")

        # Convert the list of input dictionaries to a DataFrame
        input_df = load_transform_data(input_data)

        # Make a prediction using the enriched data
        predictions = model.predict(input_df)

        # Prepare the response with predictions, model version, and datetime as metadata
        response = [
            {
                "prediction": prediction,
                "model_version": model_info['version'],
                "model_train_date": model_info['train_date'],
                "prediction_date": datetime.datetime.now()
            }
            for prediction in predictions
        ]

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
