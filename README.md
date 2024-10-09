# Census Income Model API

## Overview

The Census Income API is a FastAPI-based application designed to serve predictions for Census data. The application includes a machine learning model that predicts income based on various features, including demographic data. The API is designed to be deployed in a production environment using Docker, with the flexibility to update the model through a training script that integrates with a locally-hosted MLflow registry and pushes the new model to the Docker-hosted API.

## Features

- **RESTful API:** FastAPI-based application that serves predictions through a `/predict` endpoint.
- **MLflow Integration:** Tracks model training experiments using MLflow, storing metrics, parameters, and artifacts in a locally hosted SQLite database.
- **Automated Model Deployment:** The training script (`create_model.py`) automatically trains a new model, tracks the experiment locally in MLflow, and remotely updates the model served by the API.
- **Dockerized Deployment:** The application is containerized using Docker, making it easy to deploy in various environments.
- **Scalable Architecture:** Designed to allow model updates without downtime by utilizing Docker volumes.

## Project Structure

```
census_income/
├── app/                        # Application code and related files
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── routers/                # Routers for API endpoints
│   │   ├── __init__.py
│   │   └── predict.py          # API logic for prediction
│   ├── utils/                  # Utility functions and configurations
│   │   ├── __init__.py
│   │   └── config.py           # Configuration including model and CSV paths
│   ├── model/                  # Directory for the model files
│   │   └── model.pkl           # Serialized model file copied from `train/`
│   ├── data/                   # Data files
│   │   ├── adult.csv
│   └── deployment/             # Deployment configurations
│       ├── Dockerfile          # Dockerfile for building the app container
│       └── requirements.txt    # Python dependencies
│
├── test/                      # Test for the application
│   └── test_api.py             # Test script for the API
│
├── train/                      # Training scripts and related files
│   ├── create_model.py         # Script for training the model, integrated with MLflow
│   ├── data/                   # Data files for training
│   │   ├── adult.csv
│   └── push_model_to_docker.sh # Script to push the trained model to the Docker volume
└── README.md                   # Documentation for the project
```

## Setup Instructions

### Prerequisites

- Docker
- Python 3.7+
- MLflow (included in `requirements.txt`)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/davidrkersey/census_income.git
   cd census_income
   ```

2. **Set Up the Python Environment:**

   Use `pipenv` or `virtualenv` to create a virtual environment, then install the dependencies:

   ```bash
   pip install -r app/deployment/requirements.txt
   ```

3. **Set Up Docker:**

   Build the Docker image for the application:

   ```bash
   docker build -f app/deployment/Dockerfile -t census_income .
   ```

4. **Run the Application:**

   Start the FastAPI application using Docker:

   ```bash
   docker run -p 8000:8000 sound_realty
   ```

   The API will be accessible at `http://localhost:8000`.

### Training and Updating the Model

1. **Train the Model:**

   The training script is located in `train/create_model.py`. This script:

   - Loads the data from the specified CSV files.
   - Trains a model using a KNN regressor.
   - Logs the training process, including metrics and artifacts, to MLflow.
   - Saves the trained model to `app/model/model.pkl`.

   Run the training script:

   ```bash
   python train/create_model.py
   ```

2. **Automatically Update the Model in Docker:**

   The `create_model.py` script is designed to automatically copy the newly trained model to the Docker volume used by the FastAPI application. This allows the API to serve the latest model without requiring a container restart.

### Testing the API

A basic test script is provided in `test/test_api.py`. This script sends example data from the 'future_unseen_examples.csv' dataset to the `/predict` endpoint and checks the response.

Run the test script:

```bash
python tests/test_api.py
```

### MLflow Integration

MLflow is used to track experiments. By default, MLflow logs are stored in a SQLite database located at `train/mlflow.db` after the model trains for the first time.

To start the MLflow UI and explore the experiments:

```bash
mlflow ui
```

Access the MLflow UI at `http://localhost:5000`.

### API Endpoints

- **`POST /predict`**: Accepts JSON input and returns predictions along with metadata such as model version and timestamp.

Example request:

```json
POST /predict
Content-Type: application/json

[
    {
        "age":31, 
        "fnlwgt":5, 
        "education.num":4, 
        "capital.gain":1, 
        "capital.loss":0, 
        "hours.per.week":40, 
        "sex":1, 
        "marital.status":0
    }
]
```

Example response:

```json
{
    "predictions": 49000,
    "model_version": "1.0",
    "datetime": "2024-08-20T10:00:00Z"
}
```