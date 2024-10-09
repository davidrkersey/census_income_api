import json
import pathlib
from typing import List, Tuple
import pickle
import pandas as pd
from sklearn import model_selection, neighbors, pipeline, preprocessing, metrics
import mlflow
import mlflow.sklearn
import os
import datetime
import subprocess
import shutil

#--------
import pandas as pd
import numpy as np

from collections import Counter

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to save model outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "train", "model")

# Path to the SQLite database
DATABASE_PATH = os.path.join(BASE_DIR, "train", "mlflow.db")

# Path to the CSV files
DATA_PATH = os.path.join(BASE_DIR, "train", "data", "adult.csv")

def setup_mlflow():
    """Sets up MLflow to use SQLite as the backend store."""
    if not os.path.exists(DATABASE_PATH):
        # Automatically create the SQLite database if it doesn't exist
        with open(DATABASE_PATH, 'w'):
            pass

    # Set the tracking URI to the SQLite database
    mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")

def load_transform_data(
    data_path: str
) -> pd.DataFrame:
    """Load dataset and transform"""
    data = pd.read_csv(data_path)
    dataset = data.fillna(np.nan)
    dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
    # Identify Numeric features
    numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']
    # Identify Categorical features
    cat_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native']

    #Feature Engineer
    # Convert Sex value to binary
    dataset["sex"] = dataset["sex"].map({"Male": 0, "Female":1})

    # Create Married Column - Binary Yes(1) or No(0)
    dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
    dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
    dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
    dataset["marital.status"] = dataset["marital.status"].astype(int)

    # Ensure that only the relevant features are kept
    relevant_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'sex', 'marital.status','income']
    dataset = dataset[relevant_features]

    return dataset


def save_model_artifacts(model, metrics, params, model_info, artifacts_dir):
    """Save model, metrics, parameters, and model metadata."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save model as pickle
    pickle_file_path = os.path.join(artifacts_dir, "model.pkl")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {pickle_file_path}")

    # Save metrics as JSON
    metrics_file_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved as {metrics_file_path}")

    # Save parameters as JSON
    params_file_path = os.path.join(artifacts_dir, "params.json")
    with open(params_file_path, "w") as f:
        json.dump(params, f)
    print(f"Params saved as {params_file_path}")

    # Save model info as JSON
    model_info_file_path = os.path.join(artifacts_dir, "model_info.json")
    with open(model_info_file_path, "w") as f:
        json.dump(model_info, f)
    print(f"Model info saved as {model_info_file_path}")

    return pickle_file_path, model_info_file_path

def update_bash_script(container_id, bash_script_path):
    """Update the bash script with the current Docker container ID."""
    
    # Define the path for the temporary script file
    temp_dir = './train/tmp'
    temp_script_path = os.path.join(temp_dir, 'push_model_to_docker.sh')
    
    # Ensure the temporary directory exists
    os.makedirs(temp_dir, exist_ok=True)

    #Copy bash script
    shutil.copyfile(bash_script_path, temp_script_path)
        
    # Replace the placeholder with the actual container ID in the temporary script
    with open(temp_script_path, 'r') as file:
        script_content = file.read()
    
    # Replace the placeholder with the actual container ID
    script_content = script_content.replace('__CONTAINER_ID__', container_id)
    
    # Write the updated content back to the bash script
    with open(temp_script_path, 'w') as file:
        file.write(script_content)

def push_files_to_docker(pickle_file_path, model_info_file_path):
    """Push the model and JSON files to the Docker container."""
    # Retrieve the container ID or name
    result = subprocess.run(['docker', 'ps', '-q', '-f', 'ancestor=census_income'], stdout=subprocess.PIPE)
    container_id = result.stdout.decode('utf-8').strip()

    if not container_id:
        print("No running container found for the image 'census_income'.")
        return
    
    # Path to the bash script
    bash_script_path = './train/push_model_to_docker.sh'

    # Update the bash script with the current container ID
    update_bash_script(container_id, bash_script_path)

    # Path to the bash script
    updated_bash_script_path = './train/tmp/push_model_to_docker.sh'

    try:
        # Use the full path to Git Bash
        git_bash_path = "C:/Program Files/Git/bin/bash.exe"
        
        # Call the bash script using Git Bash
        subprocess.run([git_bash_path, updated_bash_script_path], check=True)
        print("Model and JSON files have been pushed to the Docker container.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pushing files to the Docker container: {e}")

def main():
    """Load data, train model, and export artifacts."""
    # Set up MLflow with SQLite
    setup_mlflow()

    # Load and transform the data
    dataset = load_transform_data(DATA_PATH)

    print(dataset.columns)
    
    # Last column is the target variable (income level)
    X = dataset.iloc[:, :-1]  # Features (all columns except the last one)
    Y = dataset.iloc[:, -1]   # Target variable (the last column)

    validation_size = 0.20
    seed = 7

    # Split data into training and validation sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=validation_size, random_state=seed)

    # Parameters for Random Forest
    num_trees = 100
    max_features = 3

    # Get the current date
    train_date = datetime.datetime.now().isoformat()

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train a Random Forest classifier
        model = pipeline.make_pipeline(
            ensemble.RandomForestClassifier(n_estimators=num_trees, max_features=max_features, random_state=seed)
        ).fit(X_train, Y_train)

        # Log the model and parameters with MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": num_trees,
            "max_features": max_features,
            "train_date": train_date
        })

        # Predict and evaluate on the test set
        Y_pred_train = model.predict(X_train)
        Y_pred_test = model.predict(X_validation)

        # Calculate and log classification metrics
        train_accuracy = metrics.accuracy_score(Y_train, Y_pred_train)
        test_accuracy = metrics.accuracy_score(Y_validation, Y_pred_test)
        precision = metrics.precision_score(Y_validation, Y_pred_test)
        recall = metrics.recall_score(Y_validation, Y_pred_test)
        f1 = metrics.f1_score(Y_validation, Y_pred_test)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Register the model and get the version
        model_info = mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            "IncomePredictionModel"
        )
        
        model_version = model_info.version

        # Save model, metrics, params, and model info in the model/ directory
        local_metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        local_params = {
            "model_type": "RandomForestClassifier",
            "n_estimators": num_trees,
            "max_features": max_features,
            "train_date": train_date
        }

        model_info_data = {
            "run_id": run.info.run_id,
            "version": model_version,
            "train_date": train_date
        }

        pickle_file_path, model_info_file_path = save_model_artifacts(
            model, local_metrics, local_params, model_info_data, OUTPUT_DIR)

        # Push the model and JSON files to the Docker container
        push_files_to_docker(pickle_file_path, model_info_file_path)


if __name__ == "__main__":
    main()
