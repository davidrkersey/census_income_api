#!/bin/bash

# Variables
CONTAINER_NAME="__CONTAINER_ID__" 
MODEL_PATH="./train/model/model.pkl"
MODEL_INFO_PATH="./train/model/model_info.json"
DOCKER_MODEL_PATH="/app/model/model.pkl"
DOCKER_MODEL_INFO_PATH="/app/model/model_info.json"

# Copy the files into the Docker container
docker cp $MODEL_PATH $CONTAINER_NAME:$DOCKER_MODEL_PATH
docker cp $MODEL_INFO_PATH $CONTAINER_NAME:$DOCKER_MODEL_INFO_PATH

echo "Model and model info files have been copied to the Docker container."