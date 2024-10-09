FROM python:3.9-slim

WORKDIR /app

# Copy the app directory's contents into the /app directory in the container
COPY ./app/ /app/

# Declare a volume to store model and model info
VOLUME /app/model

# Copy the contents of sound_realty/app/model/ into the volume
COPY ./train/model/ /app/model/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]