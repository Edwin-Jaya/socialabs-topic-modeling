# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  make \
  python3-dev \
  libpython3-dev \
  && rm -rf /var/lib/apt/lists/*
  
# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Flask application will run on
EXPOSE 8000

ENV APP_ENV=local
ENV APP_DEBUG=True
ENV APP_PORT=8000
ENV DB_URL=mongodb+srv://application:skripsi@socialabs.pjkgs8t.mongodb.net/
ENV DB_NAME=tweets
ENV AZURE_OPENAI_KEY=6YwpSUX7CKAhTAWRFieW7zj7Q3OoXJNtjGOCsvZsFnTN7g7MyX7SJQQJ99BGACYeBjFXJ3w3AAABACOGG9Qs
ENV AZURE_OPENAI_ENDPOINT=https://research-etm.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview
ENV AZURE_OPENAI_MODEL_VERSION="2025-01-01-preview"
ENV AZURE_OPENAI_DEPLOYMENT=o4-mini
ENV AZURE_OPENAI_MODEL_NAME=o4-mini
ENV RABBITMQ_URL=amqp://admin:admin123@70.153.61.68:5672/socialabs

# AZURE_OPENAI_KEY="6YwpSUX7CKAhTAWRFieW7zj7Q3OoXJNtjGOCsvZsFnTN7g7MyX7SJQQJ99BGACYeBjFXJ3w3AAABACOGG9Qs"
# AZURE_OPENAI_ENDPOINT="https://research-etm.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
# AZURE_OPENAI_MODEL_VERSION="2025-01-01-preview"
# AZURE_OPENAI_DEPLOYMENT="o4-mini"
# Copy the Gunicorn configuration file
COPY gunicorn_config.py gunicorn_config.py

# Copy the start_services.py script and make it executable
COPY start_services.py start_services.py
RUN chmod +x start_services.py

# Command to run the start_services.py script
CMD ["python", "start_services.py"]
