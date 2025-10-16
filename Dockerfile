# Base image with Python
FROM python:3.10-slim

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR $AIRFLOW_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DAG file
COPY airflow_ml_pipeline.py dags/

# Initialize Airflow
RUN airflow db init

# Default command (optional for CI)
CMD ["airflow", "tasks", "test", "iris_ml_pipeline", "evaluate_model", "2023-01-01"]
