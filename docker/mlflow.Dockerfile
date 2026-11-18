FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir --default-timeout=180 --retries=5 \
    "SQLAlchemy==2.0.35" \
    mlflow==2.17.2

EXPOSE 5000
