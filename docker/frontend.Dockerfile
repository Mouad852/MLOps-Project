FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=180 --retries=5 -r requirements.txt

COPY frontend/ .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
