# Medical AI Assistant — Dockerfile for Fly.io deployment
FROM python:3.12-slim

# System deps needed by some ML/PDF libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (app.py, knowledge_base/, etc.)
COPY . .

# Gradio needs to bind to 0.0.0.0 and the port Fly expects
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
