FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for torch
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model during build (so no internet needed at runtime)
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='google/vit-base-patch16-224')"

COPY app/ .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "main:app"]