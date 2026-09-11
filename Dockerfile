# Multi-stage Dockerfile for RetinaSight (SIH 2026, PS 26038)
# Stage 1: Build React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npx vite build

# Stage 2: Python Inference Backend
FROM python:3.10-slim
WORKDIR /app

# Install system libraries for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and models
COPY preprocessing.py gradcam.py train_dr_classifier.py main.py ./
COPY retinasight_resnet50.onnx retinasight_resnet50.pth ./
COPY matlab/ ./matlab/
COPY docs/ ./docs/
COPY outputs/ ./outputs/

# Copy built frontend assets
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY frontend/public ./frontend/public

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
