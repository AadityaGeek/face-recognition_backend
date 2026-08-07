# Use an official Python image (TensorFlow supports up to 3.11/3.12)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for deepface, opencv, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Pre-download DeepFace Facenet weights during docker build for fast initial requests
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

# Copy project files
COPY . .

# Expose port (Railway will set $PORT dynamically)
EXPOSE 8000

# Start command (using shell form to dynamically bind to Railway's $PORT env var)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
