# Use Python slim base to reduce image size
FROM python:3.11-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies (needed for opencv & mediapipe)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app
COPY . .

# Expose the port Cloud Run will use
EXPOSE 8080

# Start FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
