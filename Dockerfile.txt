# Use Python 3.9 slim image for Railway
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install essential system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads static/detected static/reports static/recordings templates

# Set environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway will set this automatically)
EXPOSE $PORT

# Run the application
CMD ["python", "app.py"]
