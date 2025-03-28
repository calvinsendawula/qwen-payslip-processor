FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies (first installing PyTorch with CUDA support)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY server.py download_model.py ./

# Create model cache directory with proper permissions
RUN mkdir -p /app/models && chmod -R 777 /app/models

# Set environment variables for model caching
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Define volume for model persistence
VOLUME /app/models

# Expose the port
EXPOSE 27842

# Create a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:27842/status || exit 1

# Start the server - model will be downloaded on first use if needed
CMD ["python", "server.py"]