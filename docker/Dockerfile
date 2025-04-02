FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    # Add yaml support for config management
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch with CUDA support (falls back to CPU when GPU not available)
# Using index-url to specify CUDA-enabled PyTorch
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for configuration management
RUN pip install --no-cache-dir pyyaml==6.0.2

# Copy the server code
COPY server.py download_model.py ./

# Create model cache directory with proper permissions
# This directory will store the downloaded model files for persistent use
RUN mkdir -p /app/models && chmod -R 777 /app/models

# Create config directory for storing configurations
RUN mkdir -p /app/configs && chmod -R 777 /app/configs

# Set environment variables for model caching
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models

# Optimize transformer behavior
ENV TRANSFORMERS_VERBOSITY=error

# Define volumes for model persistence and configuration storage
VOLUME /app/models
VOLUME /app/configs

# Expose the port
EXPOSE 27842

# Create a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:27842/status || exit 1

# Start the server - model will be downloaded on first use if needed
CMD ["python", "server.py"]