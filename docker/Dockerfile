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
COPY server.py ./

# Create directory structure
RUN mkdir -p /app/models /app/configs && chmod -R 777 /app/models /app/configs

# Set environment variables for model caching
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models
ENV TRANSFORMERS_OFFLINE=1

# Set CPU as default but allow override when container is run
ENV FORCE_CPU=true

# Copy pre-downloaded model files
# This replaces the runtime downloading
COPY model_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/ /app/models/models--Qwen--Qwen2.5-VL-7B-Instruct/
COPY model_cache/model/ /app/models/model/
COPY model_cache/processor/ /app/models/processor/
COPY model_cache/QWEN_MODEL_READY /app/models/

# Optimize transformer behavior
ENV TRANSFORMERS_VERBOSITY=error

# Define volume for configuration storage
VOLUME /app/configs

# Expose the port
EXPOSE 27842

# Create a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:27842/status || exit 1

# Start the server - model will be loaded from pre-packaged files
CMD ["python", "server.py"]