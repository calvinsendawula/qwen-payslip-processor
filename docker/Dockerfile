FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY server.py .

# Expose the port
EXPOSE 27842

# Pre-download the model during build
RUN python -c "from qwen_payslip_processor import QwenPayslipProcessor; QwenPayslipProcessor()"

# Create a health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 CMD curl -f http://localhost:27842/status || exit 1

# Start the server
CMD ["python", "server.py"] 