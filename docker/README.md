# Qwen Payslip Processor Docker Container

This directory contains the files needed to build a Docker container for the Qwen Payslip Processor. The container runs a FastAPI server that exposes endpoints for processing payslips.

## Building the Container

```bash
# From the docker directory
docker build -t qwen-payslip-processor:latest .
```

## Running the Container

```bash
# Run with default settings
docker run -d -p 27842:27842 --name qwen-model qwen-payslip-processor:latest

# Run with GPU support (if you have NVIDIA GPUs and nvidia-docker installed)
docker run -d -p 27842:27842 --gpus all --name qwen-model qwen-payslip-processor:latest
```

## API Endpoints

### GET /status

Check if the model server is running.

```bash
curl http://localhost:27842/status
```

### POST /process/pdf

Process a PDF document.

```bash
curl -X POST http://localhost:27842/process/pdf \
  -F "file=@payslip.pdf" \
  -F "pages=1,2" \
  -F "window_mode=vertical" \
  -F "selected_windows=top,bottom"
```

### POST /process/image

Process an image.

```bash
curl -X POST http://localhost:27842/process/image \
  -F "file=@payslip.jpg" \
  -F "window_mode=quadrant" \
  -F "selected_windows=top_left,bottom_right"
```

## Distributing the Container

To share the container with environments that don't have internet access:

1. Save the container as a tarball:
```bash
docker save qwen-payslip-processor:latest > qwen-payslip-processor.tar
```

2. Transfer the tarball to the target environment.

3. Load the container:
```bash
docker load < qwen-payslip-processor.tar
```

4. Run the container as described above.

## Resource Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB+ recommended
- **Disk Space**: 20GB+ for the Docker image
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance (not required) 