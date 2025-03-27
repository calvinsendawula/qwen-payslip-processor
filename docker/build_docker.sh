#!/bin/bash
# Script to build and publish the Qwen Payslip Processor Docker container

set -e  # Exit on error

# Configuration
IMAGE_NAME="calvin189/qwen-payslip-processor"
IMAGE_TAG="latest"

# Print header
echo "=== Building Qwen Payslip Processor Docker Container ==="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo

# Build the container
echo "Building Docker container... This will take several minutes."
echo "Note: The model will be downloaded on first container run, not during build."
if ! docker build -t "$IMAGE_NAME:$IMAGE_TAG" .; then
    echo
    echo "Docker build FAILED! See error messages above."
    echo "Common issues:"
    echo "- Make sure Docker is running"
    echo "- Check if you have enough disk space (need at least 5GB)"
    echo "- There may be dependency conflicts in requirements.txt"
    exit 1
fi

# Print success message
echo
echo "Docker container built successfully!"
echo "To run the container locally (model will download on first run):"
echo "  docker run -d -p 27842:27842 --name qwen-model $IMAGE_NAME:$IMAGE_TAG"
echo
echo "To save the container for air-gapped environments:"
echo "  docker save $IMAGE_NAME:$IMAGE_TAG > qwen-payslip-processor.tar"
echo
echo "To push to Docker Hub (requires login):"
echo "  docker push $IMAGE_NAME:$IMAGE_TAG" 