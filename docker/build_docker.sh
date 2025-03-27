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
echo "Building Docker container..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

# Print success message
echo
echo "Docker container built successfully!"
echo "To run the container locally:"
echo "  docker run -d -p 27842:27842 --name qwen-model $IMAGE_NAME:$IMAGE_TAG"
echo
echo "To save the container for air-gapped environments:"
echo "  docker save $IMAGE_NAME:$IMAGE_TAG > qwen-payslip-processor.tar"
echo
echo "To push to Docker Hub (requires login):"
echo "  docker push $IMAGE_NAME:$IMAGE_TAG" 