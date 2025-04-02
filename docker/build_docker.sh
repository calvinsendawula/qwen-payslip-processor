#!/bin/bash
# Script to build and push the Docker image for Qwen Payslip Processor

set -e  # Exit on error

VERSION="0.1.4"
IMAGE_NAME="qwen-payslip-processor"
REPOSITORY="calvin189"  # Docker Hub username/organization
TAG="${REPOSITORY}/${IMAGE_NAME}:latest"
VERSION_TAG="${REPOSITORY}/${IMAGE_NAME}:${VERSION}"

echo "Building Docker image for Qwen Payslip Processor with pre-packaged model..."
echo "Note: This will create a large Docker image (~14GB) since it includes the model files."
echo "Make sure you have enough disk space."

# Check if model cache exists at expected location
MODEL_CACHE_PATH="../../model_cache"
if [ ! -d "$MODEL_CACHE_PATH" ]; then
    echo "Error: Model cache directory not found at $MODEL_CACHE_PATH"
    echo "Please download the model files first or adjust the path in this script."
    exit 1
fi

# Check for key model directories
if [ ! -d "$MODEL_CACHE_PATH/models--Qwen--Qwen2.5-VL-7B-Instruct" ] || [ ! -d "$MODEL_CACHE_PATH/model" ]; then
    echo "Error: Required model files not found in $MODEL_CACHE_PATH"
    echo "Please download the complete model before building the Docker image."
    exit 1
fi

# Copy model files to current directory for build
echo "Copying model files for Docker build..."
mkdir -p ./model_cache
cp -r $MODEL_CACHE_PATH/models--Qwen--Qwen2.5-VL-7B-Instruct ./model_cache/
cp -r $MODEL_CACHE_PATH/model ./model_cache/
cp -r $MODEL_CACHE_PATH/processor ./model_cache/
cp -r $MODEL_CACHE_PATH/QWEN_MODEL_READY ./model_cache/ 2>/dev/null || :

# Build the Docker image
echo "Building Docker image..."
docker build -t "$TAG" -t "$VERSION_TAG" .

# Clean up temporary files
echo "Cleaning up temporary model files..."
rm -rf ./model_cache

echo "Image built successfully: $TAG and $VERSION_TAG"
echo "To push to Docker Hub:"
echo "docker push $TAG"
echo "docker push $VERSION_TAG"

# Add information about running with different settings
echo ""
echo -e "\033[1;32mHow to run the container:\033[0m"
echo -e "\033[1;33m1. Basic run (uses CPU by default):\033[0m"
echo "   docker run -d -p 27842:27842 --name qwen-processor $TAG"
echo ""
echo -e "\033[1;33m2. Run with GPU support (if available):\033[0m"
echo "   docker run -d -p 27842:27842 --gpus all -e FORCE_CPU=false --name qwen-processor $TAG"
echo ""
echo -e "\033[1;33m3. Run with customized default settings:\033[0m"
echo "   docker run -d -p 27842:27842 --name qwen-processor \\"
echo "     -e FORCE_CPU=false \\"
echo "     -v your-configs:/app/configs \\"
echo "     $TAG"
echo ""
echo -e "\033[1;36mNOTE: Even when running with default CPU mode, you can still override this at runtime\033[0m"
echo -e "\033[1;36mby passing force_cpu=false in the API requests.\033[0m"

# Optionally push to Docker Hub
read -p "Do you want to push the image to Docker Hub? (y/n): " PUSH_CHOICE

if [[ "$PUSH_CHOICE" =~ ^[Yy]$ ]]; then
    echo "Pushing images to Docker Hub..."
    docker push "$TAG"
    docker push "$VERSION_TAG"
    echo "Images pushed successfully."
else
    echo "Images not pushed. You can push them later with:"
    echo "docker push $TAG"
    echo "docker push $VERSION_TAG"
fi

echo "Done!" 