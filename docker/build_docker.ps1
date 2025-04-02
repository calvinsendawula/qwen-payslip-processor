# PowerShell script to build and publish the Qwen Payslip Processor Docker container

# Configuration
$IMAGE_NAME="calvin189/qwen-payslip-processor"
$IMAGE_TAG="latest"
$VERSION="v0.1.4"

# Print header
Write-Host "=== Building Qwen Payslip Processor Docker Container ===" -ForegroundColor Cyan
Write-Host "Image: $IMAGE_NAME`:$IMAGE_TAG"
Write-Host ""

# Build the container
Write-Host "Building Docker container... This will take several minutes." -ForegroundColor Yellow
Write-Host "Note: The model will be downloaded on first container run, not during build."

try {
    docker build -t "$IMAGE_NAME`:$IMAGE_TAG" .
    
    # Also tag with version number
    docker tag "$IMAGE_NAME`:$IMAGE_TAG" "$IMAGE_NAME`:$VERSION"
}
catch {
    Write-Host ""
    Write-Host "Docker build FAILED! See error messages above." -ForegroundColor Red
    Write-Host "Common issues:" -ForegroundColor Red
    Write-Host "- Make sure Docker is running" -ForegroundColor Red
    Write-Host "- Check if you have enough disk space (need at least 5GB)" -ForegroundColor Red
    Write-Host "- There may be dependency conflicts in requirements.txt" -ForegroundColor Red
    exit 1
}

# Print success message
Write-Host ""
Write-Host "Docker container built successfully!" -ForegroundColor Green

Write-Host ""
Write-Host "To run the container with CPU (basic setup):" -ForegroundColor Green
Write-Host "  docker run -d -p 27842:27842 --name qwen-processor ``" -ForegroundColor Yellow
Write-Host "    -v qwen-models:/app/models ``" -ForegroundColor Yellow
Write-Host "    -v qwen-configs:/app/configs ``" -ForegroundColor Yellow
Write-Host "    $IMAGE_NAME`:$IMAGE_TAG" -ForegroundColor Yellow

Write-Host ""
Write-Host "To run with GPU support (recommended for performance):" -ForegroundColor Green
Write-Host "  docker run -d -p 27842:27842 --gpus all --name qwen-processor ``" -ForegroundColor Yellow
Write-Host "    --memory=12g --memory-swap=24g --shm-size=1g ``" -ForegroundColor Yellow
Write-Host "    -v qwen-models:/app/models ``" -ForegroundColor Yellow
Write-Host "    -v qwen-configs:/app/configs ``" -ForegroundColor Yellow
Write-Host "    $IMAGE_NAME`:$IMAGE_TAG" -ForegroundColor Yellow

Write-Host ""
Write-Host "Memory-optimized setup for systems with limited resources:" -ForegroundColor Green
Write-Host "  docker run -d -p 27842:27842 --name qwen-processor ``" -ForegroundColor Yellow
Write-Host "    -v qwen-models:/app/models ``" -ForegroundColor Yellow
Write-Host "    -v qwen-configs:/app/configs ``" -ForegroundColor Yellow
Write-Host "    -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 ``" -ForegroundColor Yellow
Write-Host "    $IMAGE_NAME`:$IMAGE_TAG" -ForegroundColor Yellow

Write-Host ""
Write-Host "To save the container for air-gapped environments:" -ForegroundColor Green
Write-Host "  docker save $IMAGE_NAME`:$IMAGE_TAG > qwen-payslip-processor.tar" -ForegroundColor Yellow

Write-Host ""
Write-Host "To push to Docker Hub (requires login):" -ForegroundColor Green
Write-Host "  docker login" -ForegroundColor Yellow
Write-Host "  docker push $IMAGE_NAME`:$IMAGE_TAG" -ForegroundColor Yellow
Write-Host "  docker push $IMAGE_NAME`:$VERSION" -ForegroundColor Yellow 