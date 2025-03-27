# PowerShell script to build and publish the Qwen Payslip Processor Docker container

# Configuration
$IMAGE_NAME = "calvin189/qwen-payslip-processor"
$IMAGE_TAG = "latest"

# Print header
Write-Host "=== Building Qwen Payslip Processor Docker Container ===" -ForegroundColor Green
Write-Host "Image: $IMAGE_NAME`:$IMAGE_TAG"
Write-Host ""

# Build the container
Write-Host "Building Docker container..." -ForegroundColor Yellow
docker build -t "$IMAGE_NAME`:$IMAGE_TAG" .

# Print success message
Write-Host ""
Write-Host "Docker container built successfully!" -ForegroundColor Green
Write-Host "To run the container locally:"
Write-Host "  docker run -d -p 27842:27842 --name qwen-model $IMAGE_NAME`:$IMAGE_TAG"
Write-Host ""
Write-Host "To save the container for air-gapped environments:"
Write-Host "  docker save $IMAGE_NAME`:$IMAGE_TAG -o qwen-payslip-processor.tar"
Write-Host ""
Write-Host "To push to Docker Hub (requires login):"
Write-Host "  docker push $IMAGE_NAME`:$IMAGE_TAG" 