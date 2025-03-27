# PowerShell script to build and publish the Qwen Payslip Processor Docker container

# Configuration
$IMAGE_NAME = "calvin189/qwen-payslip-processor"
$IMAGE_TAG = "latest"

# Print header
Write-Host "=== Building Qwen Payslip Processor Docker Container ===" -ForegroundColor Green
Write-Host "Image: $IMAGE_NAME`:$IMAGE_TAG"
Write-Host ""

# Build the container
Write-Host "Building Docker container... This will take several minutes." -ForegroundColor Yellow
Write-Host "Note: The model will be downloaded on first container run, not during build." -ForegroundColor Yellow
docker build -t "$IMAGE_NAME`:$IMAGE_TAG" .

# Check if build was successful
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Docker build FAILED! See error messages above." -ForegroundColor Red
    Write-Host "Common issues:"
    Write-Host "- Make sure Docker Desktop is running" -ForegroundColor Yellow
    Write-Host "- Check if you have enough disk space (need at least 5GB)" -ForegroundColor Yellow
    Write-Host "- There may be dependency conflicts in requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Print success message
Write-Host ""
Write-Host "Docker container built successfully!" -ForegroundColor Green
Write-Host "To run the container locally (model will download on first run):"
Write-Host "  docker run -d -p 27842:27842 --name qwen-model $IMAGE_NAME`:$IMAGE_TAG"
Write-Host ""
Write-Host "To save the container for air-gapped environments:"
Write-Host "  docker save $IMAGE_NAME`:$IMAGE_TAG -o qwen-payslip-processor.tar"
Write-Host ""
Write-Host "To push to Docker Hub (requires login):"
Write-Host "  docker push $IMAGE_NAME`:$IMAGE_TAG" 