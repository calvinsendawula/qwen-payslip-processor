# PowerShell script to build and push the Docker image for Qwen Payslip Processor

# Configuration
$VERSION = "0.1.4"
$IMAGE_NAME = "qwen-payslip-processor"
$REPOSITORY = "calvin189"  # Docker Hub username/organization
$TAG = "${REPOSITORY}/${IMAGE_NAME}:latest"
$VERSION_TAG = "${REPOSITORY}/${IMAGE_NAME}:${VERSION}"

Write-Host "Building Docker image for Qwen Payslip Processor with pre-packaged model..."
Write-Host "Note: This will create a large Docker image (~14GB) since it includes the model files."
Write-Host "Make sure you have enough disk space."

# Check if model cache exists at expected location
$MODEL_CACHE_PATH = "..\..\model_cache"
if (-not (Test-Path $MODEL_CACHE_PATH)) {
    Write-Host "Error: Model cache directory not found at $MODEL_CACHE_PATH" -ForegroundColor Red
    Write-Host "Please download the model files first or adjust the path in this script." -ForegroundColor Red
    exit 1
}

# Check for key model directories
if (-not (Test-Path "$MODEL_CACHE_PATH\models--Qwen--Qwen2.5-VL-7B-Instruct") -or -not (Test-Path "$MODEL_CACHE_PATH\model")) {
    Write-Host "Error: Required model files not found in $MODEL_CACHE_PATH" -ForegroundColor Red
    Write-Host "Please download the complete model before building the Docker image." -ForegroundColor Red
    exit 1
}

# Copy model files to current directory for build
Write-Host "Copying model files for Docker build..."
New-Item -ItemType Directory -Force -Path .\model_cache | Out-Null
Copy-Item -Path "$MODEL_CACHE_PATH\models--Qwen--Qwen2.5-VL-7B-Instruct" -Destination .\model_cache\ -Recurse -Force
Copy-Item -Path "$MODEL_CACHE_PATH\model" -Destination .\model_cache\ -Recurse -Force
Copy-Item -Path "$MODEL_CACHE_PATH\processor" -Destination .\model_cache\ -Recurse -Force
if (Test-Path "$MODEL_CACHE_PATH\QWEN_MODEL_READY") {
    Copy-Item -Path "$MODEL_CACHE_PATH\QWEN_MODEL_READY" -Destination .\model_cache\ -Force
}

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t "$TAG" -t "$VERSION_TAG" .

# Clean up temporary files
Write-Host "Cleaning up temporary model files..."
Remove-Item -Path .\model_cache -Recurse -Force

Write-Host "Image built successfully: $TAG and $VERSION_TAG"
Write-Host "To push to Docker Hub:"
Write-Host "docker push $TAG"
Write-Host "docker push $VERSION_TAG"

# Add information about running with different settings
Write-Host "How to run the container:" -ForegroundColor Green
Write-Host "1. Basic run (uses CPU by default):" -ForegroundColor Yellow
Write-Host "   docker run -d -p 27842:27842 --name qwen-processor $TAG"
Write-Host ""
Write-Host "2. Run with GPU support (if available):" -ForegroundColor Yellow
Write-Host "   docker run -d -p 27842:27842 --gpus all -e FORCE_CPU=false --name qwen-processor $TAG"
Write-Host ""
Write-Host "3. Run with customized default settings:" -ForegroundColor Yellow
Write-Host "   docker run -d -p 27842:27842 --name qwen-processor \\"
Write-Host "     -e FORCE_CPU=false \\"
Write-Host "     -v your-configs:/app/configs \\"
Write-Host "     $TAG"
Write-Host ""
Write-Host "NOTE: Even when running with default CPU mode, you can still override this at runtime" -ForegroundColor Cyan
Write-Host "by passing force_cpu=false in the API requests." -ForegroundColor Cyan

# Optionally push to Docker Hub
$PUSH_CHOICE = Read-Host "Do you want to push the image to Docker Hub? (y/n)"

if ($PUSH_CHOICE -eq "y" -or $PUSH_CHOICE -eq "Y") {
    Write-Host "Pushing images to Docker Hub..."
    docker push "$TAG"
    docker push "$VERSION_TAG"
    Write-Host "Images pushed successfully."
} else {
    Write-Host "Images not pushed. You can push them later with:"
    Write-Host "docker push $TAG"
    Write-Host "docker push $VERSION_TAG"
}

Write-Host "Done!" 