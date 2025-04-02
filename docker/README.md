# Qwen Payslip Processor Docker Container

This directory contains the files needed to build a Docker container for the Qwen Payslip Processor. The container runs a FastAPI server that exposes endpoints for processing payslips with the Qwen2.5-VL-7B-Instruct model.

## Building the Container

```bash
# From the docker directory
docker build -t qwen-payslip-processor:latest .

# Or for a specific version
docker build -t qwen-payslip-processor:v0.1.4 .
```

## Running the Container

```bash
# Run with CPU (basic setup)
docker run -d -p 27842:27842 --name qwen-processor \
  -v qwen-models:/app/models \
  -v qwen-configs:/app/configs \
  qwen-payslip-processor:latest

# Run with GPU support (recommended for performance)
docker run -d -p 27842:27842 --gpus all --name qwen-processor \
  --memory=12g --memory-swap=24g --shm-size=1g \
  -v qwen-models:/app/models \
  -v qwen-configs:/app/configs \
  qwen-payslip-processor:latest
```

The `-v` options create named volumes to persist:
- Model files (avoid re-downloading with each restart)
- Configuration files (store your saved settings)

## API Endpoints

### GET /status

Check if the model server is running and get GPU status.

```bash
curl http://localhost:27842/status
```

Example response:
```json
{
  "status": "ok",
  "model": "Qwen2.5-VL-7B-Instruct",
  "version": "0.1.4",
  "ready": true,
  "device": "cuda:0",
  "gpu": true,
  "gpu_info": "NVIDIA GeForce RTX 4060",
  "memory": {
    "total": 8.0,
    "allocated": 2.3,
    "reserved": 4.5
  }
}
```

### POST /process/pdf

Process a PDF document with comprehensive configuration options.

```bash
curl -X POST http://localhost:27842/process/pdf \
  -F "file=@payslip.pdf" \
  -F "pages=1,2" \
  -F "window_mode=horizontal" \
  -F "selected_windows=left,right" \
  -F "memory_isolation=strict" \
  -F "image_resolution_steps=600,500,400" \
  -F "pdf_dpi=350"
```

### POST /process/image

Process an image with comprehensive configuration options.

```bash
curl -X POST http://localhost:27842/process/image \
  -F "file=@payslip.jpg" \
  -F "window_mode=quadrant" \
  -F "selected_windows=top_left,bottom_right" \
  -F "image_resolution_steps=600,500,400" \
  -F "text_generation_max_new_tokens=512"
```

### Configuration Management Endpoints

#### POST /config
Save a configuration for later use.

```bash
curl -X POST http://localhost:27842/config \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_config",
    "pdf": {"dpi": 350},
    "image": {"resolution_steps": [600, 500, 400]},
    "window_mode": "horizontal"
  }'
```

#### GET /configs
List all available saved configurations.

```bash
curl http://localhost:27842/configs
```

#### GET /config/{name}
Get a specific saved configuration.

```bash
curl http://localhost:27842/config/my_config
```

#### DELETE /config/{name}
Delete a saved configuration.

```bash
curl -X DELETE http://localhost:27842/config/my_config
```

## Complete Parameter Reference

The container now supports ALL configuration parameters from the Qwen Payslip Processor package:

### Core Parameters

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Window Mode | string | How to split the document | "vertical" | `window_mode` |
| Selected Windows | string/array | Which windows to process | (all) | `selected_windows` |
| Memory Isolation | string | Memory isolation level | "auto" | `memory_isolation` |
| Force CPU | boolean | Force CPU even if GPU available | false | `force_cpu` |
| GPU Memory Fraction | float | Fraction of GPU memory to use | - | `gpu_memory_fraction` |

### PDF Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| DPI | integer | PDF rendering DPI | 600 | `pdf_dpi` |

### Image Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Resolution Steps | array/int | Image resolutions to try | [1500, 1200, 1000, 800, 600] | `image_resolution_steps` |
| Enhance Contrast | boolean | Enable contrast enhancement | true | `image_enhance_contrast` |
| Sharpen Factor | float | Sharpening intensity | 2.5 | `image_sharpen_factor` |
| Contrast Factor | float | Contrast enhancement | 1.8 | `image_contrast_factor` |
| Brightness Factor | float | Brightness adjustment | 1.1 | `image_brightness_factor` |
| OCR Language | string | OCR language | "deu" | `image_ocr_language` |
| OCR Threshold | integer | OCR confidence threshold | 90 | `image_ocr_threshold` |

### Window Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Overlap | float | Overlap between windows | 0.1 | `window_overlap` |
| Min Size | integer | Minimum window size | 100 | `window_min_size` |

### Text Generation Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Max New Tokens | integer | Max tokens to generate | 768 | `text_generation_max_new_tokens` |
| Use Beam Search | boolean | Use beam search | false | `text_generation_use_beam_search` |
| Num Beams | integer | Number of beams | 1 | `text_generation_num_beams` |
| Temperature | float | Generation temperature | 0.1 | `text_generation_temperature` |
| Top P | float | Top-p sampling | 0.95 | `text_generation_top_p` |

### Extraction Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Confidence Threshold | float | Minimum confidence | 0.7 | `extraction_confidence_threshold` |
| Fuzzy Matching | boolean | Use fuzzy matching | true | `extraction_fuzzy_matching` |

### Global Settings

| Parameter | Type | Description | Default | Form-data Key |
|-----------|------|-------------|---------|--------------|
| Global Mode | string | Default mode for all pages | "whole" | `global_mode` |
| Global Prompt | string | Default prompt for all pages | - | `global_prompt` |
| Global Selected Windows | string/array | Default windows for all pages | - | `global_selected_windows` |

### Custom Prompts

Custom prompts for each window position (use these form-data keys):
- `prompt_top`
- `prompt_bottom`
- `prompt_left`
- `prompt_right`
- `prompt_top_left`
- `prompt_top_right`
- `prompt_bottom_left`
- `prompt_bottom_right`
- `prompt_whole`

### Page-Specific Configurations

You can provide page-specific settings using the `page_configs` parameter with a JSON string.

### Full Configuration Option

For complex scenarios, you can send the entire configuration as a single JSON blob using the `full_config` parameter.

## Pre-packaged Model vs. Runtime Download

This Docker container can be built in two ways:

### 1. Pre-packaged Model (Recommended)

The recommended approach is to build the Docker image with the model files already included. This results in a larger image (approximately 14GB) but eliminates the need to download the model at runtime, which is more reliable and allows the container to work in environments with limited internet connectivity.

#### Advantages:
- Immediate startup - no need to wait for model downloads
- Works in air-gapped or limited connectivity environments
- More reliable and predictable performance
- No download timeouts or network issues

#### Disadvantages:
- Larger Docker image size (~14GB vs ~2.9GB)
- Requires more disk space during build and deployment

### 2. Runtime Download (Original Approach)

The original approach downloads the model when the container first starts. This results in a smaller image but requires a stable internet connection and can lead to timeouts or failures in environments with limited connectivity.

#### Advantages:
- Smaller Docker image size (~2.9GB)
- Less disk space required during build

#### Disadvantages:
- Requires downloading ~10GB+ of model files on first run
- Needs stable, high-bandwidth internet connection
- Can time out or fail in limited connectivity environments
- Container not ready until download completes (can take 30+ minutes)

### How to Build with Pre-packaged Model

To build the Docker image with the model pre-packaged:

1. First, ensure you have the model downloaded locally in a `model_cache` directory:
   ```bash
   # Directory structure should look like:
   model_cache/
   ├── models--Qwen--Qwen2.5-VL-7B-Instruct/
   ├── model/
   └── processor/
   ```

2. Run the build script which will copy these files into the Docker image:
   ```bash
   # For Linux/macOS
   ./build_docker.sh
   
   # For Windows
   .\build_docker.ps1
   ```

3. The script will build and tag the Docker image with the model included

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
- **RAM**: 16GB+ recommended (12GB minimum)
- **Disk Space**: 20GB+ for the Docker image and models
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance (not required)

## Memory Optimization Tips

If you encounter out-of-memory errors:

1. Reduce the image resolution:
   ```bash
   -F "image_resolution_steps=600,500,400" -F "pdf_dpi=350"
   ```

2. Reduce token generation:
   ```bash
   -F "text_generation_max_new_tokens=512"
   ```

3. Use memory isolation:
   ```bash
   -F "memory_isolation=strict"
   ```

4. Set GPU memory fraction (for GPU users):
   ```bash
   -F "gpu_memory_fraction=0.7"
   ``` 