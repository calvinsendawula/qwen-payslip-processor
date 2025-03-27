# Qwen Payslip Processor

A Python package for processing German payslips using the Qwen2.5-VL-7B vision-language model.

## Features

- Extracts key information from German payslips in PDF or image format
- Automatically downloads and caches the Qwen model if not present
- Highly customizable processing options:
  - Page selection for multi-page PDFs
  - Multiple window division modes (whole, vertical split, horizontal split, quadrants)
  - Selective processing of specific windows/regions
  - Adjustable image resolution, enhancement, and preprocessing parameters
  - Customizable prompts for specific document regions
- Optimized with CUDA GPU acceleration (falls back to CPU when unavailable)

## Installation

```bash
pip install qwen-payslip-processor
```

## Requirements

- Python 3.11.5 or higher
- CUDA-compatible GPU with at least 8GB+ of VRAM for optimal performance
- PyTorch with CUDA support for GPU acceleration
- 16GB+ RAM (with at least 8GB+ of VRAM) regardless of CPU or GPU usage
- 20GB+ free disk space for the model cache

## Basic Usage

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Initialize with default settings
processor = QwenPayslipProcessor()

# Process a PDF
with open("path/to/payslip.pdf", "rb") as f:
    pdf_data = f.read()
    
results = processor.process_pdf(pdf_data)
print(results)

# Process an image
with open("path/to/payslip.jpg", "rb") as f:
    image_data = f.read()
    
results = processor.process_image(image_data)
print(results)
```

## Simple Configuration Example

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Create processor with custom settings
processor = QwenPayslipProcessor(
    window_mode="vertical",
    selected_windows=["top"],
    force_cpu=False,
    custom_prompts={
        "top": "Find employee name in this section..."
    },
    config={
        "pdf": {
            "dpi": 300
        },
        "image": {
            "resolution_steps": [1200, 1000, 800],
            "enhance_contrast": True,
            "sharpen_factor": 2.0,
            "contrast_factor": 1.5
        },
        "text_generation": {
            "max_new_tokens": 512,
            "temperature": 0.1
        }
    }
)

# Process a PDF
with open("path/to/payslip.pdf", "rb") as f:
    pdf_data = f.read()
    
results = processor.process_pdf(pdf_data, pages=1)  # Process only page 1
print(results)
```

## Complete Parameters Reference

### Parameter Reference Table

| Category | Parameter | Description | Default | Valid Range/Options |
|----------|-----------|-------------|---------|-------------|
| **Main Parameters** | | | | |
| | `window_mode` | Division mode for processing | `"vertical"` | `"whole"`, `"vertical"`, `"horizontal"`, `"quadrant"` |
| | `selected_windows` | Which windows to process | `None` (all) | • `"whole"` mode: parameter ignored<br>• `"vertical"` mode: `"top"`, `"bottom"`, or `["top", "bottom"]`<br>• `"horizontal"` mode: `"left"`, `"right"`, or `["left", "right"]`<br>• `"quadrant"` mode: any combination of `"top_left"`, `"top_right"`, `"bottom_left"`, `"bottom_right"` |
| | `force_cpu` | Force CPU even if GPU available | `False` | `True`, `False` |
| | `custom_prompts` | Custom instructions for windows | `{}` | Dict with keys matching window names (e.g., `{"top": "prompt text", "bottom": "prompt text"}`) |
| **PDF** | | | | |
| | `config["pdf"]["dpi"]` | PDF rendering DPI | `600` | `150`-`600` |
| **Image** | | | | |
| | `config["image"]["resolution_steps"]` | Image resolutions to try | `[1500, 1200, 1000, 800, 600]` | List of integers (pixel sizes) or single integer |
| | `config["image"]["enhance_contrast"]` | Enable contrast enhancement | `True` | `True`, `False` |
| | `config["image"]["sharpen_factor"]` | Sharpening intensity | `2.5` | `1.0`-`3.0` |
| | `config["image"]["contrast_factor"]` | Contrast enhancement | `1.8` | `1.0`-`2.0` |
| | `config["image"]["brightness_factor"]` | Brightness adjustment | `1.1` | `0.8`-`1.2` |
| | `config["image"]["ocr_language"]` | OCR language | `"deu"` | ISO language codes (e.g., `"deu"` for German) |
| | `config["image"]["ocr_threshold"]` | OCR confidence threshold | `90` | `0`-`100` |
| **Window** | | | | |
| | `config["window"]["overlap"]` | Overlap between windows | `0.1` | `0.0`-`0.5` (proportion of window size) |
| | `config["window"]["min_size"]` | Minimum window size | `100` | `50`+ pixels |
| **Text Generation** | | | | |
| | `config["text_generation"]["max_new_tokens"]` | Max tokens to generate | `768` | `128`-`1024` |
| | `config["text_generation"]["use_beam_search"]` | Use beam search | `False` | `True`, `False` |
| | `config["text_generation"]["num_beams"]` | Number of beams | `1` | `1`-`5` (only used if `use_beam_search` is `True`) |
| | `config["text_generation"]["temperature"]` | Generation temperature | `0.1` | `0.1`-`1.0` (lower = more deterministic) |
| | `config["text_generation"]["top_p"]` | Top-p sampling | `0.95` | `0.5`-`1.0` |
| **Extraction** | | | | |
| | `config["extraction"]["confidence_threshold"]` | Minimum confidence | `0.7` | `0.0`-`1.0` |
| | `config["extraction"]["fuzzy_matching"]` | Use fuzzy matching | `True` | `True`, `False` |

Here's a complete example showing all available parameters with their default values:

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Create processor with all parameters explicitly set
processor = QwenPayslipProcessor(
    # Window division mode - exactly one of: "whole", "vertical", "horizontal", "quadrant"
    window_mode="vertical",
    
    # Only process specific windows - must match the window_mode you selected
    # For "vertical" mode, can use: ["top", "bottom"] or just "top" or just "bottom"
    # For "horizontal" mode, can use: ["left", "right"] or just "left" or just "right"
    # For "quadrant" mode, can use any combination of: ["top_left", "top_right", "bottom_left", "bottom_right"]
    # For "whole" mode: this parameter is ignored
    selected_windows=["top"],  
    
    # Force CPU processing even if GPU is available
    force_cpu=False,
    
    # Custom prompts that MUST match your window_mode and selected_windows
    # Keys MUST be exactly the same as the window names (e.g., "top", "bottom_right")
    custom_prompts={
        "top": "Find employee name in this section...",
        "bottom": "Find gross and net amounts in this section..."
    },
    
    # Configuration dictionary with all parameters
    config={
        "pdf": {
            "dpi": 600,               # PDF rendering DPI (Range: 150-600)
        },
        "image": {
            # Image resolution options - can use multiple resolutions
            "resolution_steps": [1500, 1200, 1000, 800, 600],
            # OR use a single resolution
            # "resolution_steps": 1200,  # Can also provide a single value
            
            # Enhancement parameters
            "enhance_contrast": True,  # Enable/disable contrast enhancement
            "sharpen_factor": 2.5,     # Sharpening intensity (Range: 1.0-3.0)
            "contrast_factor": 1.8,    # Contrast enhancement (Range: 1.0-2.0)
            "brightness_factor": 1.1,  # Brightness adjustment (Range: 0.8-1.2)
            
            # OCR-specific settings
            "ocr_language": "deu",     # German language for OCR
            "ocr_threshold": 90,       # Confidence threshold for OCR (%)
        },
        "window": {
            "overlap": 0.1,           # Overlap between windows (Range: 0.0-0.5)
            "min_size": 100,          # Minimum size in pixels for a window
        },
        "text_generation": {
            "max_new_tokens": 768,    # Max tokens to generate (Range: 128-1024)
            "use_beam_search": False, # Use beam search instead of greedy decoding
            "num_beams": 1,           # Number of beams for beam search
            "temperature": 0.1,       # Temperature for generation (Range: 0.1-1.0)
            "top_p": 0.95,            # Top-p sampling parameter (Range: 0.5-1.0)
        },
        "extraction": {
            "confidence_threshold": 0.7,  # Minimum confidence for extracted values
            "fuzzy_matching": True        # Use fuzzy matching for field names
        }
    }
)

# Process a PDF with page selection
with open("path/to/payslip.pdf", "rb") as f:
    pdf_data = f.read()
    
# Process multiple specific pages
results = processor.process_pdf(
    pdf_data,
    pages=[1, 3, 5]  # Process pages 1, 3, and 5
)

# OR process a single page
results = processor.process_pdf(
    pdf_data,
    pages=2  # Process only page 2 - can also provide a single value
)
```

The sections below explain each parameter group in more detail.

## PDF Processing Options

```python
# Process multiple specific pages
processor = QwenPayslipProcessor()
results = processor.process_pdf(
    pdf_bytes,
    pages=[1, 3]  # Process pages 1 and 3
)

# Process a single page
results = processor.process_pdf(
    pdf_bytes,
    pages=2  # Process only page 2
)

# Set custom DPI for PDF conversion
processor = QwenPayslipProcessor(
    config={
        "pdf": {
            "dpi": 300  # Range: 150-600, Default: 600
        }
    }
)
```

## Window Division Modes

You can only use ONE of these window division modes at a time:

```python
# MODE 1: Process each page as a whole (no division)
processor = QwenPayslipProcessor(window_mode="whole")

# MODE 2: Split each page into top and bottom halves
processor = QwenPayslipProcessor(window_mode="vertical")

# MODE 3: Split each page into left and right halves
processor = QwenPayslipProcessor(window_mode="horizontal")

# MODE 4: Split each page into four quadrants
processor = QwenPayslipProcessor(window_mode="quadrant")
```

## Selective Window Processing

Here are ALL the valid combinations of window_mode and selected_windows:

```python
# For VERTICAL mode (2 possible windows)
processor = QwenPayslipProcessor(
    window_mode="vertical",
    # Process both windows
    selected_windows=["top", "bottom"]
)

# Process just top half in vertical mode
processor = QwenPayslipProcessor(
    window_mode="vertical",
    selected_windows="top"  # Just a string is fine for single window
)

# Process just bottom half in vertical mode
processor = QwenPayslipProcessor(
    window_mode="vertical",
    selected_windows="bottom"
)

# For HORIZONTAL mode (2 possible windows)
processor = QwenPayslipProcessor(
    window_mode="horizontal",
    # Process both halves
    selected_windows=["left", "right"]
)

# Process just left side in horizontal mode
processor = QwenPayslipProcessor(
    window_mode="horizontal",
    selected_windows="left"
)

# Process just right side in horizontal mode
processor = QwenPayslipProcessor(
    window_mode="horizontal",
    selected_windows="right"
)

# For QUADRANT mode (4 possible windows)
# Process all four quadrants
processor = QwenPayslipProcessor(
    window_mode="quadrant",
    selected_windows=["top_left", "top_right", "bottom_left", "bottom_right"]
)

# Process any combination of quadrants - examples:
# Just one quadrant
processor = QwenPayslipProcessor(
    window_mode="quadrant",
    selected_windows="bottom_right"  # Process only bottom-right quadrant
)

# Two specific quadrants
processor = QwenPayslipProcessor(
    window_mode="quadrant",
    selected_windows=["top_left", "bottom_right"]  # Diagonal quadrants
)

# Three specific quadrants (exclude one)
processor = QwenPayslipProcessor(
    window_mode="quadrant",
    selected_windows=["top_left", "top_right", "bottom_left"]  # All except bottom-right
)

# For WHOLE mode (1 possible window)
processor = QwenPayslipProcessor(
    window_mode="whole",
    # selected_windows parameter is ignored in "whole" mode
)
```

## Custom Prompts

Define custom prompts that MUST match your selected window names:

```python
# For VERTICAL mode - must use keys "top" and/or "bottom"
processor = QwenPayslipProcessor(
    window_mode="vertical",
    selected_windows=["top", "bottom"],
    custom_prompts={
        "top": """
        Find the employee name in the top section.
        Look for text following "Herrn/Frau".
        Return JSON: {"found_in_top": {"employee_name": "NAME", "gross_amount": "0", "net_amount": "0"}}
        """,
        
        "bottom": """
        Find the gross and net amounts in the bottom section.
        Look for "Gesamt-Brutto" and "Auszahlungsbetrag".
        Return JSON: {"found_in_bottom": {"employee_name": "unknown", "gross_amount": "1234,56", "net_amount": "789,10"}}
        """
    }
)

# For HORIZONTAL mode - must use keys "left" and/or "right"
processor = QwenPayslipProcessor(
    window_mode="horizontal",
    selected_windows=["left"],
    custom_prompts={
        "left": """
        Find the employee name in the left section.
        Return JSON: {"found_in_left": {"employee_name": "NAME", "gross_amount": "0", "net_amount": "0"}}
        """
    }
)

# For QUADRANT mode - must use keys matching your selected quadrants
processor = QwenPayslipProcessor(
    window_mode="quadrant",
    selected_windows=["top_left", "bottom_right"],
    custom_prompts={
        "top_left": """
        Find the employee name in the top-left section.
        Return JSON: {"found_in_top_left": {"employee_name": "NAME", "gross_amount": "0", "net_amount": "0"}}
        """,
        
        "bottom_right": """
        Find the net amount in the bottom-right section.
        Return JSON: {"found_in_bottom_right": {"employee_name": "unknown", "gross_amount": "0", "net_amount": "789,10"}}
        """
    }
)

# For WHOLE mode - must use key "whole"
processor = QwenPayslipProcessor(
    window_mode="whole",
    custom_prompts={
        "whole": """
        Find all values in the entire document.
        Return JSON: {"found_in_whole": {"employee_name": "NAME", "gross_amount": "1234,56", "net_amount": "789,10"}}
        """
    }
)
```

The JSON response format MUST match the window name with the prefix `found_in_`. For example:
- Window "top" → JSON key must be "found_in_top"
- Window "bottom_right" → JSON key must be "found_in_bottom_right"

## Image Processing Parameters

Fine-tune image processing with a comprehensive set of parameters:

```python
processor = QwenPayslipProcessor(
    config={
        "pdf": {
            "dpi": 300,               # PDF rendering DPI (Range: 150-600)
        },
        "image": {
            # Multiple resolutions - will try each until one works
            "resolution_steps": [1200, 1000, 800],  # Try these resolutions in order
            
            # Or specify a single resolution
            # "resolution_steps": 1200,  # Only use 1200px resolution
            
            # Enhancement parameters
            "enhance_contrast": True,  # Enable/disable contrast enhancement
            "sharpen_factor": 2.5,     # Sharpening intensity (Range: 1.0-3.0)
            "contrast_factor": 1.8,    # Contrast enhancement (Range: 1.0-2.0)
            "brightness_factor": 1.1,  # Brightness adjustment (Range: 0.8-1.2)
            
            # OCR-specific settings (if needed later)
            "ocr_language": "deu",     # OCR language for potential OCR integration
            "ocr_threshold": 90,       # Confidence threshold for OCR (%)
        },
        "window": {
            "overlap": 0.1,           # Overlap between windows (Range: 0.0-0.5)
            "min_size": 100,          # Minimum size in pixels for a window
        },
        "text_generation": {
            "max_new_tokens": 512,    # Max tokens to generate (Range: 128-1024)
            "use_beam_search": False, # Use beam search instead of greedy decoding
            "num_beams": 1,           # Number of beams for beam search
            "temperature": 0.1,       # Temperature for generation (Range: 0.1-1.0)
            "top_p": 0.95,            # Top-p sampling parameter (Range: 0.5-1.0)
        },
        "extraction": {
            "confidence_threshold": 0.7,  # Minimum confidence for extracted values
            "fuzzy_matching": True        # Use fuzzy matching for field names
        }
    }
)
```

## Complete Configuration Reference

Below is the complete configuration structure with default values and valid ranges:

```python
default_config = {
    "pdf": {
        "dpi": 600  # Range: 150-600
    },
    "image": {
        "resolution_steps": [1500, 1200, 1000, 800, 600],
        "enhance_contrast": True,
        "sharpen_factor": 2.5,  # Range: 1.0-3.0
        "contrast_factor": 1.8,  # Range: 1.0-2.0
        "brightness_factor": 1.1  # Range: 0.8-1.2
    },
    "window": {
        "overlap": 0.1  # Range: 0.0-0.5
    },
    "text_generation": {
        "max_new_tokens": 768,  # Range: 128-1024
        "use_beam_search": False,
        "num_beams": 1  # Only relevant if use_beam_search is True
    }
}
```

You can override any of these values by passing a partial configuration dictionary:

```python
# Override specific settings while keeping defaults for others
processor = QwenPayslipProcessor(
    config={
        "pdf": {"dpi": 300},
        "image": {
            "resolution_steps": [1200, 800],
            "sharpen_factor": 2.0
        }
    }
)
```

## Output Format

Results are returned in this format:

```python
{
    "results": [
        # One item per processed page
        {
            "employee_name": "Max Mustermann",
            "gross_amount": "3.500,00",
            "net_amount": "2.100,00",
            "page_index": 0,
            "page_number": 1
        },
        # If multiple pages were processed
        {
            "employee_name": "unknown",
            "gross_amount": "4.200,00", 
            "net_amount": "2.500,00",
            "page_index": 2,
            "page_number": 3
        }
    ],
    "processing_time": 12.34,  # Total processing time in seconds
    "total_pages": 5,          # Total pages in the document
    "processed_pages": 2       # Number of pages that were processed
}
```

## Isolated Environment Usage

For environments with restricted internet access or where you want to keep the model server separate from your application code, we provide a Docker-based solution. This approach offers several advantages:

- Complete isolation between model server and client application
- No need to install large model files on the client machine
- Easy setup in air-gapped environments
- Ability to scale model serving independently

### Option 1: Using the Pre-built Docker Image (Internet Access Required Once)

This option requires internet access only when pulling the Docker image.

1. **Pull the Docker image** (on a machine with internet access):

```bash
docker pull calvinsendawula/qwen-payslip-processor:latest
```

2. **Save the Docker image to a file**:

```bash
docker save calvinsendawula/qwen-payslip-processor:latest > qwen-payslip-processor.tar
```

3. **Transfer the image file** to your isolated environment.

4. **Load the Docker image** in your isolated environment:

```bash
docker load < qwen-payslip-processor.tar
```

5. **Run the Docker container** with an exposed port:

```bash
docker run -d -p 27842:27842 --name qwen-model calvinsendawula/qwen-payslip-processor:latest
```

6. **Configure your client application** to use the Docker container:

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Initialize the processor with the Docker endpoint
processor = QwenPayslipProcessor(
    model_endpoint="http://localhost:27842"  # The endpoint can be any accessible host/port
)

# Use the processor normally - it will send requests to the Docker container
with open("path/to/payslip.pdf", "rb") as f:
    pdf_data = f.read()
    
results = processor.process_pdf(pdf_data)
print(results)
```

### Option 2: Building the Docker Image Yourself

If you prefer to build the Docker image yourself:

1. **Clone the repository** (on a machine with internet access):

```bash
git clone https://github.com/calvinsendawula/qwen-payslip-processor.git
cd qwen-payslip-processor/docker
```

2. **Build the Docker image**:

```bash
docker build -t qwen-payslip-processor:custom .
```

3. **Follow steps 2-6 from Option 1** above, replacing the image name with `qwen-payslip-processor:custom`.

### Option 3: Using Remote Endpoints

You can also use a remote endpoint if you have a separate server running the model:

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Connect to a remote endpoint
processor = QwenPayslipProcessor(
    model_endpoint="http://model-server.example.com:27842"
)

# Use the processor as normal
results = processor.process_pdf(pdf_data)
```

### Docker Container API Reference

The Docker container exposes the following API endpoints:

#### POST /process/pdf

Process a PDF document.

**Request Body**: 
- `file`: PDF file content (multipart/form-data)
- `pages`: (Optional) Page numbers to process as comma-separated values (e.g., "1,3,5")
- `window_mode`: (Optional) Window mode (e.g., "vertical", "whole")
- `selected_windows`: (Optional) Windows to process (e.g., "top" or "top,bottom")

**Response**: JSON with extraction results

**Example using curl**:
```bash
curl -X POST http://localhost:27842/process/pdf \
  -F "file=@payslip.pdf" \
  -F "pages=1,2" \
  -F "window_mode=vertical" \
  -F "selected_windows=top,bottom"
```

#### POST /process/image

Process an image.

**Request Body**: 
- `file`: Image file content (multipart/form-data)
- `window_mode`: (Optional) Window mode
- `selected_windows`: (Optional) Windows to process

**Response**: JSON with extraction results

**Example using curl**:
```bash
curl -X POST http://localhost:27842/process/image \
  -F "file=@payslip.jpg" \
  -F "window_mode=quadrant" \
  -F "selected_windows=top_left,bottom_right"
```

#### GET /status

Check if the model server is running.

**Response**: JSON with status information

**Example using curl**:
```bash
curl http://localhost:27842/status
```

### Docker Container Resource Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB+ recommended
- **Disk Space**: 20GB+ for the Docker image
- **GPU**: NVIDIA GPU with 8GB+ VRAM for optimal performance (not required)

To run with GPU support:

```bash
docker run -d -p 27842:27842 --gpus all --name qwen-model calvinsendawula/qwen-payslip-processor:latest
```

## Notes and Performance Considerations

- First use will download the Qwen model (approximately 14GB) unless pre-downloaded
- Memory requirements increase with higher resolutions:
  - 1500px resolution may require up to 16GB VRAM
  - Use lower resolutions (800-1000px) for GPUs with less VRAM
- For multi-page PDFs, processing time scales linearly with page count

## License

MIT
