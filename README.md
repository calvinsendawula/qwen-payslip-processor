# Qwen Payslip Processor

A Python package for processing German payslips using the Qwen2.5-VL-7B vision-language model.

## Features

- Extracts key information from German payslips in PDF or image format
- Automatically downloads and caches the Qwen model if not present
- Highly customizable processing options:
  - Page selection for multi-page PDFs
  - Page-specific configurations for multi-page documents
  - Multiple window division modes (whole, vertical split, horizontal split, quadrants)
  - Auto-detection of optimal processing mode
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
    memory_isolation="auto",
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
| | `window_mode` | Division mode for processing | `"vertical"` | `"whole"`, `"vertical"`, `"horizontal"`, `"quadrant"`, `"auto"` |
| | `selected_windows` | Which windows to process | `None` (all) | * `"whole"` mode: parameter ignored<br>* `"vertical"` mode: `"top"`, `"bottom"`, or `["top", "bottom"]`<br>* `"horizontal"` mode: `"left"`, `"right"`, or `["left", "right"]`<br>* `"quadrant"` mode: any combination of `"top_left"`, `"top_right"`, `"bottom_left"`, `"bottom_right"` |
| | `force_cpu` | Force CPU even if GPU available | `False` | `True`, `False` |
| | `memory_isolation` | Control model memory isolation | `"auto"` | `"none"`, `"medium"`, `"strict"`, `"auto"` |
| | `custom_prompts` | Custom instructions for windows | `{}` | Dict with keys matching window names (e.g., `{"top": "prompt text", "bottom": "prompt text"}`) |
| **Global Config** | | | | |
| | `config["global"]["mode"]` | Default mode for all pages | `"whole"` | `"whole"`, `"vertical"`, `"horizontal"`, `"quadrant"`, `"auto"` |
| | `config["global"]["prompt"]` | Default prompt for all pages | `None` | Any string |
| | `config["global"]["selected_windows"]` | Default windows for all pages | `None` (all) | Same as main `selected_windows` parameter |
| **Page-Specific Config** | | | | |
| | `config["pages"]["range"]["mode"]` | Mode for specific pages | - | `"whole"`, `"vertical"`, `"horizontal"`, `"quadrant"`, `"auto"` |
| | `config["pages"]["range"]["prompt"]` | Prompt for specific pages | - | Any string |
| | `config["pages"]["range"]["selected_windows"]` | Windows for specific pages | - | Same as main `selected_windows` parameter |
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
    # Window division mode - exactly one of: "whole", "vertical", "horizontal", "quadrant", "auto"
    window_mode="vertical",
    
    # Only process specific windows - must match the window_mode you selected
    # For "vertical" mode, can use: ["top", "bottom"] or just "top" or just "bottom"
    # For "horizontal" mode, can use: ["left", "right"] or just "left" or just "right"
    # For "quadrant" mode, can use any combination of: ["top_left", "top_right", "bottom_left", "bottom_right"]
    # For "whole" mode: this parameter is ignored
    # For "auto" mode: the appropriate windows will be selected based on the detected mode
    selected_windows=["top"],  
    
    # Force CPU processing even if GPU is available
    force_cpu=False,
    
    # Control memory isolation behavior (new in v0.2.0)
    # Options:
    # - "none": No special memory isolation (fastest but may have context bleeding)
    # - "medium": Uses prompt engineering to prevent context bleeding (balanced)
    # - "strict": Complete process isolation for each window (slowest but most reliable)
    # - "auto": Automatically select based on hardware (default)
    memory_isolation="auto",
    
    # Custom prompts that MUST match your window_mode and selected_windows
    # Keys MUST be exactly the same as the window names (e.g., "top", "bottom_right")
    custom_prompts={
        "top": "Find employee name in this section...",
        "bottom": "Find gross and net amounts in this section..."
    },
    
    # Configuration dictionary with all parameters
    config={
        # NEW: Global settings that apply to all pages by default
        "global": {
            "mode": "vertical",  # Default mode for all pages
            "prompt": "Extract payslip information",  # Default prompt for all pages
            "selected_windows": ["top", "bottom"]  # Default windows for all pages
        },
        
        # NEW: Page-specific configurations that override global settings
        "pages": {
            "1": {  # Settings for page 1 only
                "mode": "quadrant",
                "prompt": "Extract header information",
                "selected_windows": ["top_left"]
            },
            "2-3": {  # Settings for pages 2-3
                "mode": "vertical",
                "prompt": "Extract tabular data",
                "selected_windows": ["bottom"]
            },
            "4,6-8": {  # Settings for pages 4, 6, 7, and 8
                "mode": "auto"  # Use auto-detection for these pages
            }
        },
        
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

## Memory Isolation (New in v0.2.0)

The memory isolation feature addresses an important limitation of large language models: they tend to remember the context from previous interactions, which can lead to "context bleeding" between different parts of a document being processed. This can result in incorrect information extraction as the model may mix up content from different parts of the document.

### Available Memory Isolation Modes

1. **None (`"none"`)**
   - No special memory isolation techniques
   - Fastest processing speed
   - May experience context bleeding between windows
   - Best for single-window processing or when processing speed is critical

2. **Medium (`"medium"`)**
   - Uses prompt engineering techniques to reset context
   - Adds special instructions to the prompt telling the model to forget previous context
   - Good balance between speed and isolation
   - Works well for most use cases
   - Default for GPU processing

3. **Strict (`"strict"`)**
   - Complete process isolation for each window
   - Loads a fresh model instance for each window in a separate process
   - Most reliable isolation with zero context bleeding
   - Significantly slower due to model reloading
   - Higher memory requirements
   - Default for CPU processing

4. **Auto (`"auto"`)**
   - Automatically selects the appropriate mode based on hardware
   - Uses "medium" for GPU processing
   - Uses "strict" for CPU processing
   - Default behavior

### Usage Example

```python
from qwen_payslip_processor import QwenPayslipProcessor

# For maximum reliability with multi-page documents
processor = QwenPayslipProcessor(
    memory_isolation="strict"  # Complete process isolation
)

# For balanced performance
processor = QwenPayslipProcessor(
    memory_isolation="medium"  # Prompt-based isolation
)

# For maximum speed (when context bleeding is not a concern)
processor = QwenPayslipProcessor(
    memory_isolation="none"  # No special isolation
)

# Let the processor decide based on hardware (default)
processor = QwenPayslipProcessor(
    memory_isolation="auto"  # Auto-selection
)
```

### When to Use Each Mode

- **Use "strict" when:** Processing multi-page documents where accuracy is critical and processing time is not a concern.
- **Use "medium" when:** Processing multi-page documents with a good balance between speed and accuracy.
- **Use "none" when:** Processing single-page documents or when maximum speed is required.
- **Use "auto" when:** You're unsure which mode to use and want the processor to make the best choice based on your hardware.

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
    "processed_pages": 2,       # Number of pages that were processed
    "isolation_mode": {  # New in v0.2.0
        "requested": "strict",  # The isolation mode that was requested
        "actual": "mixed",  # What was actually used (mixed = some fallbacks occurred)
        "stats": {
            "requested": "strict",
            "windows_processed": 4,
            "strict_succeeded": 2,
            "medium_used": 2,
            "fallbacks_occurred": 2,
            "failures": 0
        }
    }
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
docker pull calvin189/qwen-payslip-processor:latest
```

2. **Save the Docker image to a file**:

```bash
docker save calvin189/qwen-payslip-processor:latest > qwen-payslip-processor.tar
```

3. **Transfer the image file** to your isolated environment.

4. **Load the Docker image** in your isolated environment:

```bash
docker load < qwen-payslip-processor.tar
```

5. **Run the Docker container** with an exposed port:

```bash
docker run -d -p 27842:27842 --name qwen-model calvin189/qwen-payslip-processor:latest
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
docker run -d -p 27842:27842 --gpus all --name qwen-model calvin189/qwen-payslip-processor:latest
```

## Notes and Performance Considerations

- First use will download the Qwen model (approximately 14GB) unless pre-downloaded
- Memory requirements increase with higher resolutions:
  - 1500px resolution may require up to 16GB VRAM
  - Use lower resolutions (800-1000px) for GPUs with less VRAM
- For multi-page PDFs, processing time scales linearly with page count

## License

MIT

## Multi-Page Configuration

For multi-page documents, you can apply different processing modes and settings to different pages:

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Create processor with page-specific configurations
processor = QwenPayslipProcessor(
    config={
        # Global settings (apply to all pages by default)
        "global": {
            "mode": "whole",  # Default mode for all pages
            "prompt": "Extract all information from this payslip"
        },
        # Page-specific settings (override globals for specific pages)
        "pages": {
            "1": {  # Settings for page 1
                "mode": "quadrant",
                "prompt": "Extract header information from this payslip"
            },
            "2-3": {  # Settings for pages 2-3
                "mode": "vertical",
                "selected_windows": ["top", "bottom"],
                "prompt": "Extract tabular data from this payslip"
            },
            "4,6-8": {  # Settings for pages 4, 6, 7, and 8
                "mode": "auto",  # Auto-detect best mode
                "prompt": "Extract any additional information from this page"
            }
        }
    }
)

# Process all pages with their specific configurations
with open("path/to/multi_page_document.pdf", "rb") as f:
    pdf_data = f.read()
    
results = processor.process_pdf(pdf_data)
```

### Page Range Specification

You can specify page ranges using these formats:
- Single page: `"1"` 
- Page range: `"2-5"` (processes pages 2, 3, 4, and 5)
- Multiple pages and ranges: `"1,3,5-7"` (processes pages 1, 3, 5, 6, and 7)

### Auto-Detection Mode

The new `"auto"` mode intelligently analyzes document dimensions to select the optimal processing approach:

```python
from qwen_payslip_processor import QwenPayslipProcessor

# Use auto-detection mode for all pages
processor = QwenPayslipProcessor(
    config={
        "global": {
            "mode": "auto"  # Automatically detect best processing mode
        }
    }
)

# Process a document with auto-detection
with open("path/to/document.pdf", "rb") as f:
    results = processor.process_pdf(f.read())
```

The auto-detection algorithm selects:
- `"horizontal"` mode for wide documents (aspect ratio > 1.5)
- `"vertical"` mode for tall documents (aspect ratio < 0.75)
- `"quadrant"` mode for large documents (> 1500px in both dimensions)
- `"whole"` mode for standard documents

## Memory Isolation Details

The `memory_isolation` parameter controls how the package prevents context bleeding between different windows of the document. Starting with v0.2.0, this is a critical feature for accurate multi-page and multi-window processing:

### Isolation Modes

- **none**: No special isolation. Fastest, but may cause the model to remember information from previous windows, potentially leading to incorrect extractions.

- **medium**: Uses prompt engineering to reset context between windows. Good balance between speed and reliability. The package adds special instructions to each prompt telling the model to forget previous context.

- **strict**: Complete process isolation. Each window is processed in its own separate process with a fresh model instance. This is the most reliable method but can be slower and more resource-intensive.

- **auto** (default): Automatically selects the isolation mode based on your hardware. Uses `medium` on GPU systems (since model reloading is inefficient on GPU) and `strict` on CPU systems.

### Automatic Fallback

The package includes a built-in fallback mechanism:

1. When `strict` isolation is requested but fails (e.g., on platforms with multiprocessing limitations), the package automatically falls back to `medium` isolation.

2. This fallback is completely transparent - your application code doesn't need to handle any errors or implement special logic.

3. The processing results include information about which isolation mode was actually used, allowing you to track if fallbacks occurred.

### Isolation Mode Usage

```python
# Request strict isolation (with automatic fallback to medium if necessary)
processor = QwenPayslipProcessor(memory_isolation="strict")

# Process with confidence that the package will handle isolation properly
result = processor.process_pdf(pdf_bytes)

# Check what isolation mode was actually used (optional)
isolation_info = result["isolation_mode"]
print(f"Requested: {isolation_info['requested']}, Actual: {isolation_info['actual']}")
print(f"Windows processed: {isolation_info['stats']['windows_processed']}")
print(f"Fallbacks occurred: {isolation_info['stats']['fallbacks_occurred']}")
```

## Result Format

The package returns comprehensive results with detailed information about the processing:

```python
{
    "results": [
        {
            # Page 1 results
            "employee_name": "Max Mustermann",
            "gross_amount": "3.500,00",
            "net_amount": "2.100,00",
            "page_number": 1,
            "page_index": 0
        },
        {
            # Page 2 results
            "employee_name": "unknown",  # Not found on page 2
            "gross_amount": "0",  # Not found on page 2
            "net_amount": "0",  # Not found on page 2
            "page_number": 2,
            "page_index": 1
        }
    ],
    "processing_time": 25.78,  # Total processing time in seconds
    "total_pages": 2,  # Total pages in the document
    "processed_pages": 2,  # Number of pages processed
    "isolation_mode": {  # New in v0.2.0
        "requested": "strict",  # The isolation mode that was requested
        "actual": "mixed",  # What was actually used (mixed = some fallbacks occurred)
        "stats": {
            "requested": "strict",
            "windows_processed": 4,
            "strict_succeeded": 2,
            "medium_used": 2,
            "fallbacks_occurred": 2,
            "failures": 0
        }
    }
}
```

### Isolation Statistics

The `isolation_mode` section provides valuable information:

- `requested`: The isolation mode that was requested when creating the processor
- `actual`: Which isolation mode was actually used
  - Same as `requested` if no fallbacks occurred
  - `"mixed"` if some windows used different isolation modes due to fallbacks
- `stats`: Detailed statistics about isolation
  - `windows_processed`: Total number of windows processed
  - `strict_succeeded`: Number of windows successfully processed with strict isolation
  - `medium_used`: Number of windows processed with medium isolation
  - `fallbacks_occurred`: Number of times strict isolation failed and fell back to medium
  - `failures`: Number of windows that couldn't be processed with any isolation method

| **Global Config** | | | | |
|----------|-----------|-------------|---------|-------------|
| | `config["global"]["mode"]` | Default mode for all pages | `"whole"` | `"whole"`, `"vertical"`, `"horizontal"`, `"quadrant"`, `"auto"` |
| | `config["global"]["prompt"]` | Default prompt for all pages | `None` | Any string |
| | `config["global"]["selected_windows"]` | Default windows for all pages | `None` (all) | Same as main `selected_windows` parameter |
