# Qwen Payslip Processor

A Python package for processing German payslips using the Qwen2.5-VL-7B vision-language model.

## Features

- Extracts key information from German payslips in PDF or image format
- Automatically downloads and caches the Qwen model if not present
- Supports multiple window modes (whole image, vertical split, horizontal split, quadrants, etc.)
- Customizable prompts for specific document regions
- Includes progressive resolution reduction to handle memory constraints
- Memory-optimized with CUDA GPU acceleration (falls back to CPU when unavailable)

## Installation

```bash
pip install qwen-payslip-processor
```

## Requirements

- Python 3.11.5 or higher
- CUDA-compatible GPU recommended for optimal performance
- PyTorch with CUDA support for GPU acceleration
- 16GB+ RAM (32GB recommended for GPU-accelerated processing)
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

## Advanced Usage

```python
# Custom window configuration
processor = QwenPayslipProcessor(
    window_mode="quadrant",  # Split image into quadrants
    force_cpu=False,         # Use GPU if available (default)
    custom_prompts={
        "top_left": "Custom prompt for the top left section...",
        "bottom_right": "Custom prompt for the bottom right section..."
    },
    config={
        "pdf": {"dpi": 300},  # Lower DPI for faster processing
        "image": {
            "resolution_steps": [1200, 1000, 800],  # Custom resolution steps
            "enhance_contrast": True
        }
    }
)

# Define custom regions
regions = [
    {"name": "header", "box": (0, 0, 2000, 500)},      # Left, Top, Right, Bottom
    {"name": "amounts", "box": (1500, 500, 2000, 1000)}
]

# Process with custom regions
processor = QwenPayslipProcessor(
    window_mode="custom",
    window_regions=regions
)
```

## Extracted Information

The processor extracts:
- Employee name
- Gross amount 
- Net amount

## Supported Window Modes

- `"whole"`: Process the entire image
- `"vertical"`: Split into top and bottom halves
- `"horizontal"`: Split into left and right halves
- `"quadrant"`: Split into four quadrants (top_left, top_right, bottom_left, bottom_right)
- `"custom"`: Custom regions defined by the user

## GPU Acceleration

The package automatically uses CUDA GPU acceleration when available:

```python
# Force CPU-only processing regardless of GPU availability
processor = QwenPayslipProcessor(force_cpu=True)

# Check which device is being used
print(processor.device)  # Will show 'cuda' or 'cpu'
```

## Isolated Environment Usage

For environments with restricted internet access:

1. Download the model on a machine with internet access:
   ```bash
   # In the package directory
   python download_model.py
   ```

2. Copy the `model_files` directory to the same relative path in the isolated environment

3. The package will use the pre-downloaded model files without attempting to download

## Package Development and Publishing

### Setup for Development

```bash
# Clone the repository
git clone https://github.com/calvinsendawula/qwen-payslip-processor.git
cd qwen-payslip-processor

# Install in development mode
pip install -e .
```

### Building and Publishing

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Test in TestPyPI (optional)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Notes

- On first use, the package will download the Qwen model (approximately 14GB)
- The model files are cached for future use
- GPU is strongly recommended for optimal performance
- Processing times vary significantly between CPU (very slow) and GPU (much faster)

## License

MIT
