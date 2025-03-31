"""
Default configuration values for the Qwen Payslip Processor
"""

DEFAULT_CONFIG = {
    "pdf": {
        "dpi": 600
    },
    "image": {
        "resolution_steps": [1500, 1200, 1000, 800, 600],
        "enhance_contrast": True,
        "sharpen_factor": 2.5,
        "contrast_factor": 1.8,
        "brightness_factor": 1.1,
        "ocr_language": "deu",     # German language for potential OCR integration
        "ocr_threshold": 90        # Confidence threshold for OCR (%)
    },
    "window": {
        "overlap": 0.1,
        "min_size": 100           # Minimum size in pixels for a window
    },
    "text_generation": {
        "max_new_tokens": 768,
        "use_beam_search": False,
        "num_beams": 1,
        "temperature": 0.1,       # Temperature for generation
        "top_p": 0.95,            # Top-p sampling parameter
        "auto_process_results": True
    },
    "extraction": {
        "confidence_threshold": 0.7,  # Minimum confidence for extracted values
        "fuzzy_matching": True        # Use fuzzy matching for field names
    },
    "global": {                   # Global settings that apply to all pages by default
        "mode": "whole",          # Default mode for all pages: "whole", "vertical", "horizontal", "quadrant", "auto"
        "prompt": None            # Default prompt for all pages (None = use default prompt for mode)
    },
    "pages": {}                   # Page-specific configurations, will be populated as needed
} 