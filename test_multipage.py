import os
import sys
import json
import logging
from qwen_payslip_processor.processor import QwenPayslipProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_multipage_config():
    """Test the multi-page configuration feature"""
    # Path to a test PDF document
    pdf_path = "./test.pdf"  # Replace with an actual path to a multi-page PDF
    
    if not os.path.exists(pdf_path):
        print(f"Error: Test PDF file not found at {pdf_path}")
        print("Please place a multi-page PDF for testing at this location or change the path.")
        return
    
    # Read the PDF file
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Create configuration with page-specific settings
    config = {
        "global": {
            "mode": "whole",  # Default mode for all pages
            "prompt": "Extract all information from this payslip"
        },
        "pages": {
            "1": {  # Settings for page 1
                "mode": "quadrant",
                "prompt": "Extract header information from this payslip"
            },
            "2-3": {  # Settings for pages 2-3
                "mode": "vertical",
                "window_count": 2,
                "prompt": "Extract tabular data from this payslip"
            }
        }
    }
    
    # Create processor with configuration
    processor = QwenPayslipProcessor(config=config)
    
    # Process the PDF with all pages
    print("Processing all pages...")
    result = processor.process_pdf(pdf_bytes)
    print(json.dumps(result, indent=2))
    
    # Process only specific pages
    print("\nProcessing only pages 1 and 3...")
    result = processor.process_pdf(pdf_bytes, pages=[1, 3])
    print(json.dumps(result, indent=2))
    
    # Test auto-detection mode
    auto_config = {
        "global": {
            "mode": "auto"
        }
    }
    
    print("\nTesting auto-detection mode...")
    processor = QwenPayslipProcessor(config=auto_config)
    result = processor.process_pdf(pdf_bytes, pages=[1])
    print(json.dumps(result, indent=2))
    
    print("Multi-page configuration test completed")

if __name__ == "__main__":
    test_multipage_config() 