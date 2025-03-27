#!/usr/bin/env python
"""
Script to download and save the Qwen2.5-VL-7B model for packaging.
This script should be run before building the PyPI package.
"""

import os
import sys
import torch
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    """Download and save Qwen model files"""
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    logger.info(f"Downloading {model_id}...")
    
    # Define paths
    script_dir = Path(__file__).parent.absolute()
    model_dir = os.path.join(script_dir, "qwen_payslip_processor", "model_files")
    model_path = os.path.join(model_dir, "model")
    processor_path = os.path.join(model_dir, "processor")
    
    # Create directories
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(processor_path, exist_ok=True)
    
    logger.info(f"Model will be saved to: {model_path}")
    logger.info(f"Processor will be saved to: {processor_path}")
    
    try:
        # Download processor
        logger.info("Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(processor_path)
        logger.info("Processor downloaded and saved successfully!")
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            torch_dtype=torch.float16  # Use half-precision to reduce package size
        )
        model.save_pretrained(model_path)
        logger.info("Model downloaded and saved successfully!")
        
        # Create a marker file
        with open(os.path.join(model_dir, "MODEL_READY"), "w") as f:
            f.write("Model files downloaded and ready for packaging")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting model download for packaging...")
    
    success = download_model()
    
    if success:
        logger.info("Download completed successfully! The model is ready for packaging.")
        sys.exit(0)
    else:
        logger.error("Download failed! Please check the logs for details.")
        sys.exit(1) 