#!/usr/bin/env python
"""
Script to download model on container startup if needed.
This is run by the container entrypoint.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_downloader")

def download_model_if_needed():
    """Check if model exists and download if needed"""
    try:
        logger.info("Checking if model needs to be downloaded...")
        from qwen_payslip_processor import QwenPayslipProcessor
        
        # Just create an instance - it will auto-download if needed
        processor = QwenPayslipProcessor(force_cpu=True)
        logger.info("Model ready for use!")
        return True
    except Exception as e:
        logger.error(f"Error preparing model: {e}")
        return False

if __name__ == "__main__":
    download_model_if_needed() 