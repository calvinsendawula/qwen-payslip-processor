#!/usr/bin/env python
"""
Script to build and publish the qwen-payslip-processor package to PyPI.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command):
    """Run a shell command and log the output"""
    logger.info(f"Running: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    
    if stdout:
        logger.info(stdout.decode())
    if stderr:
        logger.error(stderr.decode())
    
    return process.returncode

def check_model_files():
    """Check if model files exist"""
    script_dir = Path(__file__).parent.absolute()
    model_dir = os.path.join(script_dir, "qwen_payslip_processor", "model_files")
    marker_file = os.path.join(model_dir, "MODEL_READY")
    model_path = os.path.join(model_dir, "model")
    processor_path = os.path.join(model_dir, "processor")
    
    if (os.path.exists(marker_file) and 
        os.path.exists(model_path) and 
        os.path.exists(processor_path)):
        logger.info("Model files found!")
        return True
    
    logger.warning("Model files not found. Need to download them first.")
    return False

def download_model():
    """Download the model using the download script"""
    script_dir = Path(__file__).parent.absolute()
    download_script = os.path.join(script_dir, "download_model.py")
    
    logger.info("Downloading model files...")
    return run_command(f"python {download_script}") == 0

def install_build_tools():
    """Install the necessary build tools"""
    logger.info("Installing build tools...")
    return run_command("pip install build twine") == 0

def build_package():
    """Build the package"""
    script_dir = Path(__file__).parent.absolute()
    
    logger.info("Building package...")
    os.chdir(script_dir)
    return run_command("python -m build") == 0

def publish_package():
    """Publish the package to PyPI"""
    script_dir = Path(__file__).parent.absolute()
    dist_dir = os.path.join(script_dir, "dist")
    
    logger.info("Publishing package to PyPI...")
    os.chdir(script_dir)
    
    # Check if credentials exist
    username = os.environ.get("PYPI_USERNAME")
    password = os.environ.get("PYPI_PASSWORD")
    
    if username and password:
        return run_command(f"python -m twine upload dist/* -u {username} -p {password}") == 0
    else:
        logger.info("No PyPI credentials found in environment. Using ~/.pypirc or prompting...")
        return run_command("python -m twine upload dist/*") == 0

def main():
    """Main function to build and publish the package"""
    # Check if model files exist, if not download them
    if not check_model_files():
        success = download_model()
        if not success:
            logger.error("Failed to download model files. Aborting.")
            return False
    
    # Install build tools
    if not install_build_tools():
        logger.error("Failed to install build tools. Aborting.")
        return False
    
    # Build the package
    if not build_package():
        logger.error("Failed to build package. Aborting.")
        return False
    
    # Ask user if they want to publish
    publish = input("Do you want to publish the package to PyPI? (y/n): ").lower() == 'y'
    
    if publish:
        if not publish_package():
            logger.error("Failed to publish package.")
            return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting package build process...")
    
    success = main()
    
    if success:
        logger.info("Package build completed successfully!")
        sys.exit(0)
    else:
        logger.error("Package build failed! Please check the logs for details.")
        sys.exit(1) 