"""
Utility functions for the Qwen Payslip Processor
"""

import torch
import re
import json
import logging
from PIL import Image, ImageEnhance
from io import BytesIO
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_image_for_vl_model(image, target_long_side, enhance_contrast=True, 
                               sharpen_factor=2.5, contrast_factor=1.8, brightness_factor=1.1):
    """
    Optimize image for vision-language model while preserving aspect ratio and readability.
    
    Args:
        image (PIL.Image): Input image
        target_long_side (int): Target length of the long side of the image
        enhance_contrast (bool): Whether to apply contrast enhancement
        sharpen_factor (float): Sharpening factor to apply
        contrast_factor (float): Contrast enhancement factor
        brightness_factor (float): Brightness adjustment factor
        
    Returns:
        PIL.Image: Optimized image
    """
    # Calculate the scaling factor
    long_side = max(image.width, image.height)
    scale_factor = target_long_side / long_side

    # Resize while maintaining aspect ratio
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Ensure dimensions are even (sometimes helps with memory alignment)
    if new_width % 2 == 1:
        new_width += 1
    if new_height % 2 == 1:
        new_height += 1

    # Apply image enhancement if enabled
    if enhance_contrast:
        try:
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)

            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

            # Apply sharpening
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpen_factor)
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")

    # Resize image with high-quality interpolation
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    logger.debug(f"Resized image from {image.width}x{image.height} to {new_width}x{new_height}")
    return resized_image

def split_image_for_window_mode(image, window_mode="vertical", window_regions=None, overlap=0.1):
    """
    Split the image into multiple windows based on the specified mode.
    
    Args:
        image (PIL.Image): Input image
        window_mode (str): How to split the image - "vertical", "horizontal", "quadrant", or "whole"
        window_regions (list): Deprecated, kept for backward compatibility
        overlap (float): Overlap between windows as a fraction (0.0-0.5)
        
    Returns:
        list: List of tuples containing (window_image, window_position)
    """
    width, height = image.size
    windows = []
    
    # Calculate overlap pixels (different for each dimension)
    overlap_height = int(height * overlap)
    overlap_width = int(width * overlap)
    
    if window_mode == "whole":
        # Process the whole image as one window
        windows.append((image, "whole"))
    
    elif window_mode == "vertical":
        # Split into top and bottom with overlap
        top_height = height // 2 + overlap_height // 2
        bottom_start = height // 2 - overlap_height // 2
        
        top_window = image.crop((0, 0, width, top_height))
        bottom_window = image.crop((0, bottom_start, width, height))
        
        windows.append((top_window, "top"))
        windows.append((bottom_window, "bottom"))
    
    elif window_mode == "horizontal":
        # Split into left and right with overlap
        left_width = width // 2 + overlap_width // 2
        right_start = width // 2 - overlap_width // 2
        
        left_window = image.crop((0, 0, left_width, height))
        right_window = image.crop((right_start, 0, width, height))
        
        windows.append((left_window, "left"))
        windows.append((right_window, "right"))
    
    elif window_mode == "quadrant":
        # Split into four quadrants with overlap
        mid_x = width // 2
        mid_y = height // 2
        
        # Calculate boundaries with overlap
        top_boundary = mid_y + overlap_height // 2
        bottom_boundary = mid_y - overlap_height // 2
        left_boundary = mid_x + overlap_width // 2
        right_boundary = mid_x - overlap_width // 2
        
        # Create the four quadrants
        top_left = image.crop((0, 0, left_boundary, top_boundary))
        top_right = image.crop((right_boundary, 0, width, top_boundary))
        bottom_left = image.crop((0, bottom_boundary, left_boundary, height))
        bottom_right = image.crop((right_boundary, bottom_boundary, width, height))
        
        windows.append((top_left, "top_left"))
        windows.append((top_right, "top_right"))
        windows.append((bottom_left, "bottom_left"))
        windows.append((bottom_right, "bottom_right"))
    
    else:
        # Default to whole image if the mode is invalid
        logger.warning(f"Unknown window mode '{window_mode}'. Using whole image.")
        windows.append((image, "whole"))
    
    # Log what we've created
    for i, (window, position) in enumerate(windows):
        logger.debug(f"Created window {i+1} ({position}): Dimensions {window.size[0]}x{window.size[1]}")
    
    return windows

def cleanup_memory():
    """Clean up CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def extract_json_from_text(text):
    """Extract JSON object from text response"""
    json_pattern = r'({.*})'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
    
    logger.warning("No valid JSON found in text")
    return None
