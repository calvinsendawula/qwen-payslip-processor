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
import multiprocessing
import io
import pickle
import copy
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add GPU memory cleanup
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Set multiprocessing start method
# This needs to be done here at import time
if __name__ != "__main__":  # Only set when imported, not when run directly
    try:
        # Check if we're already using spawn (probably on Windows)
        if multiprocessing.get_start_method() != 'spawn':
            # Check if we can set it to spawn
            if hasattr(multiprocessing, 'get_all_start_methods') and 'spawn' in multiprocessing.get_all_start_methods():
                # Set the start method for better cross-platform compatibility
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Multiprocessing start method set to 'spawn'")
            else:
                logger.warning("Could not set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        # This happens if the start method has already been set
        logger.info(f"Multiprocessing start method already set to: {multiprocessing.get_start_method()}")

def optimize_image_for_vl_model(image, resolution=1000, enhance_contrast=True, 
                               sharpen_factor=2.5, contrast_factor=1.8, brightness_factor=1.1):
    """Optimize an image for processing with a vision-language model
    
    Args:
        image (PIL.Image): Input image
        resolution (int): Target resolution for longest side
        enhance_contrast (bool): Whether to enhance contrast
        sharpen_factor (float): Sharpening factor
        contrast_factor (float): Contrast adjustment factor
        brightness_factor (float): Brightness adjustment factor
        
    Returns:
        PIL.Image: Optimized image
    """
    # Resize the image while preserving aspect ratio
    width, height = image.size
    if width > height:
        new_width = resolution
        new_height = int(height * (resolution / width))
    else:
        new_height = resolution
        new_width = int(width * (resolution / height))
    
    resized_img = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Skip enhancement if not requested
    if not enhance_contrast:
        return resized_img
    
    # Apply enhancements
    # 1. Adjust brightness
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(resized_img)
        resized_img = enhancer.enhance(brightness_factor)
    
    # 2. Adjust contrast
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(resized_img)
        resized_img = enhancer.enhance(contrast_factor)
    
    # 3. Sharpen the image
    if sharpen_factor > 1.0:
        enhancer = ImageEnhance.Sharpness(resized_img)
        resized_img = enhancer.enhance(sharpen_factor)
    
    return resized_img

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
    """Force garbage collection and clear CUDA cache if available"""
    gc.collect()
    
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize CUDA operations
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error cleaning up CUDA memory: {e}")
            
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

def parse_page_range(page_range_str):
    """
    Parse a page range string into a list of page numbers
    
    Supports formats like:
    - "1" (single page)
    - "1-3" (page range)
    - "1,3,5" (comma-separated pages)
    - "1-3,5,7-9" (combination)
    
    Args:
        page_range_str (str): String representing page range
        
    Returns:
        list: List of page numbers (1-indexed)
    """
    if not page_range_str:
        return []
        
    page_numbers = []
    parts = page_range_str.split(',')
    
    for part in parts:
        if '-' in part:
            # Handle ranges like "1-5"
            try:
                start, end = map(int, part.split('-'))
                page_numbers.extend(range(start, end + 1))
            except ValueError:
                logger.warning(f"Invalid page range: '{part}'")
        else:
            # Handle single pages
            try:
                page_numbers.append(int(part))
            except ValueError:
                logger.warning(f"Invalid page number: '{part}'")
    
    return sorted(set(page_numbers))  # Remove duplicates and sort

def get_page_config(config, page_number):
    """
    Get the configuration for a specific page by merging global and page-specific settings.
    
    Args:
        config (dict): Full configuration dictionary with 'global' and 'pages' keys
        page_number (int): The page number to get configuration for (1-indexed)
        
    Returns:
        dict: Configuration for the specific page
    """
    # Start with global config
    if "global" not in config:
        return {}
    
    page_config = config["global"].copy()
    
    # No page-specific configs
    if "pages" not in config or not config["pages"]:
        return page_config
    
    # Check for page-specific configs
    for page_range_str, range_config in config["pages"].items():
        page_range = parse_page_range(page_range_str)
        
        if page_number in page_range:
            # Update the configuration with page-specific settings
            page_config.update(range_config)
    
    return page_config

def detect_best_processing_mode(image):
    """
    Automatically detect the best processing mode for a given image.
    
    This function analyzes the image content and dimensions to determine the optimal 
    processing approach (whole, vertical, horizontal, or quadrant).
    
    Args:
        image (PIL.Image): Input image to analyze
        
    Returns:
        str: Recommended processing mode ("whole", "vertical", "horizontal", or "quadrant")
    """
    width, height = image.size
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # 1. For very wide documents (e.g., wide tables or horizontal layouts)
    if aspect_ratio > 1.5:
        logger.info(f"Auto detection: wide document (aspect ratio {aspect_ratio:.2f}), using horizontal mode")
        return "horizontal"
    
    # 2. For very tall documents (e.g., multi-column layouts or receipts)
    elif aspect_ratio < 0.75:
        logger.info(f"Auto detection: tall document (aspect ratio {aspect_ratio:.2f}), using vertical mode")
        return "vertical"
    
    # 3. For large, detailed documents, use quadrant mode
    elif width > 1500 and height > 1500:
        logger.info(f"Auto detection: large document ({width}x{height}), using quadrant mode")
        return "quadrant"
    
    # 4. For smaller or balanced documents, process as a whole
    else:
        logger.info(f"Auto detection: standard document ({width}x{height}), using whole mode")
        return "whole"

def convert_image_to_bytes(image):
    """Convert PIL Image to bytes for passing to subprocess"""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()
    
def _process_isolation_worker(window_bytes, prompt_text, window_position, config, force_cpu, model_id, processor_id, result_queue):
    """Worker function for isolated processing - must be at module level for Windows compatibility"""
    try:
        import torch
        import os
        from pathlib import Path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        # Set device
        device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Isolated worker using device: {device}")
        
        # Convert bytes back to PIL Image
        window = Image.open(io.BytesIO(window_bytes))
        
        # Determine if we're using local model files or HuggingFace
        using_local_files = False
        local_model_path = None
        local_processor_path = None
        
        # Check if we should look for local files (if model_id contains a slash, it's a HuggingFace path)
        if "/" not in model_id:
            # Get the package directory to find local model files
            package_dir = Path(__file__).parent.absolute()
            model_dir = os.path.join(package_dir, "model_files")
            local_model_path = os.path.join(model_dir, "model")
            local_processor_path = os.path.join(model_dir, "processor")
            
            # Check if local files exist
            if os.path.exists(os.path.join(model_dir, "MODEL_READY")) and \
               os.path.exists(local_model_path) and \
               os.path.exists(local_processor_path):
                using_local_files = True
                logger.info(f"Using local model files from {local_model_path}")
        
        # Load model and processor only in this process
        try:
            if using_local_files:
                logger.info(f"Loading local model in isolated process...")
                processor = AutoProcessor.from_pretrained(local_processor_path)
                model = AutoModelForImageTextToText.from_pretrained(
                    local_model_path, 
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    device_map=device
                )
            else:
                logger.info(f"Loading model {model_id} in isolated process...")
                processor = AutoProcessor.from_pretrained(processor_id)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    device_map=device
                )
        except Exception as model_error:
            logger.error(f"Failed to load model in isolated process: {str(model_error)}")
            # Send error information back through the queue
            result_queue.put({"error": f"Model loading failed: {str(model_error)}"})
            return
        
        # Process the window
        logger.info(f"Processing {window_position} window in isolated process...")
        
        # Try each resolution in sequence until one works
        found_result = None
        all_errors = []
        
        for resolution in config["image"]["resolution_steps"]:
            try:
                logger.info(f"Trying {window_position} window with resolution {resolution}...")
                
                # Resize image
                processed_window = optimize_image_for_vl_model(
                    window, 
                    resolution,
                    enhance_contrast=config["image"]["enhance_contrast"],
                    sharpen_factor=config["image"]["sharpen_factor"],
                    contrast_factor=config["image"]["contrast_factor"],
                    brightness_factor=config["image"]["brightness_factor"]
                )
                
                # Prepare conversation with image
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                
                # Process with model
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[text_prompt], images=[processed_window], padding=True, return_tensors="pt")
                inputs = inputs.to(device)
                
                # Generate output
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=config["text_generation"]["max_new_tokens"],
                        do_sample=config["text_generation"]["temperature"] > 0.1,
                        temperature=config["text_generation"]["temperature"],
                        top_p=config["text_generation"]["top_p"],
                        use_cache=True,
                        num_beams=config["text_generation"]["num_beams"] if config["text_generation"]["use_beam_search"] else 1
                    )
                
                # Process the output
                generated_ids = [output_ids[0][inputs.input_ids.shape[1]:]]
                response_text = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                # Extract JSON from the response
                json_result = extract_json_from_text(response_text)
                if json_result:
                    logger.info(f"Successfully extracted data with resolution {resolution}")
                    found_result = json_result
                    break
                else:
                    error_msg = "Failed to extract valid JSON from model response"
                    all_errors.append(f"Resolution {resolution}: {error_msg}")
                    raise ValueError(error_msg)
                
            except Exception as e:
                error_msg = f"Failed with resolution {resolution}: {e}"
                all_errors.append(error_msg)
                logger.warning(error_msg)
                # Clean up memory after each try
                cleanup_memory()
                continue
        
        # Return result or error information
        if found_result:
            result_queue.put(found_result)
        else:
            # Return error information if available
            if all_errors:
                error_details = {"error": f"All resolutions failed: {'; '.join(all_errors)}"}
                result_queue.put(error_details)
            else:
                # Return empty result based on window position
                empty_result = {}
                if window_position == "employee":
                    empty_result = {"name": "", "id": "", "title": "", "department": ""}
                elif window_position == "gross":
                    empty_result = {"amount": "", "currency": "", "period": ""}
                elif window_position == "net":
                    empty_result = {"amount": "", "currency": "", "period": ""}
                elif window_position == "supervisor":
                    empty_result = {"name": "", "id": "", "title": "", "department": ""}
                
                result_queue.put(empty_result)
            
    except Exception as e:
        logger.error(f"Error in isolated processing: {str(e)}")
        result_queue.put({"error": f"Process error: {str(e)}"})
    
    finally:
        # Always clean up memory before exiting
        cleanup_memory()

def isolated_process_window(window_bytes, prompt_text, window_position, config, force_cpu, model_id, processor_id):
    """
    Process a window in an isolated process to prevent context bleeding
    
    This function is designed to be called in a separate process for each window,
    ensuring complete memory isolation between different parts of the document.
    
    Args:
        window_bytes: PNG image bytes
        prompt_text: Prompt text for the model
        window_position: Position identifier (e.g., "top", "bottom_right")
        config: Configuration dictionary 
        force_cpu: Whether to force CPU processing
        model_id: Model ID string
        processor_id: Processor ID string
        
    Returns:
        dict: Extracted information or empty dict on failure
        
    Raises:
        Exception: If process isolation fails with a specific error
    """
    # Create a deep copy of config to ensure serializability
    config_copy = copy.deepcopy(config)
    
    # Create process and queue
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_process_isolation_worker,
        args=(window_bytes, prompt_text, window_position, config_copy, 
              force_cpu, model_id, processor_id, result_queue)
    )
    
    try:
        # Start process and wait for result with timeout
        process.start()
        process.join(timeout=300)  # 5 minute timeout
        
        # Check if process is still alive (timeout occurred)
        if process.is_alive():
            logger.error(f"Isolated processing timeout for {window_position}")
            process.terminate()
            process.join()
            raise Exception(f"Timeout occurred when processing {window_position} with strict isolation")
        
        # Get result from queue
        if not result_queue.empty():
            result = result_queue.get()
            
            # Check if result contains error information
            if isinstance(result, dict) and "error" in result:
                error_msg = result["error"]
                logger.error(f"Error in isolated process: {error_msg}")
                raise Exception(f"Strict isolation failed: {error_msg}")
            
            return result
        else:
            logger.error(f"No result from isolated process for {window_position}")
            raise Exception(f"No result received from isolated process for {window_position}")
            
    except Exception as e:
        # Log for debug information
        logger.error(f"Error managing isolated process: {str(e)}")
        if process.is_alive():
            process.terminate()
            process.join()
        # Re-raise the exception to signal that strict isolation failed
        raise
