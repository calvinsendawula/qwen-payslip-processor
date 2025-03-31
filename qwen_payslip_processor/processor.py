"""
Main processor module for Qwen Payslip extraction
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import fitz  # PyMuPDF
import yaml
import logging
import time
import re
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
from typing import Dict, List, Union, Optional, Any

from .utils import (
    optimize_image_for_vl_model,
    split_image_for_window_mode,
    cleanup_memory,
    extract_json_from_text,
    get_page_config,
    detect_best_processing_mode,
    parse_page_range,
    convert_image_to_bytes,
    isolated_process_window
)
from .config_defaults import DEFAULT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QwenPayslipProcessor:
    """Processes payslips using Qwen2.5-VL-7B vision-language model with customizable window approach"""
    
    def __init__(self, 
                 config=None,
                 custom_prompts=None,
                 window_mode="vertical",  # "whole", "vertical", "horizontal", "quadrant", "auto"
                 selected_windows=None,   # List of windows to process, e.g. ["top", "bottom_right"]
                 force_cpu=False,
                 model_endpoint=None,     # API endpoint for Docker-based processing
                 memory_isolation="auto"): # "none", "medium", "strict", or "auto"
        """Initialize the QwenPayslipProcessor with configuration
        
        Args:
            config (dict): Custom configuration (will be merged with defaults)
            custom_prompts (dict): Custom prompts for different window positions
            window_mode (str): How to split images - "whole", "vertical", "horizontal", "quadrant", "auto"
            selected_windows (list or str): Window positions to process (default: process all windows)
                                     Can be a list ["top", "bottom"] or a single string "top"
                                     Valid options depend on window_mode:
                                     - "vertical": ["top", "bottom"]
                                     - "horizontal": ["left", "right"]
                                     - "quadrant": ["top_left", "top_right", "bottom_left", "bottom_right"]
                                     - "whole": this parameter is ignored
            force_cpu (bool): Whether to force CPU usage even if GPU is available
            model_endpoint (str): URL of a remote API endpoint for Docker-based processing
                                  e.g., "http://localhost:27842"
            memory_isolation (str): Controls how memory is isolated between processing tasks
                                   - "none": No special isolation (fastest but potential context bleeding)
                                   - "medium": Uses prompt engineering to reset context (good balance)
                                   - "strict": Complete process isolation for each window (slowest but most reliable)
                                   - "auto": Automatically choose based on hardware (default)
        """
        # Store the model endpoint if provided
        self.model_endpoint = model_endpoint
        
        # Load configuration
        self.config = self._merge_config(config if config else {})
        
        # Set custom prompts if provided directly
        if custom_prompts:
            self.custom_prompts = custom_prompts
        else:
            # Check if custom prompts are provided in the config
            self.custom_prompts = {}
        
        # Set global window mode from parameters or config
        if window_mode != "vertical" or "global" not in self.config:
            # Parameter overrides or no global config present
            self.window_mode = window_mode
        else:
            # Use global mode from config if parameter uses default
            self.window_mode = self.config["global"].get("mode", "whole")
        
        # Handle selected_windows as either list or string
        if selected_windows is not None:
            if isinstance(selected_windows, str):
                self.selected_windows = [selected_windows]
            else:
                self.selected_windows = selected_windows
        else:
            self.selected_windows = None
        
        # Set memory isolation mode
        if memory_isolation == "auto":
            # Choose isolation based on hardware
            if torch.cuda.is_available():
                # If GPU available, use medium isolation (GPU reloading is inefficient)
                self.memory_isolation = "medium"
            else:
                # For CPU, strict isolation might be more efficient
                self.memory_isolation = "strict"
        else:
            self.memory_isolation = memory_isolation
        
        # Store whether to force CPU usage
        self.force_cpu = force_cpu
        
        # Set device
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}, memory isolation: {self.memory_isolation}")
        
        # Track what isolation modes were actually used
        self.isolation_stats = {
            "requested": self.memory_isolation,
            "windows_processed": 0,
            "strict_succeeded": 0,
            "medium_used": 0,
            "fallbacks_occurred": 0,
            "failures": 0
        }
        
        # Only load the model locally if no model_endpoint is provided
        if not model_endpoint:
            # Configure PyTorch for better memory management
            if torch.cuda.is_available():
                # Enable memory optimizations
                torch.cuda.empty_cache()
                # Set PyTorch to release memory more aggressively
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Define model paths for later use
            package_dir = Path(__file__).parent.absolute()
            self.model_dir = os.path.join(package_dir, "model_files")
            
            # Set model IDs for all modes
            self.model_id = "Qwen/Qwen2.5-VL-7B"
            self.processor_id = "Qwen/Qwen2.5-VL-7B"
            
            # Only load model here if not using strict isolation
            if self.memory_isolation != "strict":
                self._load_model()
        else:
            logger.info(f"Using remote model endpoint: {model_endpoint}")
            # Check if we have the requests library
            try:
                import requests
            except ImportError:
                logger.error("The 'requests' library is required for API-based processing. "
                            "Please install it using 'pip install requests'.")
                raise
    
    def _merge_config(self, user_config):
        """Merge user configuration with defaults"""
        default_config = self._get_default_config()
        
        # Deep merge the configurations
        merged_config = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                for k, v in value.items():
                    # Special handling for resolution_steps - can be list or single value
                    if k == "resolution_steps" and not isinstance(v, list):
                        merged_config[key][k] = [v]  # Convert single value to list
                    else:
                        merged_config[key][k] = v
            elif key == "global" and isinstance(value, dict):
                # Ensure global config exists
                if "global" not in merged_config:
                    merged_config["global"] = {}
                # Update global settings
                merged_config["global"].update(value)
            elif key == "pages" and isinstance(value, dict):
                # Handle page-specific configurations
                merged_config["pages"] = value
            else:
                # Override with user value for other keys
                merged_config[key] = value
        
        return merged_config
    
    def _get_default_config(self):
        """Return default configuration"""
        return DEFAULT_CONFIG.copy()
    
    def _load_model(self):
        """Load model and processor"""
        try:
            logger.info("Loading Qwen model...")
            
            # Load processor and model with appropriate settings
            self.processor = AutoProcessor.from_pretrained(self.processor_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # Move to CPU if needed
            if self.device.type != "cuda":
                self.model = self.model.to(self.device)
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def process_pdf(self, pdf_bytes, pages=None):
        """
        Process a PDF using the Qwen model
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            pages (list or int, optional): Page numbers to process (1-indexed).
                                       Can be a list [1, 3, 5] or a single page number 2.
                                       If None, processes all pages.
            
        Returns:
            dict: Extracted information
        """
        # If model_endpoint is set, use the API
        if self.model_endpoint:
            return self._process_pdf_via_api(pdf_bytes, pages)
        
        # Otherwise, use the local model
        start_time = time.time()
        logger.info("Starting PDF processing")
        
        # Import page config and auto-detection utilities
        from .utils import get_page_config, detect_best_processing_mode
        
        # Convert PDF to images using PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)
            
            # Determine which pages to process
            if pages is None:
                # Process all pages
                pages_to_process = list(range(total_pages))
            else:
                # Handle pages parameter as either list or single int
                if isinstance(pages, int):
                    page_list = [pages]
                else:
                    page_list = pages
                
                # Validate page indices (convert from 1-indexed to 0-indexed)
                pages_to_process = []
                for page_num in page_list:
                    # Convert 1-indexed page number to 0-indexed
                    page_idx = page_num - 1
                    if 0 <= page_idx < total_pages:
                        pages_to_process.append(page_idx)
                    else:
                        logger.warning(f"Skipping invalid page number {page_num} (PDF has {total_pages} pages)")
                
                if not pages_to_process:
                    logger.error("No valid pages to process")
                    return {"error": "No valid pages to process. Check page numbers."}
            
            logger.info(f"Processing {len(pages_to_process)} pages out of {total_pages} total")
            
            # Extract images for selected pages
            images = []
            for page_idx in pages_to_process:
                page = doc.load_page(page_idx)
                pix = page.get_pixmap(dpi=self.config["pdf"]["dpi"])
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((img, page_idx))
                
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            return {"error": f"PDF conversion failed: {str(e)}"}
        
        # Process images
        results = []
        for image, page_idx in images:
            # Use 1-indexed page numbers in logs and results
            page_num = page_idx + 1
            logger.info(f"Processing page {page_num}/{total_pages}")
            
            # Get page-specific configuration
            page_config = get_page_config(self.config, page_num)
            
            # Determine window mode for this page
            page_window_mode = page_config.get("mode", self.window_mode)
            logger.info(f"Using '{page_window_mode}' window mode for page {page_num}")
            
            # Handle auto mode (intelligent detection)
            if page_window_mode == "auto":
                # Use the auto-detection function
                page_window_mode = detect_best_processing_mode(image)
                logger.info(f"Auto mode: selected '{page_window_mode}' for page {page_num}")
            
            # Split image based on window mode
            windows = split_image_for_window_mode(
                image, 
                window_mode=page_window_mode,
                overlap=self.config["window"]["overlap"]
            )
            
            # Get page-specific selected windows
            page_selected_windows = None
            if "selected_windows" in page_config:
                if isinstance(page_config["selected_windows"], str):
                    page_selected_windows = [page_config["selected_windows"]]
                else:
                    page_selected_windows = page_config["selected_windows"]
            else:
                page_selected_windows = self.selected_windows
            
            # Filter windows based on selected_windows
            if page_selected_windows:
                filtered_windows = []
                for window_img, window_position in windows:
                    if window_position in page_selected_windows:
                        filtered_windows.append((window_img, window_position))
                windows = filtered_windows
                if not windows:
                    logger.warning(f"No windows selected for processing on page {page_num}. Using all windows.")
                    windows = split_image_for_window_mode(
                        image, 
                        window_mode=page_window_mode,
                        overlap=self.config["window"]["overlap"]
                    )
            
            window_results = []
            # Process each window
            for window_img, window_position in windows:
                # Get page-specific custom prompt if available
                page_custom_prompt = page_config.get("prompt")
                
                # Process the window using page-specific configuration
                if self.memory_isolation == "strict":
                    # Use strict isolation with process-based approach
                    result = self._process_window_with_isolation(window_img, window_position, page_custom_prompt)
                else:
                    # Use normal or medium isolation within same process
                    result = self._process_window(window_img, window_position, page_custom_prompt)
                
                window_results.append((window_position, result))
                
                # Ensure memory is cleaned up between windows
                cleanup_memory()
            
            # Combine window results
            combined_result = self._combine_window_results(window_results)
            
            # Add page information to result (use 1-indexed page number)
            combined_result["page_index"] = page_idx
            combined_result["page_number"] = page_num
            
            results.append(combined_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"PDF processing completed in {processing_time:.2f} seconds")
        
        # Add processing time to results
        final_result = {
            "results": results,
            "processing_time": processing_time,
            "total_pages": total_pages,
            "processed_pages": len(pages_to_process),
            "isolation_mode": {
                "requested": self.memory_isolation,
                "actual": "mixed" if self.isolation_stats["fallbacks_occurred"] > 0 else self.memory_isolation,
                "stats": self.isolation_stats
            }
        }
        
        return final_result
    
    def process_image(self, image_bytes):
        """
        Process an image using the Qwen model
        
        Args:
            image_bytes (bytes): Image file content as bytes
            
        Returns:
            dict: Extracted information
        """
        # If model_endpoint is set, use the API
        if self.model_endpoint:
            return self._process_image_via_api(image_bytes)
        
        # Otherwise, use the local model
        start_time = time.time()
        logger.info("Starting image processing")
        
        # Import auto-detection utility
        from .utils import detect_best_processing_mode
        
        # Get global configuration
        global_config = self.config.get("global", {})
        
        # Determine window mode from global config
        image_window_mode = global_config.get("mode", self.window_mode)
        
        # Convert bytes to PIL Image
        try:
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"Loaded image: {image.width}x{image.height}")
            
            # Handle auto mode (intelligent detection)
            if image_window_mode == "auto":
                # Use the auto-detection function
                image_window_mode = detect_best_processing_mode(image)
                logger.info(f"Auto mode: selected '{image_window_mode}' for image")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return {"error": f"Image loading failed: {str(e)}"}
        
        # Split image based on window mode
        windows = split_image_for_window_mode(
            image, 
            window_mode=image_window_mode,
            overlap=self.config["window"]["overlap"]
        )
        
        # Get selected windows from global config or instance
        global_selected_windows = None
        if "selected_windows" in global_config:
            if isinstance(global_config["selected_windows"], str):
                global_selected_windows = [global_config["selected_windows"]]
            else:
                global_selected_windows = global_config["selected_windows"]
        else:
            global_selected_windows = self.selected_windows
        
        # Filter windows based on selected_windows
        if global_selected_windows:
            filtered_windows = []
            for window_img, window_position in windows:
                if window_position in global_selected_windows:
                    filtered_windows.append((window_img, window_position))
            windows = filtered_windows
            if not windows:
                logger.warning(f"No windows selected for processing. Using all windows.")
                windows = split_image_for_window_mode(
                    image, 
                    window_mode=image_window_mode,
                    overlap=self.config["window"]["overlap"]
                )
        
        window_results = []
        # Process each window
        for window_img, window_position in windows:
            # Get global custom prompt if available
            global_custom_prompt = global_config.get("prompt")
            
            # Process the window using global configuration
            if self.memory_isolation == "strict":
                # Use strict isolation with process-based approach
                result = self._process_window_with_isolation(window_img, window_position, global_custom_prompt)
            else:
                # Use normal or medium isolation within same process
                result = self._process_window(window_img, window_position, global_custom_prompt)
            
            window_results.append((window_position, result))
            
            # Ensure memory is cleaned up between windows
            cleanup_memory()
        
        # Combine window results
        combined_result = self._combine_window_results(window_results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Image processing completed in {processing_time:.2f} seconds")
        
        # Add processing time to results
        final_result = {
            "results": [combined_result],  # Keep format consistent with PDF processing
            "processing_time": processing_time,
            "isolation_mode": {
                "requested": self.memory_isolation,
                "actual": "mixed" if self.isolation_stats["fallbacks_occurred"] > 0 else self.memory_isolation,
                "stats": self.isolation_stats
            }
        }
        
        return final_result
    
    def _get_prompt_for_position(self, window_position):
        """Get the appropriate prompt for the window position"""
        # Check if user provided a custom prompt for this position
        if window_position in self.custom_prompts:
            return self.custom_prompts[window_position]
        
        # Default prompts based on window position
        if window_position == "top":
            return """Du siehst die obere Hälfte einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH: Dem Namen des Angestellten, der direkt nach der Überschrift "Herrn/Frau" steht.
            SCHAUE IN DIESEM BEREICH: Im oberen linken Viertel des Dokuments, meist unter dem Label "Herrn/Frau".
            Beispiel für die Position: Der Name steht 3-4 Zeilen unter der Personalnummer.
            
            WICHTIG: Wenn du keinen Namen findest, gib "unknown" zurück.
            Ich brauche KEINEN Namen einer Firma oder einer Krankenversicherung, nur den Namen des Angestellten.
            
            Gib deinen Fund als JSON zurück:
            {
            "found_in_top": {
                "employee_name": "Name des Angestellten oder 'unknown'",
                "gross_amount": "0",
                "net_amount": "0"
            }
            }"""
        elif window_position == "bottom":
            return """Du siehst die untere Hälfte einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH BEIDEN WERTEN:
            1. Bruttogehalt ("Gesamt-Brutto"): Schaue auf der rechten Seite unter der Überschrift "Gesamt-Brutto".
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            2. Nettogehalt ("Auszahlungsbetrag"): Schaue ganz unten rechts neben "Auszahlungsbetrag".
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "0" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_bottom": {
                "employee_name": "unknown",
                "gross_amount": "Bruttogehalt oder '0'",
                "net_amount": "Nettogehalt oder '0'"
            }
            }"""
        elif window_position == "left":
            return """Du siehst die linke Hälfte einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH: Dem Namen des Angestellten, der direkt nach der Überschrift "Herrn/Frau" steht.
            SCHAUE IN DIESEM BEREICH: Im oberen linken Viertel des Dokuments, meist unter dem Label "Herrn/Frau".
            Beispiel für die Position: Der Name steht 3-4 Zeilen unter der Personalnummer.
            
            WICHTIG: Wenn du keinen Namen findest, gib "unknown" zurück.
            Ich brauche KEINEN Namen einer Firma oder einer Krankenversicherung, nur den Namen des Angestellten.
            
            Gib deinen Fund als JSON zurück:
            {
            "found_in_left": {
                "employee_name": "Name des Angestellten oder 'unknown'",
                "gross_amount": "0",
                "net_amount": "0"
            }
            }"""
        elif window_position == "right":
            return """Du siehst die rechte Hälfte einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH BEIDEN WERTEN:
            1. Bruttogehalt ("Gesamt-Brutto"): Schaue auf der rechten Seite unter der Überschrift "Gesamt-Brutto".
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            2. Nettogehalt ("Auszahlungsbetrag"): Schaue ganz unten rechts neben "Auszahlungsbetrag".
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "0" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_right": {
                "employee_name": "unknown",
                "gross_amount": "Bruttogehalt oder '0'",
                "net_amount": "Nettogehalt oder '0'"
            }
            }"""
        elif window_position == "top_left":
            return """Du siehst den oberen linken Quadranten einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH: Dem Namen des Angestellten, der direkt nach der Überschrift "Herrn/Frau" steht.
            SCHAUE IN DIESEM BEREICH: Im oberen linken Viertel des Dokuments, meist unter dem Label "Herrn/Frau".
            Beispiel für die Position: Der Name steht 3-4 Zeilen unter der Personalnummer.
            
            WICHTIG: Wenn du keinen Namen findest, gib "unknown" zurück.
            Ich brauche KEINEN Namen einer Firma oder einer Krankenversicherung, nur den Namen des Angestellten.
            
            Gib deinen Fund als JSON zurück:
            {
            "found_in_top_left": {
                "employee_name": "Name des Angestellten oder 'unknown'",
                "gross_amount": "0",
                "net_amount": "0"
            }
            }"""
        elif window_position == "top_right":
            return """Du siehst den oberen rechten Quadranten einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH BEIDEN WERTEN:
            1. Bruttogehalt ("Gesamt-Brutto"): Falls es in diesem Abschnitt sichtbar ist.
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "0" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_top_right": {
                "employee_name": "unknown",
                "gross_amount": "Bruttogehalt oder '0'", 
                "net_amount": "0"
            }
            }"""
        elif window_position == "bottom_left":
            return """Du siehst den unteren linken Quadranten einer deutschen Gehaltsabrechnung.
            
            SUCHE NACH: Allem was zur Gehaltsabrechnung gehört und in diesem Abschnitt sichtbar ist.
            Aber konzentriere dich hauptsächlich auf wichtige Beträge, wenn sie sichtbar sind.
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "0" oder "unknown" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_bottom_left": {
                "employee_name": "unknown",
                "gross_amount": "0",
                "net_amount": "0"
            }
            }"""
        elif window_position == "bottom_right":
            return """Du siehst den unteren rechten Quadranten einer deutschen Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH:
            Nettogehalt ("Auszahlungsbetrag"): Schaue nach dem Wert neben "Auszahlungsbetrag".
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "0" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_bottom_right": {
                "employee_name": "unknown",
                "gross_amount": "0",
                "net_amount": "0"
            }
            }"""
        else:  # whole or any other position
            return """Du siehst eine deutsche Gehaltsabrechnung.
            
            SUCHE PRÄZISE NACH DIESEN WERTEN:
            
            1. Name des Angestellten: Steht meist im oberen linken Viertel, nach "Herrn/Frau"
            
            2. Bruttogehalt ("Gesamt-Brutto"): Steht meist auf der rechten Seite
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            3. Nettogehalt ("Auszahlungsbetrag"): Steht meist unten rechts
            Format: Eine Zahl mit Punkten als Tausendertrennzeichen und Komma als Dezimaltrennzeichen
            
            WICHTIG: Gib NUR Werte zurück, die tatsächlich im Bild zu sehen sind.
            Gib "unknown" oder "0" zurück, wenn du einen Wert nicht findest.
            
            Gib deine Funde als JSON zurück:
            {
            "found_in_whole": {
                "employee_name": "Name des Angestellten oder 'unknown'",
                "gross_amount": "Bruttogehalt oder '0'",
                "net_amount": "Nettogehalt oder '0'"
            }
            }"""
    
    def _process_window(self, window, window_position, page_custom_prompt=None):
        """Process a window with the model, trying different resolutions"""
        # Clean up memory before processing
        cleanup_memory()
        
        # Track window processing
        self.isolation_stats["windows_processed"] += 1
        
        # When using strict isolation, delegate to isolated processing
        if self.memory_isolation == "strict":
            # Strict isolation requires one complete model load per window
            # This will be handled by calling this method from process_pdf
            # This branch should not be reached directly in strict mode
            logger.warning("Direct _process_window call in strict isolation mode - this should not happen!")
            return self._get_empty_result(window_position)
        
        # Track medium isolation usage
        if self.memory_isolation == "medium":
            self.isolation_stats["medium_used"] += 1
        
        # Get prompt text
        if self.memory_isolation == "medium":
            # Add strong context reset instructions to prompt
            reset_prefix = "WICHTIG: Dies ist eine vollständig neue und separate Aufgabe. " \
                          "Vergiss ALLES, was du vorher gesehen oder bearbeitet hast. " \
                          "Ignoriere jeden vorherigen Kontext oder Bilder.\n\n"
            
            if page_custom_prompt:
                prompt_text = reset_prefix + page_custom_prompt
            else:
                default_prompt = self._get_prompt_for_position(window_position)
                prompt_text = reset_prefix + default_prompt
        else:
            # Normal mode - no special isolation
            prompt_text = page_custom_prompt or self._get_prompt_for_position(window_position)
        
        # Try each resolution in sequence until one works
        for resolution in self.config["image"]["resolution_steps"]:
            try:
                logger.info(f"Trying {window_position} window with resolution {resolution}...")
                
                # Resize image
                processed_window = optimize_image_for_vl_model(
                    window, 
                    resolution,
                    enhance_contrast=self.config["image"]["enhance_contrast"],
                    sharpen_factor=self.config["image"]["sharpen_factor"],
                    contrast_factor=self.config["image"]["contrast_factor"],
                    brightness_factor=self.config["image"]["brightness_factor"]
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
                text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[processed_window], padding=True, return_tensors="pt")
                inputs = inputs.to(self.device)
                
                # Generate output
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["text_generation"]["max_new_tokens"],
                        do_sample=self.config["text_generation"]["temperature"] > 0.1,
                        temperature=self.config["text_generation"]["temperature"],
                        top_p=self.config["text_generation"]["top_p"],
                        use_cache=True,
                        num_beams=self.config["text_generation"]["num_beams"] if self.config["text_generation"]["use_beam_search"] else 1
                    )
                
                # Process the output
                generated_ids = [output_ids[0][inputs.input_ids.shape[1]:]]
                response_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                # Extract JSON from the response
                json_result = extract_json_from_text(response_text)
                if json_result:
                    logger.info(f"Successfully extracted data with resolution {resolution}")
                    return json_result
                else:
                    raise ValueError("Failed to extract valid JSON from model response")
                
            except Exception as e:
                logger.warning(f"Failed with resolution {resolution}: {e}")
                cleanup_memory()
                continue
        
        # If all resolutions fail, return empty result
        logger.warning(f"All resolutions failed for {window_position} window")
        return self._get_empty_result(window_position)
    
    def _get_empty_result(self, window_position):
        """Return an empty result for a window position"""
        if window_position == "employee":
            return {"name": "", "id": "", "title": "", "department": ""}
        elif window_position == "gross":
            return {"amount": "", "currency": "", "period": ""}
        elif window_position == "net":
            return {"amount": "", "currency": "", "period": ""}
        elif window_position == "supervisor":
            return {"name": "", "id": "", "title": "", "department": ""}
        else:
            return {}
            
    def _process_window_with_isolation(self, window, window_position, page_custom_prompt=None):
        """Process a window with strict memory isolation using a separate process"""
        logger.info(f"Processing {window_position} window with strict memory isolation...")
        
        # Convert window to bytes for passing to subprocess
        window_bytes = convert_image_to_bytes(window)
        
        # Get prompt text
        prompt_text = page_custom_prompt or self._get_prompt_for_position(window_position)
        
        # Track window processing
        self.isolation_stats["windows_processed"] += 1
        
        # Check for local model files and adjust model_id accordingly
        # This helps the worker process find the right files
        local_model_path = None
        if hasattr(self, 'model_dir'):
            local_model_path = os.path.join(self.model_dir, "model")
            if os.path.exists(os.path.join(self.model_dir, "MODEL_READY")):
                # If we have local files, strip the repo prefix from model_id
                # so the worker knows to look for local files
                model_id = self.model_id.split('/')[-1] if '/' in self.model_id else self.model_id
                processor_id = self.processor_id.split('/')[-1] if '/' in self.processor_id else self.processor_id
                logger.info(f"Using local model files in isolated process with model_id: {model_id}")
            else:
                # No local files, use the full HuggingFace path
                model_id = self.model_id
                processor_id = self.processor_id
                logger.info(f"Using HuggingFace model in isolated process: {model_id}")
        else:
            # No model directory, use the full HuggingFace path
            model_id = self.model_id
            processor_id = self.processor_id
            logger.info(f"Using HuggingFace model in isolated process: {model_id}")
        
        # First, try using strict isolation with process-based approach
        try:
            result = isolated_process_window(
                window_bytes=window_bytes,
                prompt_text=prompt_text,
                window_position=window_position,
                config=self.config,
                force_cpu=self.force_cpu,
                model_id=model_id,
                processor_id=processor_id
            )
            
            # Check if result is valid
            if result and not self._is_empty_result(result, window_position):
                logger.info(f"Successfully processed {window_position} window with strict isolation")
                # Track strict isolation success
                self.isolation_stats["strict_succeeded"] += 1
                return result
            else:
                # If we got an empty result, fall back to medium isolation
                logger.warning(f"Strict isolation returned empty result for {window_position}, falling back to medium isolation")
                # Temporarily switch to medium isolation for this window
                original_mode = self.memory_isolation
                self.memory_isolation = "medium"
                
                # Ensure model is loaded for medium isolation
                if not hasattr(self, 'model') or not hasattr(self, 'processor'):
                    logger.info("Loading model for medium isolation fallback")
                    self._load_model()
                
                # Process with medium isolation
                medium_result = self._process_window(window, window_position, page_custom_prompt)
                
                # Restore original mode
                self.memory_isolation = original_mode
                
                # Track fallback to medium isolation
                self.isolation_stats["fallbacks_occurred"] += 1
                self.isolation_stats["medium_used"] += 1
                
                if medium_result:
                    logger.info(f"Successfully processed {window_position} with fallback to medium isolation")
                    return medium_result
                else:
                    logger.warning(f"Both isolation methods failed for {window_position}")
                    # Track failure
                    self.isolation_stats["failures"] += 1
                    return self._get_empty_result(window_position)
                
        except Exception as e:
            # If strict isolation fails with an exception, fall back to medium isolation
            logger.warning(f"Error with strict isolation for {window_position}: {str(e)}, falling back to medium isolation")
            
            # Temporarily switch to medium isolation for this window
            original_mode = self.memory_isolation
            self.memory_isolation = "medium"
            
            # Ensure model is loaded for medium isolation
            if not hasattr(self, 'model') or not hasattr(self, 'processor'):
                logger.info("Loading model for medium isolation fallback")
                self._load_model()
            
            # Process with medium isolation
            try:
                medium_result = self._process_window(window, window_position, page_custom_prompt)
                
                # Restore original mode
                self.memory_isolation = original_mode
                
                # Track fallback to medium isolation
                self.isolation_stats["fallbacks_occurred"] += 1
                self.isolation_stats["medium_used"] += 1
                
                if medium_result:
                    logger.info(f"Successfully processed {window_position} with fallback to medium isolation")
                    return medium_result
            except Exception as fallback_error:
                logger.error(f"Fallback to medium isolation also failed: {str(fallback_error)}")
                # Restore original mode
                self.memory_isolation = original_mode
                # Track failure
                self.isolation_stats["failures"] += 1
            
            # If we got here, both methods failed
            return self._get_empty_result(window_position)
            
    def _is_empty_result(self, result, window_position):
        """Check if a result is effectively empty"""
        # For structured results like those from employee, gross, net
        if isinstance(result, dict):
            # Check if all values are empty
            return all(not val for val in result.values())
        
        # For window-specific results
        key = f"found_in_{window_position}"
        if key in result:
            # Check default values
            data = result[key]
            if "employee_name" in data and data["employee_name"] == "unknown":
                if "gross_amount" in data and data["gross_amount"] == "0":
                    if "net_amount" in data and data["net_amount"] == "0":
                        return True
        
        return False
    
    def _combine_window_results(self, window_results):
        """Combine results from multiple windows
        
        Args:
            window_results (list): List of tuples containing (window_position, result)
        
        Returns:
            dict: Combined result with values from each window preserved
        """
        # Initialize with backward-compatible structure for payslips
        combined = {
            "employee_name": "unknown",
            "gross_amount": "0",
            "net_amount": "0"
        }
        
        # Get confidence threshold
        confidence_threshold = self.config["extraction"].get("confidence_threshold", 0.7)
        fuzzy_matching = self.config["extraction"].get("fuzzy_matching", True)
        
        # For each window, preserve the exact JSON structure returned by the model
        for position, result in window_results:
            # Extract values from result based on position
            key = f"found_in_{position}"
            if key in result:
                data = result[key]
                
                # Check for confidence values (if present)
                confidence = data.get("confidence", 1.0)
                if confidence < confidence_threshold:
                    logger.debug(f"Skipping low confidence result ({confidence}) from {position}")
                    continue
                
                # Add the entire structure directly to the combined result
                combined[key] = data
                
                # For backward compatibility with existing payslip structure:
                # Also update standard fields at the top level if they exist in this window
                if "employee_name" in data and data["employee_name"] != "unknown" and combined["employee_name"] == "unknown":
                    combined["employee_name"] = data["employee_name"]
                
                if "gross_amount" in data and data["gross_amount"] != "0" and combined["gross_amount"] == "0":
                    if fuzzy_matching and not isinstance(data["gross_amount"], str):
                        data["gross_amount"] = str(data["gross_amount"])
                    combined["gross_amount"] = data["gross_amount"]
                
                if "net_amount" in data and data["net_amount"] != "0" and combined["net_amount"] == "0":
                    if fuzzy_matching and not isinstance(data["net_amount"], str):
                        data["net_amount"] = str(data["net_amount"])
                    combined["net_amount"] = data["net_amount"]
        
        return combined

    def _process_pdf_via_api(self, pdf_bytes, pages=None):
        """Process a PDF using the remote API endpoint"""
        import requests
        import base64
        
        url = f"{self.model_endpoint}/process_pdf"
        
        # Prepare payload
        payload = {
            "config": self.config,
            "window_mode": self.window_mode
        }
        
        # Add custom prompts if set
        if self.custom_prompts:
            payload["custom_prompts"] = self.custom_prompts
        
        # Add selected windows if set
        if self.selected_windows:
            payload["selected_windows"] = self.selected_windows
        
        # Add pages parameter if set
        if pages is not None:
            payload["pages"] = pages
        
        # Encode PDF as base64
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        payload["pdf_data"] = b64_pdf
        
        # Call API
        logger.info(f"Calling API endpoint: {url}")
        try:
            response = requests.post(url, json=payload, timeout=300)  # 5-minute timeout
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API error: {response.status_code}, {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def _process_image_via_api(self, image_bytes):
        """Process an image using the remote API endpoint"""
        import requests
        import base64
        
        url = f"{self.model_endpoint}/process_image"
        
        # Prepare payload
        payload = {
            "config": self.config,
            "window_mode": self.window_mode
        }
        
        # Add custom prompts if set
        if self.custom_prompts:
            payload["custom_prompts"] = self.custom_prompts
        
        # Add selected windows if set
        if self.selected_windows:
            payload["selected_windows"] = self.selected_windows
        
        # Encode image as base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload["image_data"] = b64_image
        
        # Call API
        logger.info(f"Calling API endpoint: {url}")
        try:
            response = requests.post(url, json=payload, timeout=300)  # 5-minute timeout
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API error: {response.status_code}, {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error calling API: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
