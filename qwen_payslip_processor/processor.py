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

from .utils import (
    optimize_image_for_vl_model,
    split_image_for_window_mode,
    cleanup_memory,
    extract_json_from_text
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenPayslipProcessor:
    """Processes payslips using Qwen2.5-VL-7B vision-language model with customizable window approach"""
    
    def __init__(self, 
                 config=None,
                 custom_prompts=None,
                 window_mode="vertical",  # "whole", "vertical", "horizontal", "quadrant"
                 selected_windows=None,   # List of windows to process, e.g. ["top", "bottom_right"]
                 force_cpu=False,
                 model_endpoint=None):    # API endpoint for Docker-based processing
        """Initialize the QwenPayslipProcessor with configuration
        
        Args:
            config (dict): Custom configuration (will be merged with defaults)
            custom_prompts (dict): Custom prompts for different window positions
            window_mode (str): How to split images - "whole", "vertical", "horizontal", "quadrant"
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
        """
        # Store the model endpoint if provided
        self.model_endpoint = model_endpoint
        
        # Load configuration
        self.config = self._merge_config(config if config else {})
        
        # Set custom prompts if provided
        self.custom_prompts = custom_prompts if custom_prompts else {}
        
        # Set window mode and regions
        self.window_mode = window_mode
        
        # Handle selected_windows as either list or string
        if selected_windows is not None:
            if isinstance(selected_windows, str):
                self.selected_windows = [selected_windows]
            else:
                self.selected_windows = selected_windows
        else:
            self.selected_windows = None
        
        # Only load the model locally if no model_endpoint is provided
        if not model_endpoint:
            # Set device based on user preference
            if force_cpu:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Using device: {self.device}")
            
            # Configure PyTorch for better memory management
            if torch.cuda.is_available():
                # Enable memory optimizations
                torch.cuda.empty_cache()
                # Set PyTorch to release memory more aggressively
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Load model and processor
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
            else:
                # Override with user value
                merged_config[key] = value
        
        return merged_config
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
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
            }
        }
    
    def _load_model(self):
        """Load the Qwen2.5-VL-7B model and processor from the package or download if needed"""
        try:
            logger.info("Loading Qwen2.5-VL-7B-Instruct model...")
            
            # Define paths
            package_dir = Path(__file__).parent.absolute()
            model_dir = os.path.join(package_dir, "model_files")
            model_path = os.path.join(model_dir, "model")
            processor_path = os.path.join(model_dir, "processor")
            
            # Create model directory if it doesn't exist
            os.makedirs(model_dir, exist_ok=True)
            
            # Check if model files exist
            model_exists = os.path.exists(model_path) and os.path.isdir(model_path) and os.listdir(model_path)
            processor_exists = os.path.exists(processor_path) and os.path.isdir(processor_path) and os.listdir(processor_path)
            
            if not model_exists or not processor_exists:
                logger.info("Model files not found. Downloading them now (this will take some time)...")
                
                # Download processor
                processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
                processor.save_pretrained(processor_path)
                logger.info("Processor downloaded and saved successfully!")
                
                # Download model
                model = AutoModelForImageTextToText.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    torch_dtype=torch.float16
                )
                model.save_pretrained(model_path)
                logger.info("Model downloaded and saved successfully!")
                
                # Create marker file
                with open(os.path.join(model_dir, "MODEL_READY"), "w") as f:
                    f.write("Model downloaded and ready")
            else:
                logger.info("Model files found in local cache.")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                processor_path,
                local_files_only=True  # Force use of local files
            )
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # Use half precision for memory efficiency
                device_map="auto",  # Automatically place model on available devices
                local_files_only=True  # Force use of local files
            )
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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
            
            # Split image based on window mode
            windows = split_image_for_window_mode(
                image, 
                window_mode=self.window_mode,
                overlap=self.config["window"]["overlap"]
            )
            
            # Filter windows based on selected_windows
            if self.selected_windows:
                filtered_windows = []
                for window_img, window_position in windows:
                    if window_position in self.selected_windows:
                        filtered_windows.append((window_img, window_position))
                windows = filtered_windows
                if not windows:
                    logger.warning(f"No windows selected for processing. Using all windows.")
                    windows = split_image_for_window_mode(
                        image, 
                        window_mode=self.window_mode,
                        overlap=self.config["window"]["overlap"]
                    )
            
            window_results = []
            # Process each window
            for window_img, window_position in windows:
                result = self._process_window(window_img, window_position)
                window_results.append((window_position, result))
            
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
            "processed_pages": len(pages_to_process)
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
        
        # Convert bytes to PIL Image
        try:
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"Loaded image: {image.width}x{image.height}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return {"error": f"Image loading failed: {str(e)}"}
        
        # Split image based on window mode
        windows = split_image_for_window_mode(
            image, 
            window_mode=self.window_mode,
            overlap=self.config["window"]["overlap"]
        )
        
        # Filter windows based on selected_windows
        if self.selected_windows:
            filtered_windows = []
            for window_img, window_position in windows:
                if window_position in self.selected_windows:
                    filtered_windows.append((window_img, window_position))
            windows = filtered_windows
            if not windows:
                logger.warning(f"No windows selected for processing. Using all windows.")
                windows = split_image_for_window_mode(
                    image, 
                    window_mode=self.window_mode,
                    overlap=self.config["window"]["overlap"]
                )
        
        window_results = []
        # Process each window
        for window_img, window_position in windows:
            result = self._process_window(window_img, window_position)
            window_results.append((window_position, result))
        
        # Combine window results
        combined_result = self._combine_window_results(window_results)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Image processing completed in {processing_time:.2f} seconds")
        
        # Add processing time to results
        final_result = {
            "results": [combined_result],  # Keep format consistent with PDF processing
            "processing_time": processing_time
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
    
    def _process_window(self, window, window_position):
        """Process a window with the model, trying different resolutions"""
        # Clean up memory before processing
        cleanup_memory()
        
        prompt_text = self._get_prompt_for_position(window_position)
        
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
        """Return an empty result structure based on window position"""
        # Format: found_in_<position>: {employee_name, gross_amount, net_amount}
        return {
            f"found_in_{window_position}": {
                "employee_name": "unknown",
                "gross_amount": "0",
                "net_amount": "0"
            }
        }
    
    def _combine_window_results(self, window_results):
        """Combine results from multiple windows
        
        Args:
            window_results (list): List of tuples containing (window_position, result)
        
        Returns:
            dict: Combined result with best values from each window
        """
        combined = {
            "employee_name": "unknown",
            "gross_amount": "0",
            "net_amount": "0"
        }
        
        # Get confidence threshold
        confidence_threshold = self.config["extraction"].get("confidence_threshold", 0.7)
        fuzzy_matching = self.config["extraction"].get("fuzzy_matching", True)
        
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
                
                # Update employee name if found
                if "employee_name" in data and data["employee_name"] != "unknown" and combined["employee_name"] == "unknown":
                    combined["employee_name"] = data["employee_name"]
                
                # Update gross amount if found
                if "gross_amount" in data and data["gross_amount"] != "0" and combined["gross_amount"] == "0":
                    # Optional: Clean the gross amount format if fuzzy matching is enabled
                    if fuzzy_matching and not isinstance(data["gross_amount"], str):
                        data["gross_amount"] = str(data["gross_amount"])
                    combined["gross_amount"] = data["gross_amount"]
                
                # Update net amount if found
                if "net_amount" in data and data["net_amount"] != "0" and combined["net_amount"] == "0":
                    # Optional: Clean the net amount format if fuzzy matching is enabled
                    if fuzzy_matching and not isinstance(data["net_amount"], str):
                        data["net_amount"] = str(data["net_amount"])
                    combined["net_amount"] = data["net_amount"]
        
        return combined

    def _process_pdf_via_api(self, pdf_bytes, pages=None):
        """Process a PDF by sending it to the remote API endpoint
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            pages (list or int, optional): Page numbers to process
            
        Returns:
            dict: Extracted information from the API
        """
        import requests
        
        start_time = time.time()
        logger.info(f"Sending PDF to API endpoint: {self.model_endpoint}/process/pdf")
        
        # Prepare the multipart/form-data request
        files = {
            'file': ('document.pdf', pdf_bytes, 'application/pdf')
        }
        
        # Prepare optional parameters
        data = {}
        
        # Convert pages parameter
        if pages is not None:
            if isinstance(pages, list):
                data['pages'] = ','.join(str(p) for p in pages)
            else:
                data['pages'] = str(pages)
        
        # Add window mode if specified
        if self.window_mode:
            data['window_mode'] = self.window_mode
        
        # Add selected windows if specified
        if self.selected_windows:
            if isinstance(self.selected_windows, list):
                data['selected_windows'] = ','.join(self.selected_windows)
            else:
                data['selected_windows'] = self.selected_windows
        
        # Send the request
        try:
            response = requests.post(
                f"{self.model_endpoint}/process/pdf",
                files=files,
                data=data
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            logger.info(f"PDF processing via API completed in {time.time() - start_time:.2f} seconds")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "error": f"API request failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def _process_image_via_api(self, image_bytes):
        """Process an image by sending it to the remote API endpoint
        
        Args:
            image_bytes (bytes): Image file content as bytes
            
        Returns:
            dict: Extracted information from the API
        """
        import requests
        
        start_time = time.time()
        logger.info(f"Sending image to API endpoint: {self.model_endpoint}/process/image")
        
        # Prepare the multipart/form-data request
        files = {
            'file': ('image.jpg', image_bytes, 'image/jpeg')
        }
        
        # Prepare optional parameters
        data = {}
        
        # Add window mode if specified
        if self.window_mode:
            data['window_mode'] = self.window_mode
        
        # Add selected windows if specified
        if self.selected_windows:
            if isinstance(self.selected_windows, list):
                data['selected_windows'] = ','.join(self.selected_windows)
            else:
                data['selected_windows'] = self.selected_windows
        
        # Send the request
        try:
            response = requests.post(
                f"{self.model_endpoint}/process/image",
                files=files,
                data=data
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the JSON response
            result = response.json()
            
            logger.info(f"Image processing via API completed in {time.time() - start_time:.2f} seconds")
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return {
                "error": f"API request failed: {str(e)}",
                "processing_time": time.time() - start_time
            }
