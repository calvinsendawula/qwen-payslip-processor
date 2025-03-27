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
                 window_mode="vertical",  # "whole", "vertical", "horizontal", "quadrant", "custom"
                 window_regions=None,     # For custom window mode
                 force_cpu=False):
        """Initialize the QwenPayslipProcessor with configuration
        
        Args:
            config (dict): Custom configuration (will be merged with defaults)
            custom_prompts (dict): Custom prompts for different window positions
            window_mode (str): How to split images - "whole", "vertical", "horizontal", "quadrant", "custom"
            window_regions (list): List of regions for custom window mode
            force_cpu (bool): Whether to force CPU usage even if GPU is available
        """
        # Load configuration
        self.config = self._merge_config(config if config else {})
        
        # Set custom prompts if provided
        self.custom_prompts = custom_prompts if custom_prompts else {}
        
        # Set window mode and regions
        self.window_mode = window_mode
        self.window_regions = window_regions
        
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
    
    def _merge_config(self, user_config):
        """Merge user configuration with defaults"""
        default_config = self._get_default_config()
        
        # Deep merge the configurations
        merged_config = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                for k, v in value.items():
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
                "initial_resolution": 1500,
                "resolution_steps": [1500, 1200, 1000, 800, 600],
                "enhance_contrast": True,
                "use_advanced_preprocessing": True,
                "sharpen_factor": 2.5,
                "contrast_factor": 1.8,
                "brightness_factor": 1.1
            },
            "window": {
                "overlap": 0.1
            },
            "text_generation": {
                "max_new_tokens": 768,
                "use_beam_search": False,
                "num_beams": 1,
                "auto_process_results": True
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
    
    def process_pdf(self, pdf_bytes):
        """
        Process a PDF using the Qwen model
        
        Args:
            pdf_bytes (bytes): PDF file content as bytes
            
        Returns:
            dict: Extracted information
        """
        start_time = time.time()
        logger.info("Starting PDF processing")
        
        # Convert PDF to images using PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=self.config["pdf"]["dpi"])
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            logger.info(f"Converted PDF to {len(images)} images")
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            return {"error": f"PDF conversion failed: {str(e)}"}
        
        # Process images
        results = []
        for page_num, image in enumerate(images):
            logger.info(f"Processing page {page_num+1}/{len(images)}")
            
            # Split image based on window mode
            windows = split_image_for_window_mode(
                image, 
                window_mode=self.window_mode,
                window_regions=self.window_regions,
                overlap=self.config["window"]["overlap"]
            )
            
            window_results = []
            # Process each window
            for window_img, window_position in windows:
                result = self._process_window(window_img, window_position)
                window_results.append((window_position, result))
            
            # Combine window results
            combined_result = self._combine_window_results(window_results)
            results.append(combined_result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"PDF processing completed in {processing_time:.2f} seconds")
        
        # Add processing time to results
        final_result = {
            "results": results,
            "processing_time": processing_time
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
            window_regions=self.window_regions,
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
                "net_amount": "Nettogehalt oder '0'"
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
                        do_sample=False,
                        use_cache=True,
                        num_beams=self.config["text_generation"]["num_beams"]
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
        
        for position, result in window_results:
            # Extract values from result based on position
            key = f"found_in_{position}"
            if key in result:
                data = result[key]
                
                # Update employee name if found
                if "employee_name" in data and data["employee_name"] != "unknown" and combined["employee_name"] == "unknown":
                    combined["employee_name"] = data["employee_name"]
                
                # Update gross amount if found
                if "gross_amount" in data and data["gross_amount"] != "0" and combined["gross_amount"] == "0":
                    combined["gross_amount"] = data["gross_amount"]
                
                # Update net amount if found
                if "net_amount" in data and data["net_amount"] != "0" and combined["net_amount"] == "0":
                    combined["net_amount"] = data["net_amount"]
        
        return combined
