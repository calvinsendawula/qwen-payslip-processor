import os
import uvicorn
import yaml
import json
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Union, Any
import time
from io import BytesIO
import subprocess
import torch
from qwen_payslip_processor import QwenPayslipProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Payslip Processor API")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the processor on startup
processor = None

@app.on_event("startup")
async def startup_event():
    global processor
    logger.info("Initializing Qwen Payslip Processor...")
    
    try:
        # Set HuggingFace to offline mode to prevent download attempts
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Check if model files exist
        model_path = os.path.join("/app/models", "models--Qwen--Qwen2.5-VL-7B-Instruct")
        if not os.path.exists(model_path):
            logger.error(f"Model files not found at {model_path}. Container may be missing pre-packaged model!")
            raise FileNotFoundError(f"Model files not found at {model_path}")
        
        # Import the processor class - should use cached files
        from qwen_payslip_processor import QwenPayslipProcessor
        
        # Define custom default configuration that matches requirements
        default_config = {
            "pdf": {
                "dpi": 350  # Set default PDF DPI to 350
            },
            "image": {
                "resolution_steps": [900, 600],  # Simplified resolution steps
                "enhance_contrast": True,
                "sharpen_factor": 2.0,
                "contrast_factor": 1.5,
                "brightness_factor": 1.1,
                "ocr_language": "eng",
                "ocr_threshold": 90
            },
            "window": {
                "overlap": 0.1,
                "min_size": 100
            },
            "text_generation": {
                "max_new_tokens": 768,
                "use_beam_search": False,
                "num_beams": 1,
                "temperature": 0.1,
                "top_p": 0.95
            },
            "extraction": {
                "confidence_threshold": 0.7,
                "fuzzy_matching": True
            },
            "global": {
                "mode": "vertical",  # Default to vertical mode
                "selected_windows": ["top", "bottom"]  # Default to processing both top and bottom
            }
        }
        
        # Default prompts for vertical mode (top/bottom)
        default_prompts = {
            "top": """Du siehst die obere Hälfte einer deutschen Gehaltsabrechnung.

SUCHE NACH DIESEN INFORMATIONEN:
1. Namen des Angestellten, meist nach "Herrn/Frau" oder unter "Name"
2. Bruttogehalt ("Gesamt-Brutto") falls in diesem Bereich sichtbar
3. Nettogehalt ("Auszahlungsbetrag") falls in diesem Bereich sichtbar

WICHTIG:
- Bei der Suche nach dem Namen, ignoriere Firmen- oder Versicherungsnamen
- Gib "unknown" für den Namen zurück, wenn du ihn nicht finden kannst
- Gib "0" zurück, wenn Bruttogehalt oder Nettogehalt in diesem Bereich nicht gefunden werden
- Achte auf das korrekte Format: #.###,## (mit Punkt als Tausendertrennzeichen)

Gib deine Funde als JSON zurück:
{
"found_in_top": {
    "employee_name": "Name des Angestellten oder 'unknown'",
    "gross_amount": "Bruttogehalt oder '0'",
    "net_amount": "Nettogehalt oder '0'"
}
}""",
            "bottom": """Du siehst die untere Hälfte einer deutschen Gehaltsabrechnung.

SUCHE NACH DIESEN INFORMATIONEN:
1. Namen des Angestellten falls in diesem Bereich sichtbar
2. Bruttogehalt ("Gesamt-Brutto")
   - WICHTIG: Es gibt möglicherweise zwei "Gesamt-Brutto" Werte!
   - Der korrekte Wert steht meist unter dem Label "Gesamt-Brutto" auf der rechten Seite
   - Ignoriere Werte unter "Verdienstbescheinigung" auf der linken Seite
   - Der Wert sollte im Bereich von 1.000 € bis 10.000 € liegen
3. Nettogehalt ("Auszahlungsbetrag")
   - Meist ganz unten im Dokument
   - Der Wert steht direkt neben dem Label, oft rechts ausgerichtet
   - Typischerweise die letzte Zahl im Dokument
   - Der Wert sollte kleiner als das Bruttogehalt sein

WICHTIG:
- Gib NUR die Werte zurück, die zu diesen spezifischen Labels gehören
- Achte auf das korrekte Format: #.###,## (mit Punkt als Tausendertrennzeichen)
- Gib "unknown" für den Namen zurück, wenn du ihn nicht finden kannst
- Gib "0" für Beträge zurück, wenn du sie nicht sicher identifizieren kannst

Gib deine Funde als JSON zurück:
{
"found_in_bottom": {
    "employee_name": "Name des Angestellten oder 'unknown'",
    "gross_amount": "Bruttogehalt oder '0'",
    "net_amount": "Nettogehalt oder '0'"
}
}"""
        }
        
        # Get force_cpu setting from environment but allow it to be overridden
        # Default to true (prefer CPU) but can be changed via API
        force_cpu_default = os.environ.get("FORCE_CPU", "true").lower() in ("true", "1", "yes")
        
        # Initialize the processor with custom defaults
        processor = QwenPayslipProcessor(
            window_mode="vertical",  # Default to vertical mode
            selected_windows=["top", "bottom"],  # Process both top and bottom
            force_cpu=force_cpu_default,  # Default from env but can be overridden
            memory_isolation="none",  # No memory isolation by default
            custom_prompts=default_prompts,  # Default prompts for vertical mode
            config=default_config  # Default configuration
        )
        
        logger.info("Model loaded and ready to serve requests")
        logger.info(f"Default window mode: vertical, processing top and bottom")
        logger.info(f"Default memory isolation: none")
        logger.info(f"Default force_cpu setting: {force_cpu_default}")
        logger.info(f"Default resolution steps: {default_config['image']['resolution_steps']}")
        logger.info(f"Default PDF DPI: {default_config['pdf']['dpi']}")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {str(e)}")
        logger.exception(e)
        # We'll continue running but the /status endpoint will show not ready

@app.get("/status")
async def get_status():
    """Check if the model server is running."""
    gpu_info = "Not available"
    gpu_enabled = False
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        gpu_enabled = True
        try:
            gpu_info = torch.cuda.get_device_name(0)
        except:
            gpu_info = "Unknown CUDA device"
    
    # Get memory info if available
    memory_info = {}
    if torch.cuda.is_available():
        try:
            memory_info = {
                "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                "allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
                "reserved": torch.cuda.memory_reserved(0) / (1024**3)  # GB
            }
        except:
            memory_info = {"error": "Could not retrieve memory info"}
    
    return {
        "status": "ok",
        "model": "Qwen2.5-VL-7B-Instruct",
        "version": "0.1.4",
        "ready": processor is not None,
        "device": str(next(processor.model.parameters()).device if hasattr(processor, 'model') else "unknown"),
        "gpu": gpu_enabled,
        "gpu_info": gpu_info,
        "memory": memory_info
    }

# Parse page ranges and window selections
def parse_page_range(page_str: Optional[str]) -> Optional[Union[List[int], int]]:
    """Parse page range string to list of integers or single integer"""
    if not page_str:
        return None
        
    result = []
    for part in page_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    
    return result[0] if len(result) == 1 else result

def parse_windows(windows_str: Optional[str]) -> Optional[Union[List[str], str]]:
    """Parse comma-separated window selection into list or single string"""
    if not windows_str:
        return None
        
    windows = [w.strip() for w in windows_str.split(",")]
    return windows[0] if len(windows) == 1 else windows

def parse_config_param(param: Optional[str], param_type=None):
    """Parse and convert configuration parameters"""
    if not param:
        return None
    
    try:
        # For boolean parameters
        if param_type == bool:
            return param.lower() in ['true', 'yes', '1', 't', 'y']
        
        # For integer parameters
        elif param_type == int:
            return int(param)
            
        # For float parameters
        elif param_type == float:
            return float(param)
            
        # For list parameters
        elif param_type == list:
            try:
                # Try JSON format first
                return json.loads(param)
            except:
                # Fallback to comma-separated format
                return [item.strip() for item in param.split(',')]
                
        # Default handling
        else:
            try:
                # Try parsing as JSON
                return json.loads(param)
            except:
                # Return as is if not JSON
                return param
    except Exception as e:
        logger.warning(f"Error parsing parameter {param}: {e}")
        return None

def build_config_from_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a complete configuration dictionary from request parameters"""
    config = {}
    
    # PDF settings
    if any(key.startswith('pdf_') for key in params):
        config["pdf"] = {}
        if 'pdf_dpi' in params:
            config["pdf"]["dpi"] = parse_config_param(params['pdf_dpi'], int)
    
    # Image settings
    if any(key.startswith('image_') for key in params):
        config["image"] = {}
        if 'image_resolution_steps' in params:
            try:
                steps = parse_config_param(params['image_resolution_steps'], list)
                # Handle single value (convert to list)
                if isinstance(steps, (int, str)) and not isinstance(steps, list):
                    steps = [int(steps)]
                elif isinstance(steps, list):
                    # Ensure all items are integers
                    steps = [int(step) if isinstance(step, str) else step for step in steps]
                    # Add additional validation
                    for i, step in enumerate(steps):
                        if not isinstance(step, int):
                            logger.warning(f"Non-integer value found in resolution_steps at position {i}: {step}, converting to int")
                            steps[i] = int(float(step))
                
                logger.info(f"Successfully parsed resolution_steps: {steps}")
                config["image"]["resolution_steps"] = steps
            except Exception as e:
                logger.warning(f"Invalid resolution_steps format: {params['image_resolution_steps']}, error: {str(e)}")
                # Provide a default value to prevent failures
                config["image"]["resolution_steps"] = [600, 400]
                logger.info(f"Using default resolution_steps: {config['image']['resolution_steps']}")
                
        if 'image_enhance_contrast' in params:
            config["image"]["enhance_contrast"] = parse_config_param(params['image_enhance_contrast'], bool)
        if 'image_sharpen_factor' in params:
            config["image"]["sharpen_factor"] = parse_config_param(params['image_sharpen_factor'], float)
        if 'image_contrast_factor' in params:
            config["image"]["contrast_factor"] = parse_config_param(params['image_contrast_factor'], float)
        if 'image_brightness_factor' in params:
            config["image"]["brightness_factor"] = parse_config_param(params['image_brightness_factor'], float)
        if 'image_ocr_language' in params:
            config["image"]["ocr_language"] = params['image_ocr_language']
        if 'image_ocr_threshold' in params:
            config["image"]["ocr_threshold"] = parse_config_param(params['image_ocr_threshold'], int)
    
    # Window settings
    if any(key.startswith('window_') for key in params):
        config["window"] = {}
        if 'window_overlap' in params:
            config["window"]["overlap"] = parse_config_param(params['window_overlap'], float)
        if 'window_min_size' in params:
            config["window"]["min_size"] = parse_config_param(params['window_min_size'], int)
    
    # Text generation settings
    if any(key.startswith('text_generation_') for key in params):
        config["text_generation"] = {}
        if 'text_generation_max_new_tokens' in params:
            config["text_generation"]["max_new_tokens"] = parse_config_param(params['text_generation_max_new_tokens'], int)
        if 'text_generation_use_beam_search' in params:
            config["text_generation"]["use_beam_search"] = parse_config_param(params['text_generation_use_beam_search'], bool)
        if 'text_generation_num_beams' in params:
            config["text_generation"]["num_beams"] = parse_config_param(params['text_generation_num_beams'], int)
        if 'text_generation_temperature' in params:
            config["text_generation"]["temperature"] = parse_config_param(params['text_generation_temperature'], float)
        if 'text_generation_top_p' in params:
            config["text_generation"]["top_p"] = parse_config_param(params['text_generation_top_p'], float)
    
    # Extraction settings
    if any(key.startswith('extraction_') for key in params):
        config["extraction"] = {}
        if 'extraction_confidence_threshold' in params:
            config["extraction"]["confidence_threshold"] = parse_config_param(params['extraction_confidence_threshold'], float)
        if 'extraction_fuzzy_matching' in params:
            config["extraction"]["fuzzy_matching"] = parse_config_param(params['extraction_fuzzy_matching'], bool)
    
    # Global settings
    if any(key.startswith('global_') for key in params):
        config["global"] = {}
        if 'global_mode' in params:
            config["global"]["mode"] = params['global_mode']
        if 'global_prompt' in params:
            config["global"]["prompt"] = params['global_prompt']
        if 'global_selected_windows' in params:
            config["global"]["selected_windows"] = parse_windows(params['global_selected_windows'])
    
    # Page-specific settings
    if 'page_configs' in params and params['page_configs']:
        try:
            page_configs = json.loads(params['page_configs'])
            config["pages"] = page_configs
        except Exception as e:
            logger.warning(f"Error parsing page_configs: {e}")
    
    # Handle custom prompts
    custom_prompts = {}
    for key, value in params.items():
        if key.startswith('prompt_'):
            window_position = key.replace('prompt_', '')
            if window_position in ['top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'whole']:
                custom_prompts[window_position] = value
    
    return config, custom_prompts

@app.post("/process/pdf")
async def process_pdf(
    file: UploadFile = File(...),
    pages: Optional[str] = Form(None),
    window_mode: Optional[str] = Form(None),
    selected_windows: Optional[str] = Form(None),
    memory_isolation: Optional[str] = Form(None),
    force_cpu: Optional[str] = Form(None),
    gpu_memory_fraction: Optional[str] = Form(None),
    
    # PDF settings
    pdf_dpi: Optional[str] = Form(None),
    
    # Image settings
    image_resolution_steps: Optional[str] = Form(None),
    image_enhance_contrast: Optional[str] = Form(None),
    image_sharpen_factor: Optional[str] = Form(None),
    image_contrast_factor: Optional[str] = Form(None),
    image_brightness_factor: Optional[str] = Form(None),
    image_ocr_language: Optional[str] = Form(None),
    image_ocr_threshold: Optional[str] = Form(None),
    
    # Window settings
    window_overlap: Optional[str] = Form(None),
    window_min_size: Optional[str] = Form(None),
    
    # Text generation settings
    text_generation_max_new_tokens: Optional[str] = Form(None),
    text_generation_use_beam_search: Optional[str] = Form(None),
    text_generation_num_beams: Optional[str] = Form(None),
    text_generation_temperature: Optional[str] = Form(None),
    text_generation_top_p: Optional[str] = Form(None),
    
    # Extraction settings
    extraction_confidence_threshold: Optional[str] = Form(None),
    extraction_fuzzy_matching: Optional[str] = Form(None),
    
    # Global settings
    global_mode: Optional[str] = Form(None),
    global_prompt: Optional[str] = Form(None),
    global_selected_windows: Optional[str] = Form(None),
    
    # Page-specific configurations
    page_configs: Optional[str] = Form(None),
    
    # Allow custom prompts for different window positions
    prompt_top: Optional[str] = Form(None),
    prompt_bottom: Optional[str] = Form(None),
    prompt_left: Optional[str] = Form(None),
    prompt_right: Optional[str] = Form(None),
    prompt_top_left: Optional[str] = Form(None),
    prompt_top_right: Optional[str] = Form(None),
    prompt_bottom_left: Optional[str] = Form(None),
    prompt_bottom_right: Optional[str] = Form(None),
    prompt_whole: Optional[str] = Form(None),
    
    # For any additional configuration as a JSON blob
    full_config: Optional[str] = Form(None)
):
    """Process a PDF document with comprehensive parameter control."""
    global processor
    
    start_time = time.time()
    logger.info(f"Received PDF document: {file.filename}")
    
    # Read the uploaded file
    pdf_bytes = await file.read()
    
    # Collect all form parameters in a dictionary
    params = {
        'pages': pages,
        'window_mode': window_mode,
        'selected_windows': selected_windows,
        'memory_isolation': memory_isolation,
        'force_cpu': force_cpu,
        'gpu_memory_fraction': gpu_memory_fraction,
        'pdf_dpi': pdf_dpi,
        'image_resolution_steps': image_resolution_steps,
        'image_enhance_contrast': image_enhance_contrast,
        'image_sharpen_factor': image_sharpen_factor,
        'image_contrast_factor': image_contrast_factor,
        'image_brightness_factor': image_brightness_factor,
        'image_ocr_language': image_ocr_language,
        'image_ocr_threshold': image_ocr_threshold,
        'window_overlap': window_overlap,
        'window_min_size': window_min_size,
        'text_generation_max_new_tokens': text_generation_max_new_tokens,
        'text_generation_use_beam_search': text_generation_use_beam_search,
        'text_generation_num_beams': text_generation_num_beams,
        'text_generation_temperature': text_generation_temperature,
        'text_generation_top_p': text_generation_top_p,
        'extraction_confidence_threshold': extraction_confidence_threshold,
        'extraction_fuzzy_matching': extraction_fuzzy_matching,
        'global_mode': global_mode,
        'global_prompt': global_prompt,
        'global_selected_windows': global_selected_windows,
        'page_configs': page_configs,
        'prompt_top': prompt_top,
        'prompt_bottom': prompt_bottom,
        'prompt_left': prompt_left,
        'prompt_right': prompt_right,
        'prompt_top_left': prompt_top_left,
        'prompt_top_right': prompt_top_right,
        'prompt_bottom_left': prompt_bottom_left,
        'prompt_bottom_right': prompt_bottom_right,
        'prompt_whole': prompt_whole,
        'full_config': full_config
    }
    
    # Parse pages parameter
    parsed_pages = parse_page_range(pages)
    
    # Parse window selections
    parsed_windows = parse_windows(selected_windows)
    
    # Parse force_cpu
    force_cpu_bool = parse_config_param(force_cpu, bool) if force_cpu else None
    
    # Parse memory isolation
    memory_isolation_str = memory_isolation if memory_isolation else None
    
    # Parse GPU memory fraction
    gpu_mem_fraction = parse_config_param(gpu_memory_fraction, float) if gpu_memory_fraction else None
    
    # Build config dictionary from form parameters
    config, custom_prompts = build_config_from_params(params)
    
    # Check for override_global_settings
    override_global = parse_config_param(params.get('override_global_settings'), bool)
    if override_global:
        logger.info("Override global settings flag detected, UI parameters will take precedence over global settings")
        # If we have specific window_mode and selected_windows from UI, they should override global
        if window_mode:
            # Remove any conflicting global settings
            if "global" in config:
                if "mode" in config["global"]:
                    logger.info(f"Overriding global mode '{config['global']['mode']}' with UI mode '{window_mode}'")
                    config["global"]["mode"] = window_mode
        
        # If selected_windows is provided, it should ALWAYS override global settings
        # This ensures user-selected windows are always respected
        selected_windows_param = params.get('selected_windows')
        if selected_windows_param:
            parsed_windows = parse_windows(selected_windows_param)
            if parsed_windows and "global" in config:
                logger.info(f"Overriding global selected windows with UI selection: {parsed_windows}")
                config["global"]["selected_windows"] = parsed_windows
                # Also ensure processing.selected_windows matches to avoid any conflicts
                if "processing" not in config:
                    config["processing"] = {}
                config["processing"]["selected_windows"] = parsed_windows
                logger.info(f"Also setting processing.selected_windows to: {parsed_windows}")
    
    # If full_config is provided, merge it with existing config
    if full_config:
        try:
            logger.info(f"Parsing full_config parameter: {full_config[:200]}{'...' if len(full_config) > 200 else ''}")
            json_config = json.loads(full_config)
            
            # Ensure specific numeric types are correctly handled
            if "image" in json_config and "resolution_steps" in json_config["image"]:
                steps = json_config["image"]["resolution_steps"]
                if not isinstance(steps, list):
                    if isinstance(steps, str) and ',' in steps:
                        # Handle comma-separated string
                        steps = [int(s.strip()) for s in steps.split(',')]
                    else:
                        # Handle single value
                        steps = [int(steps)]
                else:
                    # Ensure all list items are integers
                    steps = [int(s) if isinstance(s, (str, float)) else s for s in steps]
                
                logger.info(f"Processed resolution_steps from full_config: {steps}")
                json_config["image"]["resolution_steps"] = steps
            
            # Deep merge
            for key, value in json_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
                    
            logger.info(f"Successfully merged full_config into config")
        except Exception as e:
            logger.warning(f"Error parsing full_config: {str(e)}")
            logger.warning(f"Full config parsing failed, will use individual parameters instead")
    
    # For debugging: log the configuration
    logger.info(f"Configuration: window_mode={window_mode}, selected_windows={parsed_windows}, memory_isolation={memory_isolation_str}")
    logger.info(f"Full config being used: {config}")
    logger.info(f"Custom prompts: {custom_prompts}")
    
    # Final validation for critical parameters
    if "image" in config and "resolution_steps" in config["image"]:
        try:
            # Ensure resolution_steps is a list of integers
            steps = config["image"]["resolution_steps"]
            if not isinstance(steps, list):
                logger.warning(f"resolution_steps is not a list, converting: {steps}")
                if isinstance(steps, (int, float)):
                    steps = [int(steps)]
                elif isinstance(steps, str):
                    steps = [int(s.strip()) for s in steps.split(',')]
                else:
                    steps = [600, 400]  # Default fallback
            else:
                # Convert any non-integer values to integers
                for i, step in enumerate(steps):
                    if not isinstance(step, int):
                        logger.warning(f"Non-integer in resolution_steps[{i}]: {step}, converting")
                        try:
                            steps[i] = int(float(step))
                        except:
                            steps[i] = 600  # Default fallback for this position
            
            # Update the config with validated steps
            config["image"]["resolution_steps"] = steps
            logger.info(f"Final validated resolution_steps: {steps}")
        except Exception as e:
            logger.warning(f"Error validating resolution_steps: {str(e)}, using defaults")
            config["image"]["resolution_steps"] = [600, 400]  # Safe default
    
    # Create a custom processor with all parameters
    try:
        logger.info("Creating custom processor with provided parameters")
        custom_processor = QwenPayslipProcessor(
            window_mode=window_mode if window_mode else processor.window_mode,
            selected_windows=parsed_windows if parsed_windows else processor.selected_windows,
            force_cpu=force_cpu_bool if force_cpu_bool is not None else processor.force_cpu,
            memory_isolation=memory_isolation_str if memory_isolation_str else processor.memory_isolation,
            custom_prompts=custom_prompts if custom_prompts else None,
            config=config if config else None
        )
    
        # Set GPU memory fraction if provided
        if gpu_mem_fraction is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            logger.info(f"Setting GPU memory fraction to {gpu_mem_fraction}")
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
        
    # Process the PDF
        logger.info("Processing PDF document...")
        result = custom_processor.process_pdf(pdf_bytes, pages=parsed_pages)
        
        # Add processing metrics
        result["api_processing_time"] = time.time() - start_time
        logger.info(f"PDF processing completed in {result['api_processing_time']:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/image")
async def process_image(
    file: UploadFile = File(...),
    window_mode: Optional[str] = Form(None),
    selected_windows: Optional[str] = Form(None),
    memory_isolation: Optional[str] = Form(None),
    force_cpu: Optional[str] = Form(None),
    gpu_memory_fraction: Optional[str] = Form(None),
    
    # Image settings
    image_resolution_steps: Optional[str] = Form(None),
    image_enhance_contrast: Optional[str] = Form(None),
    image_sharpen_factor: Optional[str] = Form(None),
    image_contrast_factor: Optional[str] = Form(None),
    image_brightness_factor: Optional[str] = Form(None),
    image_ocr_language: Optional[str] = Form(None),
    image_ocr_threshold: Optional[str] = Form(None),
    
    # Window settings
    window_overlap: Optional[str] = Form(None),
    window_min_size: Optional[str] = Form(None),
    
    # Text generation settings
    text_generation_max_new_tokens: Optional[str] = Form(None),
    text_generation_use_beam_search: Optional[str] = Form(None),
    text_generation_num_beams: Optional[str] = Form(None),
    text_generation_temperature: Optional[str] = Form(None),
    text_generation_top_p: Optional[str] = Form(None),
    
    # Extraction settings
    extraction_confidence_threshold: Optional[str] = Form(None),
    extraction_fuzzy_matching: Optional[str] = Form(None),
    
    # Allow custom prompts for different window positions
    prompt_top: Optional[str] = Form(None),
    prompt_bottom: Optional[str] = Form(None),
    prompt_left: Optional[str] = Form(None),
    prompt_right: Optional[str] = Form(None),
    prompt_top_left: Optional[str] = Form(None),
    prompt_top_right: Optional[str] = Form(None),
    prompt_bottom_left: Optional[str] = Form(None),
    prompt_bottom_right: Optional[str] = Form(None),
    prompt_whole: Optional[str] = Form(None),
    
    # For any additional configuration as a JSON blob
    full_config: Optional[str] = Form(None)
):
    """Process an image with comprehensive parameter control."""
    global processor
    
    start_time = time.time()
    logger.info(f"Received image: {file.filename}")
    
    # Read the uploaded file
    image_bytes = await file.read()
    
    # Collect all form parameters in a dictionary
    params = {
        'window_mode': window_mode,
        'selected_windows': selected_windows,
        'memory_isolation': memory_isolation,
        'force_cpu': force_cpu,
        'gpu_memory_fraction': gpu_memory_fraction,
        'image_resolution_steps': image_resolution_steps,
        'image_enhance_contrast': image_enhance_contrast,
        'image_sharpen_factor': image_sharpen_factor,
        'image_contrast_factor': image_contrast_factor,
        'image_brightness_factor': image_brightness_factor,
        'image_ocr_language': image_ocr_language,
        'image_ocr_threshold': image_ocr_threshold,
        'window_overlap': window_overlap,
        'window_min_size': window_min_size,
        'text_generation_max_new_tokens': text_generation_max_new_tokens,
        'text_generation_use_beam_search': text_generation_use_beam_search,
        'text_generation_num_beams': text_generation_num_beams,
        'text_generation_temperature': text_generation_temperature,
        'text_generation_top_p': text_generation_top_p,
        'extraction_confidence_threshold': extraction_confidence_threshold,
        'extraction_fuzzy_matching': extraction_fuzzy_matching,
        'prompt_top': prompt_top,
        'prompt_bottom': prompt_bottom,
        'prompt_left': prompt_left,
        'prompt_right': prompt_right,
        'prompt_top_left': prompt_top_left,
        'prompt_top_right': prompt_top_right,
        'prompt_bottom_left': prompt_bottom_left,
        'prompt_bottom_right': prompt_bottom_right,
        'prompt_whole': prompt_whole,
        'full_config': full_config
    }
    
    # Parse parameters
    parsed_windows = parse_windows(selected_windows)
    force_cpu_bool = parse_config_param(force_cpu, bool) if force_cpu else None
    memory_isolation_str = memory_isolation if memory_isolation else None
    gpu_mem_fraction = parse_config_param(gpu_memory_fraction, float) if gpu_memory_fraction else None
    
    # Build config dictionary from form parameters
    config, custom_prompts = build_config_from_params(params)
    
    # Check for override_global_settings
    override_global = parse_config_param(params.get('override_global_settings'), bool)
    if override_global:
        logger.info("Override global settings flag detected, UI parameters will take precedence over global settings")
        # If we have specific window_mode and selected_windows from UI, they should override global
        if window_mode:
            # Remove any conflicting global settings
            if "global" in config:
                if "mode" in config["global"]:
                    logger.info(f"Overriding global mode '{config['global']['mode']}' with UI mode '{window_mode}'")
                    config["global"]["mode"] = window_mode
        
        # If selected_windows is provided, it should ALWAYS override global settings
        # This ensures user-selected windows are always respected
        selected_windows_param = params.get('selected_windows')
        if selected_windows_param:
            parsed_windows = parse_windows(selected_windows_param)
            if parsed_windows and "global" in config:
                logger.info(f"Overriding global selected windows with UI selection: {parsed_windows}")
                config["global"]["selected_windows"] = parsed_windows
    
    # If full_config is provided, merge it with existing config
    if full_config:
        try:
            logger.info(f"Parsing full_config parameter: {full_config[:200]}{'...' if len(full_config) > 200 else ''}")
            json_config = json.loads(full_config)
            
            # Ensure specific numeric types are correctly handled
            if "image" in json_config and "resolution_steps" in json_config["image"]:
                steps = json_config["image"]["resolution_steps"]
                if not isinstance(steps, list):
                    if isinstance(steps, str) and ',' in steps:
                        # Handle comma-separated string
                        steps = [int(s.strip()) for s in steps.split(',')]
                    else:
                        # Handle single value
                        steps = [int(steps)]
                else:
                    # Ensure all list items are integers
                    steps = [int(s) if isinstance(s, (str, float)) else s for s in steps]
                
                logger.info(f"Processed resolution_steps from full_config: {steps}")
                json_config["image"]["resolution_steps"] = steps
            
            # Deep merge
            for key, value in json_config.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
                    
            logger.info(f"Successfully merged full_config into config")
        except Exception as e:
            logger.warning(f"Error parsing full_config: {str(e)}")
            logger.warning(f"Full config parsing failed, will use individual parameters instead")
    
    # For debugging: log the configuration
    logger.info(f"Configuration: window_mode={window_mode}, selected_windows={parsed_windows}, memory_isolation={memory_isolation_str}")
    logger.info(f"Full config being used: {config}")
    logger.info(f"Custom prompts: {custom_prompts}")
    
    # Final validation for critical parameters
    if "image" in config and "resolution_steps" in config["image"]:
        try:
            # Ensure resolution_steps is a list of integers
            steps = config["image"]["resolution_steps"]
            if not isinstance(steps, list):
                logger.warning(f"resolution_steps is not a list, converting: {steps}")
                if isinstance(steps, (int, float)):
                    steps = [int(steps)]
                elif isinstance(steps, str):
                    steps = [int(s.strip()) for s in steps.split(',')]
                else:
                    steps = [600, 400]  # Default fallback
            else:
                # Convert any non-integer values to integers
                for i, step in enumerate(steps):
                    if not isinstance(step, int):
                        logger.warning(f"Non-integer in resolution_steps[{i}]: {step}, converting")
                        try:
                            steps[i] = int(float(step))
                        except:
                            steps[i] = 600  # Default fallback for this position
            
            # Update the config with validated steps
            config["image"]["resolution_steps"] = steps
            logger.info(f"Final validated resolution_steps: {steps}")
        except Exception as e:
            logger.warning(f"Error validating resolution_steps: {str(e)}, using defaults")
            config["image"]["resolution_steps"] = [600, 400]  # Safe default
    
    # Create a custom processor with all parameters
    try:
        logger.info("Creating custom processor with provided parameters")
        custom_processor = QwenPayslipProcessor(
            window_mode=window_mode if window_mode else processor.window_mode,
            selected_windows=parsed_windows if parsed_windows else processor.selected_windows,
            force_cpu=force_cpu_bool if force_cpu_bool is not None else processor.force_cpu,
            memory_isolation=memory_isolation_str if memory_isolation_str else processor.memory_isolation,
            custom_prompts=custom_prompts if custom_prompts else None,
            config=config if config else None
        )
    
        # Set GPU memory fraction if provided
        if gpu_mem_fraction is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            logger.info(f"Setting GPU memory fraction to {gpu_mem_fraction}")
            torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction)
        
    # Process the image
        logger.info("Processing image...")
        result = custom_processor.process_image(image_bytes)
        
        # Add processing metrics
        result["api_processing_time"] = time.time() - start_time
        logger.info(f"Image processing completed in {result['api_processing_time']:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/config")
async def save_config(request: Request):
    """Save a configuration for later use."""
    try:
        config_data = await request.json()
        config_name = config_data.get('name', 'default')
        
        # Create configs directory if it doesn't exist
        os.makedirs('configs', exist_ok=True)
        
        # Save config as YAML file
        with open(f'configs/{config_name}.yml', 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        return {"status": "success", "message": f"Configuration saved as {config_name}.yml"}
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving configuration: {str(e)}")

@app.get("/configs")
async def list_configs():
    """List available configurations."""
    try:
        # Check if configs directory exists
        if not os.path.exists('configs'):
            return {"configs": []}
        
        # List YAML files in the configs directory
        configs = [f[:-4] for f in os.listdir('configs') if f.endswith('.yml')]
        return {"configs": configs}
    except Exception as e:
        logger.error(f"Error listing configurations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing configurations: {str(e)}")

@app.get("/config/{name}")
async def get_config(name: str):
    """Get a stored configuration by name."""
    try:
        # Check if config file exists
        config_path = f'configs/{name}.yml'
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail=f"Configuration {name} not found")
        
        # Load the config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

@app.delete("/config/{name}")
async def delete_config(name: str):
    """Delete a stored configuration."""
    try:
        # Check if config file exists
        config_path = f'configs/{name}.yml'
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail=f"Configuration {name} not found")
        
        # Delete the file
        os.remove(config_path)
        
        return {"status": "success", "message": f"Configuration {name} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting configuration: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 27842))
    logger.info(f"Starting Qwen Payslip Processor API on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info") 