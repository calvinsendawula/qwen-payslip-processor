import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
from io import BytesIO
import logging

# Import the processor class
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
    # Initialize the processor with default settings
    processor = QwenPayslipProcessor()
    logger.info("Model loaded and ready to serve requests")

@app.get("/status")
async def get_status():
    """Check if the model server is running."""
    return {
        "status": "ok",
        "model": "Qwen2.5-VL-7B-Instruct",
        "version": "0.1.1",
        "ready": processor is not None
    }

@app.post("/process/pdf")
async def process_pdf(
    file: UploadFile = File(...),
    pages: Optional[str] = Form(None),
    window_mode: Optional[str] = Form(None),
    selected_windows: Optional[str] = Form(None)
):
    """Process a PDF document."""
    global processor
    
    start_time = time.time()
    logger.info(f"Received PDF document: {file.filename}")
    
    # Read the uploaded file
    pdf_bytes = await file.read()
    
    # Parse pages parameter
    parsed_pages = None
    if pages:
        try:
            # Convert comma-separated string to list of integers
            page_list = [int(p.strip()) for p in pages.split(",")]
            parsed_pages = page_list if len(page_list) > 1 else page_list[0]
            logger.info(f"Processing pages: {parsed_pages}")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid page format. Use comma-separated integers.")
    
    # Parse selected_windows parameter
    parsed_windows = None
    if selected_windows:
        window_list = [w.strip() for w in selected_windows.split(",")]
        parsed_windows = window_list if len(window_list) > 1 else window_list[0]
        logger.info(f"Using selected windows: {parsed_windows}")
    
    # Create a custom processor if needed
    custom_processor = processor
    if window_mode or parsed_windows:
        logger.info(f"Creating custom processor with window_mode={window_mode}, selected_windows={parsed_windows}")
        custom_processor = QwenPayslipProcessor(
            window_mode=window_mode if window_mode else processor.window_mode,
            selected_windows=parsed_windows
        )
    
    # Process the PDF
    try:
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
    selected_windows: Optional[str] = Form(None)
):
    """Process an image."""
    global processor
    
    start_time = time.time()
    logger.info(f"Received image: {file.filename}")
    
    # Read the uploaded file
    image_bytes = await file.read()
    
    # Parse selected_windows parameter
    parsed_windows = None
    if selected_windows:
        window_list = [w.strip() for w in selected_windows.split(",")]
        parsed_windows = window_list if len(window_list) > 1 else window_list[0]
        logger.info(f"Using selected windows: {parsed_windows}")
    
    # Create a custom processor if needed
    custom_processor = processor
    if window_mode or parsed_windows:
        logger.info(f"Creating custom processor with window_mode={window_mode}, selected_windows={parsed_windows}")
        custom_processor = QwenPayslipProcessor(
            window_mode=window_mode if window_mode else processor.window_mode,
            selected_windows=parsed_windows
        )
    
    # Process the image
    try:
        logger.info("Processing image...")
        result = custom_processor.process_image(image_bytes)
        # Add processing metrics
        result["api_processing_time"] = time.time() - start_time
        logger.info(f"Image processing completed in {result['api_processing_time']:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 27842))
    logger.info(f"Starting Qwen Payslip Processor API on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info") 