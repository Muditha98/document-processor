from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import json
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
import base64
from pdf2image import convert_from_bytes
import io
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('document_processor.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
required_env_vars = ["AZURE_ENDPOINT", "AZURE_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Azure Endpoint: {os.getenv('AZURE_ENDPOINT')}")
logger.info(f"Azure Key is {'set' if os.getenv('AZURE_KEY') else 'not set'}")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Configuration
try:
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
    )
    logger.info("Successfully initialized Document Analysis Client")
except Exception as e:
    logger.error(f"Failed to initialize Document Analysis Client: {str(e)}")
    raise

ENADOC_BASE_URL = "https://qaenadoc.enadocapp.com/api"

# Define field mappings
FIELD_MAPPINGS = {
    "I.D. No.": "ID Number",
    "Employee Name": "Employee Name",
    "Date Filed": "Date Filed",
    "Reason For Leave:": "Reason"
}

KEYS_OF_INTEREST = list(FIELD_MAPPINGS.keys())
logger.info(f"Configured field mappings: {FIELD_MAPPINGS}")

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

async def generate_thumbnail(file_content: bytes) -> str:
    """Generate thumbnail asynchronously using ThreadPoolExecutor."""
    def _generate():
        try:
            logger.info(f"Starting thumbnail generation for document of size {len(file_content)} bytes")
            images = convert_from_bytes(
                file_content,
                size=(400, 565),
                first_page=1,
                last_page=1,
                thread_count=2
            )
            if images:
                buffered = io.BytesIO()
                images[0].save(buffered, format="PNG", optimize=True, quality=85)
                thumbnail = base64.b64encode(buffered.getvalue()).decode()
                logger.info("Successfully generated thumbnail")
                return thumbnail
            else:
                logger.warning("No images were generated from the PDF")
                return None
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}", exc_info=True)
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _generate)

async def process_form_recognizer(file_content: bytes, filename: str) -> Dict:
    """Process document with Form Recognizer and map field names."""
    try:
        logger.info(f"Starting Form Recognizer analysis for file: {filename}")
        logger.info(f"Document size: {len(file_content)} bytes")

        # Begin document analysis
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document",
            document=file_content
        )
        logger.info("Document analysis initiated")
        
        # Get the result
        result = poller.result()
        logger.info("Document analysis completed")

        # Log all found key-value pairs for debugging
        logger.info(f"Raw key-value pairs found in {filename}:")
        for kv_pair in result.key_value_pairs:
            if kv_pair.key:
                logger.info(f"Found key: '{kv_pair.key.content.strip()}' with value: "
                          f"'{kv_pair.value.content if kv_pair.value else 'None'}'")

        # Process and map the fields
        extracted_data = {}
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key = kv_pair.key.content.strip()
                if key in KEYS_OF_INTEREST:
                    mapped_key = FIELD_MAPPINGS[key]
                    extracted_data[mapped_key] = {
                        'value': kv_pair.value.content.strip(),
                        'confidence': getattr(kv_pair, 'confidence', None)
                    }
                    logger.info(f"Mapped '{key}' to '{mapped_key}' with value: {extracted_data[mapped_key]}")

        if not extracted_data:
            logger.warning(f"No matching fields found in {filename}")
            logger.info(f"Expected keys: {KEYS_OF_INTEREST}")
            
        return extracted_data

    except Exception as e:
        logger.error(f"Form recognizer error for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Form recognizer processing failed: {str(e)}"
        )

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process a single document with optimized concurrent processing."""
    try:
        logger.info(f"Processing single document: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        
        file_content = await file.read()
        logger.info(f"File size: {len(file_content)} bytes")
        
        metadata_dict = json.loads(metadata)
        logger.info(f"Metadata: {metadata_dict}")
        
        # Process form recognition and thumbnail generation concurrently
        extracted_data_task = process_form_recognizer(file_content, file.filename)
        thumbnail_task = generate_thumbnail(file_content)
        
        # Wait for both tasks to complete
        extracted_data, thumbnail = await asyncio.gather(
            extracted_data_task,
            thumbnail_task
        )
        
        response_data = {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
        
        logger.info(f"Successfully processed document: {file.filename}")
        logger.info(f"Extracted fields: {list(extracted_data.keys())}")
        
        return response_data
    
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-documents-batch")
async def process_documents_batch(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process multiple documents concurrently with optimized batch processing."""
    if len(files) > 10:
        logger.warning(f"Batch size {len(files)} exceeds maximum limit of 10")
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 10 documents"
        )
    
    try:
        logger.info(f"Processing batch of {len(files)} documents")
        metadata_dict = json.loads(metadata)
        
        async def process_single_document(file: UploadFile):
            try:
                logger.info(f"Processing batch document: {file.filename}")
                file_content = await file.read()
                
                # Process form recognition and thumbnail generation concurrently
                extracted_data_task = process_form_recognizer(file_content, file.filename)
                thumbnail_task = generate_thumbnail(file_content)
                
                extracted_data, thumbnail = await asyncio.gather(
                    extracted_data_task,
                    thumbnail_task
                )
                
                return {
                    "filename": file.filename,
                    "extracted_data": extracted_data,
                    "thumbnail": thumbnail,
                    "metadata": metadata_dict,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
                return {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }
        
        # Process all documents concurrently
        tasks = [process_single_document(file) for file in files]
        results = await asyncio.gather(*tasks)
        
        successful = len([r for r in results if r.get("status") == "success"])
        failed = len([r for r in results if r.get("status") == "error"])
        
        logger.info(f"Batch processing completed. Successful: {successful}, Failed: {failed}")
        
        return {
            "batch_size": len(files),
            "results": results,
            "successful": successful,
            "failed": failed
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Verify Azure Form Recognizer client
        credential = document_analysis_client.credential
        return {
            "status": "healthy",
            "azure_client": "configured",
            "environment": "all required variables set"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)















