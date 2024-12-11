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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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
document_analysis_client = DocumentAnalysisClient(
    endpoint=os.getenv("AZURE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
)

ENADOC_BASE_URL = "https://qaenadoc.enadocapp.com/api"

# Define field mappings
FIELD_MAPPINGS = {
    "I.D. No.": "ID Number",
    "Employee Name": "Employee Name",
    "Date Filed": "Date Filed",
    "Reason For Leave:": "Reason"
}

KEYS_OF_INTEREST = list(FIELD_MAPPINGS.keys())

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

async def generate_thumbnail(file_content: bytes) -> str:
    """Generate thumbnail asynchronously using ThreadPoolExecutor."""
    def _generate():
        try:
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
                return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}")
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _generate)

async def process_form_recognizer(file_content: bytes) -> Dict:
    """Process document with Form Recognizer and map field names."""
    try:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document",
            document=file_content
        )
        result = poller.result()
        
        extracted_data = {}
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key = kv_pair.key.content.strip()
                if key in KEYS_OF_INTEREST:
                    # Map the original key to the desired output key
                    mapped_key = FIELD_MAPPINGS[key]
                    extracted_data[mapped_key] = {
                        'value': kv_pair.value.content.strip(),
                        'confidence': getattr(kv_pair, 'confidence', None)
                    }
        return extracted_data
    except Exception as e:
        logger.error(f"Form recognizer error: {str(e)}")
        return {}

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process a single document with optimized concurrent processing."""
    try:
        file_content = await file.read()
        metadata_dict = json.loads(metadata)
        
        # Process form recognition and thumbnail generation concurrently
        extracted_data_task = process_form_recognizer(file_content)
        thumbnail_task = generate_thumbnail(file_content)
        
        # Wait for both tasks to complete
        extracted_data, thumbnail = await asyncio.gather(
            extracted_data_task,
            thumbnail_task
        )
        
        return {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
    
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-documents-batch")
async def process_documents_batch(
    files: List[UploadFile] = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process multiple documents concurrently with optimized batch processing."""
    if len(files) > 10:  # Limit batch size for performance
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 10 documents"
        )
    
    try:
        metadata_dict = json.loads(metadata)
        
        async def process_single_document(file: UploadFile):
            try:
                file_content = await file.read()
                
                # Process form recognition and thumbnail generation concurrently
                extracted_data_task = process_form_recognizer(file_content)
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
                logger.error(f"Error processing {file.filename}: {str(e)}")
                return {
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                }
        
        # Process all documents concurrently
        tasks = [process_single_document(file) for file in files]
        results = await asyncio.gather(*tasks)
        
        return {
            "batch_size": len(files),
            "results": results,
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"])
        }
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)