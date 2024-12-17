from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Union
import json
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader, PdfWriter
import io
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import re
from dateutil import parser
import pytz

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

class DateParsingError(Exception):
    """Custom exception for date parsing errors"""
    pass

def clean_date_string(date_str: str) -> str:
    """
    Clean and normalize date string before parsing.
    """
    if not date_str:
        raise DateParsingError("Empty date string")
    
    # Convert to string if not already
    date_str = str(date_str).strip()
    
    # Remove any extra spaces
    date_str = re.sub(r'\s+', ' ', date_str)
    
    # Replace common written months with numeric
    month_replacements = {
        'january': '01', 'jan': '01',
        'february': '02', 'feb': '02',
        'march': '03', 'mar': '03',
        'april': '04', 'apr': '04',
        'may': '05',
        'june': '06', 'jun': '06',
        'july': '07', 'jul': '07',
        'august': '08', 'aug': '08',
        'september': '09', 'sep': '09',
        'october': '10', 'oct': '10',
        'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    lower_date = date_str.lower()
    for month_str, month_num in month_replacements.items():
        if month_str in lower_date:
            lower_date = lower_date.replace(month_str, month_num)
    
    # Remove any non-alphanumeric characters except spaces and common separators
    cleaned = re.sub(r'[^\w\s/-]', '', lower_date)
    
    return cleaned

def standardize_date_format(date_str: str, output_format: str = "%d/%m/%y") -> str:
    """
    Convert various date formats to specified output format (default: dd/mm/yy).
    Handles a wide variety of input formats.
    """
    try:
        # Clean and normalize the date string
        cleaned_date = clean_date_string(date_str)
        
        # Common date formats to try
        date_formats = [
            # Standard formats
            "%d/%m/%y", "%d/%m/%Y",
            "%m/%d/%y", "%m/%d/%Y",
            "%Y-%m-%d", "%d-%m-%Y",
            "%Y/%m/%d", "%d/%m/%Y",
            
            # Formats with month names
            "%d %m %Y", "%m %d %Y",
            "%Y %m %d", "%d %b %Y",
            "%b %d %Y", "%d %B %Y",
            "%B %d %Y",
            
            # Formats with time
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            
            # ISO format
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            
            # Additional formats
            "%Y%m%d", "%d%m%Y",
            "%m%d%Y", "%Y%m%d%H%M%S",
        ]
        
        # Try parsing with explicit formats first
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(cleaned_date, fmt)
                return parsed_date.strftime(output_format)
            except ValueError:
                continue
        
        # If explicit formats fail, try dateutil parser
        try:
            parsed_date = parser.parse(cleaned_date)
            return parsed_date.strftime(output_format)
        except (ValueError, parser.ParserError):
            # Last resort: try to extract dates using regex
            date_patterns = [
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # dd/mm/yyyy or mm/dd/yyyy
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',    # yyyy/mm/dd
                r'(\d{8})',                               # YYYYMMDD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, cleaned_date)
                if match:
                    groups = match.groups()
                    if len(groups[0]) == 4:  # yyyy/mm/dd format
                        try:
                            return datetime(
                                int(groups[0]),
                                int(groups[1]),
                                int(groups[2])
                            ).strftime(output_format)
                        except ValueError:
                            continue
                    else:  # dd/mm/yyyy or mm/dd/yyyy format
                        try:
                            # Assume dd/mm/yyyy
                            return datetime(
                                int(groups[2]) if len(groups[2]) == 4 else 2000 + int(groups[2]),
                                int(groups[1]),
                                int(groups[0])
                            ).strftime(output_format)
                        except ValueError:
                            continue
            
            raise DateParsingError(f"Unable to parse date: {date_str}")
            
    except Exception as e:
        logger.error(f"Date standardization error for '{date_str}': {str(e)}")
        raise DateParsingError(f"Date parsing failed: {str(e)}")

async def extract_first_page(file_content: bytes) -> Optional[str]:
    """Extract first page from PDF and return it as base64 encoded string."""
    def _extract():
        try:
            logger.info(f"Starting PDF first page extraction for document of size {len(file_content)} bytes")
            
            # Create PDF reader object from bytes
            pdf_reader = PdfReader(io.BytesIO(file_content))
            
            if len(pdf_reader.pages) > 0:
                # Create a new PDF writer object
                pdf_writer = PdfWriter()
                
                # Add only the first page
                pdf_writer.add_page(pdf_reader.pages[0])
                
                # Write to bytes buffer
                output_buffer = io.BytesIO()
                pdf_writer.write(output_buffer)
                
                # Get the value and encode to base64
                first_page_pdf = base64.b64encode(output_buffer.getvalue()).decode()
                
                logger.info("Successfully extracted first page from PDF")
                return first_page_pdf
            else:
                logger.warning("PDF document is empty")
                return None
                
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _extract)

async def process_form_recognizer(file_content: bytes, filename: str) -> Dict:
    """Process document with Form Recognizer and map field names."""
    try:
        logger.info(f"Starting Form Recognizer analysis for file: {filename}")
        
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
                    mapped_key = FIELD_MAPPINGS[key]
                    value = kv_pair.value.content.strip()
                    
                    # Apply date standardization for Date Filed field
                    if mapped_key == "Date Filed":
                        try:
                            value = standardize_date_format(value)
                            logger.info(f"Successfully standardized date: {value}")
                        except DateParsingError as e:
                            logger.warning(f"Date parsing failed for {value}: {str(e)}")
                            # Keep original value if parsing fails
                    
                    extracted_data[mapped_key] = {
                        'value': value,
                        'confidence': getattr(kv_pair, 'confidence', None)
                    }
                    logger.info(f"Mapped '{key}' to '{mapped_key}' with value: {extracted_data[mapped_key]}")
        
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
        
        # Process form recognition and first page extraction concurrently
        extracted_data_task = process_form_recognizer(file_content, file.filename)
        first_page_task = extract_first_page(file_content)
        
        # Wait for both tasks to complete
        extracted_data, first_page = await asyncio.gather(
            extracted_data_task,
            first_page_task
        )
        
        response_data = {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "first_page": first_page,
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
                
                # Process form recognition and first page extraction concurrently
                extracted_data_task = process_form_recognizer(file_content, file.filename)
                first_page_task = extract_first_page(file_content)
                
                extracted_data, first_page = await asyncio.gather(
                    extracted_data_task,
                    first_page_task
                )
                
                return {
                    "filename": file.filename,
                    "extracted_data": extracted_data,
                    "first_page": first_page,
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Verify Azure Form Recognizer client
        credential = document_analysis_client.credential
        
        # Test date parsing functionality
        test_dates = [
            "2024-01-01",
            "01/01/24",
            "January 1, 2024"
        ]
        date_parsing_status = all(
            standardize_date_format(date) for date in test_dates
        )
        
        return {
            "status": "healthy",
            "azure_client": "configured",
            "environment": "all required variables set",
            "date_parsing": "operational" if date_parsing_status else "warning"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/api/test-date-parsing/{date_string}")
async def test_date_parsing(date_string: str):
    """Test endpoint for date parsing functionality."""
    try:
        standardized_date = standardize_date_format(date_string)
        return {
            "original": date_string,
            "standardized": standardized_date,
            "status": "success"
        }
    except DateParsingError as e:
        return {
            "original": date_string,
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        # reload=True  # Enable auto-reload during development
    )





    