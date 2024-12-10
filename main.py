# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import json
import requests
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
from datetime import datetime
import base64
from pdf2image import convert_from_bytes
import tempfile
import io

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
KEYS_OF_INTEREST = ["I.D. No.", "Employee Name", "Date Filed", "Reason For Leave:"]

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str

class DocumentMetadata(BaseModel):
    filename: str
    library: str
    tag_profile: str

@app.post("/api/auth/url")
async def get_auth_url():
    url = f"{ENADOC_BASE_URL}/v3/authorization/url"
    payload = {
        'ClientId': os.getenv('ENADOC_CLIENT_ID'),
        'RedirectUri': 'http://localhost:4200/callback'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f"Bearer {os.getenv('ENADOC_CLIENT_SECRET')}"
    }
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

@app.post("/api/auth/token")
async def get_token(code: str):
    url = f"{ENADOC_BASE_URL}/v3/token"
    payload = {
        'client_secret': os.getenv('ENADOC_CLIENT_SECRET'),
        'client_id': os.getenv('ENADOC_CLIENT_ID'),
        'grant_type': 'authorization_code',
        'code': code
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

@app.get("/api/libraries")
async def get_libraries(token: str):
    headers = {'Authorization': f'bearer {token}'}
    response = requests.get(f"{ENADOC_BASE_URL}/v3/libraries", headers=headers)
    return response.json()

@app.get("/api/tagprofiles")
async def get_tag_profiles(token: str):
    headers = {'Authorization': f'bearer {token}'}
    response = requests.get(f"{ENADOC_BASE_URL}/v3/tagprofiles", headers=headers)
    return response.json()

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    file_content = await file.read()
    metadata_dict = json.loads(metadata)
    
    # Process with Azure Form Recognizer
    try:
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document", document=file_content
        )
        result = poller.result()
        
        extracted_data = {}
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                key = kv_pair.key.content.strip()
                if key in KEYS_OF_INTEREST:
                    extracted_data[key] = {
                        'value': kv_pair.value.content.strip(),
                        'confidence': kv_pair.confidence if hasattr(kv_pair, 'confidence') else None
                    }
        
        # Generate thumbnail
        thumbnail = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file.seek(0)
                images = convert_from_bytes(file_content, size=(400, 565))
                if images:
                    img = images[0]
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    thumbnail = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            print(f"Thumbnail generation error: {str(e)}")
            
        return {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-to-enadoc")
async def upload_to_enadoc(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    extracted_data: str = Form(...),
    token: str = Form(...)
):
    try:
        metadata_dict = json.loads(metadata)
        extracted_data_dict = json.loads(extracted_data)
        
        index_mapping = {
            "I.D. No.": 45,
            "Employee Name": 46,
            "Date Filed": 47,
            "Reason For Leave:": 48
        }
        
        indexes = {
            "documentName": file.filename.replace('.pdf', ''),
            "tagProfileId": int(metadata_dict["tag_profile_id"]),
            "indexes": [
                {"id": index_mapping[field_name], "value": field_value}
                for field_name, field_value in extracted_data_dict.items()
                if field_name in index_mapping
            ]
        }
        
        files = {
            'indexes': ('', json.dumps(indexes), 'application/json'),
            'document': (file.filename, await file.read(), 'application/pdf')
        }
        
        response = requests.post(
            f"{ENADOC_BASE_URL}/v3/documents/?type=simple",
            headers={'Authorization': f'bearer {token}'},
            files=files
        )
        
        if response.status_code != 201:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)