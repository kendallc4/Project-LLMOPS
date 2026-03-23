#!/usr/bin/env python3
"""
Enhanced Ollama Proxy with RAG and Vision Support
Supports file uploads, PDF processing, image analysis, and RAG functionality
"""

import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

# Try to import optional packages
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("PyPDF2 not installed. PDF processing will be disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Pillow not installed. Image processing will be limited.")

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"

app = FastAPI(title="Enhanced Ollama Proxy")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Dict[str, Any] = {}
    files: Optional[List[Dict[str, Any]]] = None
    rag_enabled: bool = False

def process_pdf_content(file_content: bytes) -> str:
    """Extract text from PDF content"""
    if not HAS_PDF:
        raise HTTPException(status_code=400, detail="PDF processing not available. Install PyPDF2.")
    
    try:
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def process_image_for_vision(image_data: str) -> str:
    """Process image data for vision models"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        return image_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Enhanced Ollama Proxy is running!"}

@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    """Process uploaded PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        content = await file.read()
        text = process_pdf_content(content)
        
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        return {"text": text, "filename": file.filename}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/api/generate")
async def generate_response(request: ChatRequest):
    """Generate response with optional RAG and vision support"""
    try:
        # Prepare the prompt
        prompt = request.prompt
        images = []
        
        # Process uploaded files if any
        if request.files and request.rag_enabled:
            context_docs = []
            
            for file_data in request.files:
                file_type = file_data.get('type', '')
                file_name = file_data.get('name', 'unknown')
                file_content = file_data.get('content', '')
                
                if file_type == 'image':
                    # For vision models, add image to the request
                    image_data = process_image_for_vision(file_content)
                    images.append(image_data)
                    
                elif file_type in ['pdf', 'text', 'document']:
                    # For text documents, add to context (limit to 2000 chars)
                    if file_content:
                        context_docs.append(file_content[:2000])
            
            # If we have text context, add it to the prompt
            if context_docs:
                context = "\n\n".join(context_docs)
                prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {prompt}

Please provide a comprehensive answer based on the context provided."""

        # Prepare request to Ollama
        ollama_request = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": request.options
        }
        
        # Add images for vision models
        if images:
            ollama_request["images"] = images
        
        # Send request to Ollama
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request,
                timeout=120.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {response.text}"
                )
            
            return response.json()
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/tags")
async def get_models():
    """Get available models from Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30.0)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {response.text}"
                )
            
            return response.json()
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "features": {
            "pdf_processing": HAS_PDF,
            "image_processing": HAS_PIL
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced Ollama Proxy...")
    print("Features available:")
    print(f"  📄 PDF Processing: {'✅' if HAS_PDF else '❌'}")
    print(f"  🖼️ Image Processing: {'✅' if HAS_PIL else '❌'}")
    print("📍 Server: http://localhost:8000")
    print("🦙 Ready for RAG and Vision!")
    print()
    
    # Install missing dependencies reminder
    missing_deps = []
    if not HAS_PDF:
        missing_deps.append("PyPDF2")
    if not HAS_PIL:
        missing_deps.append("Pillow")
    
    if missing_deps:
        print("⚠️  To enable all features, install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
        print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
