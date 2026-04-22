#!/usr/bin/env python3
"""
Enhanced Ollama Proxy with RAG, Vision, and Chat History Support
"""

import json
import base64
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

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

# In-memory chat storage
chat_sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Dict[str, Any] = {}
    files: Optional[List[Dict[str, Any]]] = None
    rag_enabled: bool = False
    session_id: Optional[str] = None


def process_pdf_content(file_content: bytes) -> str:
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
    try:
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
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "messages": []
            }

        # Save user message
        chat_sessions[session_id]["messages"].append({
            "role": "user",
            "content": request.prompt
        })

        prompt = request.prompt
        images = []

        # RAG processing
        if request.files and request.rag_enabled:
            context_docs = []

            for file_data in request.files:
                file_type = file_data.get('type', '')
                file_content = file_data.get('content', '')

                if file_type == 'image':
                    images.append(process_image_for_vision(file_content))

                elif file_type in ['pdf', 'text', 'document']:
                    if file_content:
                        context_docs.append(file_content[:2000])

            if context_docs:
                context = "\n\n".join(context_docs)
                prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question:
{request.prompt}

Please provide a comprehensive answer based on the context provided."""

        ollama_request = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": request.options
        }

        if images:
            ollama_request["images"] = images

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

        result = response.json()

        #safe response extraction
        response_text = result.get("response", "") or ""

        chat_sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": response_text
        })

        return {
            "session_id": session_id,
            "response": response_text,
            "done": result.get("done", True)
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/chats")
async def get_chats():
    sorted_chats = sorted(
        chat_sessions.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )

    return [
        {
            "id": chat["id"],
            "preview": chat["messages"][0]["content"][:40] if chat["messages"] else "New Chat",
            "created_at": chat["created_at"]
        }
        for chat in sorted_chats
    ]


@app.get("/api/chats/{session_id}")
async def get_chat(session_id: str):
    return chat_sessions.get(session_id, {"messages": []})

#delete one history 
@app.delete("/api/chats/{session_id}")
async def delete_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "ok"}


#clear all history
@app.delete("/api/chats")
async def delete_all_chats():
    chat_sessions.clear()
    return {"status": "cleared"}


@app.get("/api/tags")
async def get_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30.0)
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "pdf_processing": HAS_PDF,
            "image_processing": HAS_PIL
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
#!/usr/bin/env python3
"""
Enhanced Ollama Proxy with RAG, Vision, and Chat History Support
"""

import json
import base64
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn

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

# In-memory chat storage
chat_sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Dict[str, Any] = {}
    files: Optional[List[Dict[str, Any]]] = None
    rag_enabled: bool = False
    session_id: Optional[str] = None


def process_pdf_content(file_content: bytes) -> str:
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
    try:
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
    try:
        session_id = request.session_id or str(uuid.uuid4())

        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                "messages": []
            }

        # Save user message
        chat_sessions[session_id]["messages"].append({
            "role": "user",
            "content": request.prompt
        })

        prompt = request.prompt
        images = []

        # RAG processing
        if request.files and request.rag_enabled:
            context_docs = []

            for file_data in request.files:
                file_type = file_data.get('type', '')
                file_content = file_data.get('content', '')

                if file_type == 'image':
                    images.append(process_image_for_vision(file_content))

                elif file_type in ['pdf', 'text', 'document']:
                    if file_content:
                        context_docs.append(file_content[:2000])

            if context_docs:
                context = "\n\n".join(context_docs)
                prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question:
{request.prompt}

Please provide a comprehensive answer based on the context provided."""

        ollama_request = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": request.options
        }

        if images:
            ollama_request["images"] = images

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

        result = response.json()

        #safe response extraction
        response_text = result.get("response", "") or ""

        chat_sessions[session_id]["messages"].append({
            "role": "assistant",
            "content": response_text
        })

        return {
            "session_id": session_id,
            "response": response_text,
            "done": result.get("done", True)
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/chats")
async def get_chats():
    sorted_chats = sorted(
        chat_sessions.values(),
        key=lambda x: x["created_at"],
        reverse=True
    )

    return [
        {
            "id": chat["id"],
            "preview": chat["messages"][0]["content"][:40] if chat["messages"] else "New Chat",
            "created_at": chat["created_at"]
        }
        for chat in sorted_chats
    ]


@app.get("/api/chats/{session_id}")
async def get_chat(session_id: str):
    return chat_sessions.get(session_id, {"messages": []})

#delete one history 
@app.delete("/api/chats/{session_id}")
async def delete_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "ok"}


#clear all history
@app.delete("/api/chats")
async def delete_all_chats():
    chat_sessions.clear()
    return {"status": "cleared"}


@app.get("/api/tags")
async def get_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30.0)
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error connecting to Ollama: {str(e)}")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "pdf_processing": HAS_PDF,
            "image_processing": HAS_PIL
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
