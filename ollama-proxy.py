#!/usr/bin/env python3
"""
Enhanced Ollama Proxy with RAG, Vision, and Chat History Support
"""

import json
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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


OLLAMA_BASE_URL = "http://localhost:11434"

app = FastAPI(title="Enhanced Ollama Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ STORAGE ------------------

chat_sessions: Dict[str, Dict[str, Any]] = {}

# ✅ SIMPLE USER STORAGE
users = {}


# ------------------ MODELS ------------------

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Dict[str, Any] = {}
    files: Optional[List[Dict[str, Any]]] = None
    rag_enabled: bool = False
    session_id: Optional[str] = None


# ✅ AUTH MODELS
class AuthRequest(BaseModel):
    email: str
    password: str


class ChangePasswordRequest(BaseModel):
    email: str
    old_password: str
    new_password: str


# ------------------ AUTH ROUTES ------------------

@app.post("/api/signup")
async def signup(req: AuthRequest):
    users[req.email] = req.password
    return {"success": True}


@app.post("/api/login")
async def login(req: AuthRequest):
    if users.get(req.email) == req.password:
        return {"success": True}
    return {"success": False}


@app.post("/api/change-password")
async def change_password(req: ChangePasswordRequest):
    if users.get(req.email) == req.old_password:
        users[req.email] = req.new_password
        return {"success": True}
    return {"success": False}


# ------------------ ORIGINAL CODE ------------------

def process_pdf_content(file_content: bytes) -> str:
    if not HAS_PDF:
        raise HTTPException(status_code=400, detail="PDF processing not available.")
    
    pdf_file = BytesIO(file_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    return text.strip()


def process_image_for_vision(image_data: str) -> str:
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    return image_data


@app.get("/")
async def root():
    return {"message": "Enhanced Ollama Proxy is running!"}


@app.post("/api/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    text = process_pdf_content(content)

    return {"text": text, "filename": file.filename}


@app.post("/api/generate")
async def generate_response(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": []
        }

    chat_sessions[session_id]["messages"].append({
        "role": "user",
        "content": request.prompt
    })

    prompt = request.prompt
    images = []

    if request.files and request.rag_enabled:
        context_docs = []

        for file_data in request.files:
            if file_data.get('type') == 'image':
                images.append(process_image_for_vision(file_data.get('content')))
            else:
                context_docs.append(file_data.get('content', '')[:2000])

        if context_docs:
            context = "\n\n".join(context_docs)
            prompt = f"Context:\n{context}\n\nQuestion:\n{request.prompt}"

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

    result = response.json()
    response_text = result.get("response", "")

    chat_sessions[session_id]["messages"].append({
        "role": "assistant",
        "content": response_text
    })

    return {
        "session_id": session_id,
        "response": response_text,
        "done": result.get("done", True)
    }


@app.get("/api/chats")
async def get_chats():
    return [
        {
            "id": chat["id"],
            "preview": chat["messages"][0]["content"][:40] if chat["messages"] else "New Chat",
            "created_at": chat["created_at"]
        }
        for chat in chat_sessions.values()
    ]


@app.get("/api/chats/{session_id}")
async def get_chat(session_id: str):
    return chat_sessions.get(session_id, {"messages": []})


@app.delete("/api/chats/{session_id}")
async def delete_chat(session_id: str):
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"status": "ok"}


@app.delete("/api/chats")
async def delete_all_chats():
    chat_sessions.clear()
    return {"status": "cleared"}


@app.get("/api/tags")
async def get_models():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.json()


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
