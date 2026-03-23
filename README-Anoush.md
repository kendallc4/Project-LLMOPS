# 🦙 Ollama Chat with Vision & Multi-Model Support

A versatile web-based chat interface for Ollama featuring specialized models for different tasks: vision analysis, coding assistance, and general conversation, with document analysis (RAG) support.

## ✨ Features

- 💬 **Multi-Model Chat**: Switch between specialized models for different tasks
- 🖼️ **LLaVA**: Analyze images, read visual content, describe pictures
- 💻 **Qwen2.5-Coder**: Advanced coding assistance, debugging, code review
- 🐬 **Dolphin Mistral**: General conversation, creative writing, Q&A
- 📄 Upload and analyze PDF documents 
- 🧠 RAG (Retrieval-Augmented Generation) for document Q&A
- 🎯 Clean, intuitive web interface

## 🚀 Quick Start

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Download Models

```bash
# Download LLaVA for vision tasks
ollama pull llava:latest

# Download Qwen2.5-Coder for programming assistance  
ollama pull qwen2.5-coder:7b

# Download Dolphin Mistral for general conversation
ollama pull dolphin-mistral:latest
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv ollama-env

# Activate it
source ollama-env/bin/activate

# Install required packages
pip install fastapi uvicorn httpx PyPDF2 Pillow chromadb sentence-transformers python-multipart
```

### 4. Download the Files

1. Download `ollama-chat.html` and `enhanced-proxy.py`
2. Save them in the same folder

### 5. Run the Application

```bash
# Make sure virtual environment is active
source ollama-env/bin/activate

# Start the proxy server
python enhanced-proxy.py
```

### 6. Open the Interface

Open `ollama-chat.html` in your web browser.

## 📋 Requirements

- **Python 3.8+**
- **Ollama** installed and running
- **Ubuntu/Debian**: Install `python3-venv` first:
  ```bash
  sudo apt install python3-venv
  ```

## 🎯 Usage

### **🖼️ Vision Tasks (LLaVA)**
- Upload an image using 🖼️ button
- Ask: "What do you see?" "Read the text in this image" "Describe this picture"

### **💻 Coding Tasks (Qwen2.5-Coder)**  
- Ask: "Write a Python function to..." "Review this code" "Debug this error"
- Upload code files as PDFs for analysis

### **💬 General Chat (Dolphin Mistral)**
- Ask: "Explain quantum physics" "Write a story" "Help me plan my day"
- General Q&A and conversation

### **📄 Document Analysis (All Models)**
- Upload PDF using 📁 button
- Make sure "RAG Mode Active" is checked
- Ask questions about the document content

## 🐛 Troubleshooting

**Connection Error**: Make sure both Ollama and the proxy are running
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start proxy (in virtual environment)
source ollama-env/bin/activate
python enhanced-proxy.py
```

**No Models Error**: Download the required models first
```bash
ollama pull llava:latest
ollama pull qwen2.5-coder:7b  
ollama pull dolphin-mistral:latest
```

**PDF Processing Error**: Restart the proxy server
```bash
# Stop with Ctrl+C, then restart
python enhanced-proxy.py
```

## 📚 Model Capabilities

### 🖼️ **LLaVA (Vision + Text)**
- Analyze and describe images
- Read text in pictures/screenshots  
- Answer questions about visual content
- Combine visual and text understanding

### 💻 **Qwen2.5-Coder (Programming)**
- Write code in multiple languages
- Debug and fix code issues
- Code review and optimization
- Explain complex programming concepts
- Generate documentation

### 🐬 **Dolphin Mistral (General Chat)**
- Natural conversation and Q&A
- Creative writing and storytelling
- Explain complex topics
- General knowledge and reasoning
- Helpful for everyday tasks

## 🔧 Architecture

- **Frontend**: HTML/JavaScript interface
- **Backend**: FastAPI proxy server
- **AI**: Ollama serving multiple specialized models
- **RAG**: ChromaDB for document embeddings
- **Vision**: LLaVA for image analysis
