"""
FastAPI application configuration and entry point.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="RAG Document Q&A API",
    description="Retrieval-Augmented Generation API for document question answering using local LLM (phi3:mini) via Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for future React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8080",
        "*"  # Allow all for student/development use
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip middleware for response compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Ensure uploads directory exists
import os
from pathlib import Path
uploads_dir = Path("uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)

# Import and register routes
from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API info."""
    return {
        "service": "RAG Document Q&A",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "GET", "path": "/health/ollama", "description": "Ollama health check"},
            {"method": "POST", "path": "/documents/upload", "description": "Upload document"},
            {"method": "GET", "path": "/documents", "description": "List indexed documents"},
            {"method": "DELETE", "path": "/documents/{document_name}", "description": "Delete document"},
            {"method": "POST", "path": "/chat", "description": "Ask a question"},
        ]
    }