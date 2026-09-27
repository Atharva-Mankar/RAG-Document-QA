"""
Pydantic schemas for API request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# Health endpoints
class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    message: str


class OllamaHealthResponse(BaseModel):
    """Response for Ollama health check."""
    status: str
    ollama_available: bool
    model_available: bool
    model_name: str


# Document upload
class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""
    success: bool
    filename: str
    file_type: str
    chunks_indexed: int
    message: str


class DocumentInfo(BaseModel):
    """Information about an indexed document."""
    source: str
    file_type: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response listing all indexed documents."""
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    success: bool
    deleted_source: str
    chunks_deleted: int
    message: str


# Chat endpoints
class ChatRequest(BaseModel):
    """Request for chat/RAG endpoint."""
    question: str = Field(..., min_length=1, max_length=2000)
    collection_name: str = Field(default="documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceInfo(BaseModel):
    """Source information for chat response."""
    source: str
    chunk_index: int
    page: Optional[int] = None


class ChatResponse(BaseModel):
    """Response from chat/RAG endpoint."""
    answer: str
    sources: List[SourceInfo]
    retrieved_chunks: int
    question: str
    success: bool


class ChatErrorResponse(BaseModel):
    """Error response for chat endpoint."""
    success: bool
    error: str
    question: str