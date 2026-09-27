"""
Document management API routes (upload, list, delete).
"""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from typing import Optional

from app.services.service_factory import (
    get_document_processor,
    get_embedding_service,
    get_vector_store,
)
from app.services.document_service import DocumentService
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentDeleteResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"], prefix="/documents")


def _get_document_service() -> DocumentService:
    """Create document service instance."""
    return DocumentService(
        document_processor=get_document_processor(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        collection_name="documents"
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Upload and index a document file.

    Args:
        file: Uploaded file (PDF, DOCX, or TXT, max 10MB)

    Returns:
        DocumentUploadResponse: Upload result with indexed chunks count

    Raises:
        HTTPException: For invalid file type, size, or processing errors
    """
    # Basic file validation
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Validate content type hint
    allowed_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    }
    content_type = file.content_type or ""
    # Allow content type to be approximate; we check extension in service

    try:
        # Read file content
        content = await file.read()

        # Check file size
        file_size = len(content)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {len(content) / (1024 * 1024):.1f}MB (max: 10MB)"
            )

        document_service = _get_document_service()
        result = await document_service.process_and_index_document(
            file_content=content,
            filename=file.filename
        )

        return DocumentUploadResponse(
            success=result["success"],
            filename=result["filename"],
            file_type=result["file_type"],
            chunks_indexed=result["chunks_indexed"],
            message=result["message"]
        )

    except ValueError as e:
        logger.warning(f"Document validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """
    List all indexed documents and their chunk counts.

    Returns:
        DocumentListResponse: List of documents with metadata
    """
    document_service = _get_document_service()
    docs = document_service.list_documents()

    total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
    total_docs = len(docs)

    document_infos = [
        DocumentInfo(
            source=doc["source"],
            file_type=doc.get("file_type", "unknown"),
            chunk_count=doc.get("chunk_count", 0)
        )
        for doc in docs
    ]

    return DocumentListResponse(
        documents=document_infos,
        total_documents=total_docs,
        total_chunks=total_chunks
    )


@router.delete("/{document_name}", response_model=DocumentDeleteResponse)
async def delete_document(document_name: str):
    """
    Delete all chunks for a specific document source.

    Args:
        document_name: Source document name (e.g., "file.pdf")

    Returns:
        DocumentDeleteResponse: Deletion confirmation

    Raises:
        HTTPException: If document not found or deletion fails
    """
    document_service = _get_document_service()

    try:
        result = document_service.delete_document(document_name)

        return DocumentDeleteResponse(
            success=result["success"],
            deleted_source=result["deleted_source"],
            chunks_deleted=result["chunks_deleted"],
            message=result["message"]
        )
    except ValueError as e:
        logger.warning(f"Document deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )