"""
Chat endpoint using existing RAG pipeline.
"""
import logging
from fastapi import APIRouter, HTTPException, status

from app.services.service_factory import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse, ChatErrorResponse, SourceInfo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using the RAG pipeline.

    Args:
        request: ChatRequest containing question and optional collection/top_k

    Returns:
        ChatResponse: Generated answer with sources and chunk count

    Raises:
        HTTPException: If RAG pipeline fails
    """
    rag_service = get_rag_service()

    try:
        result = rag_service.answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k
        )

        # Convert source dicts to SourceInfo objects
        sources = [
            SourceInfo(
                source=source.get("source", "Unknown"),
                chunk_index=source.get("chunk_index", 0),
                page=source.get("page")
            )
            for source in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=sources,
            retrieved_chunks=result.get("retrieved_chunks", 0),
            question=result.get("question", request.question),
            success=True
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        # Return a graceful error response rather than crashing the server
        return ChatResponse(
            answer="An error occurred while processing your question. Please try again.",
            sources=[],
            retrieved_chunks=0,
            question=request.question,
            success=False
        )