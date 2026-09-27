"""
Health check API routes.
"""
from fastapi import APIRouter
from app.models.schemas import HealthResponse, OllamaHealthResponse
from app.services.service_factory import get_llm_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns:
        HealthResponse: Status and message confirming backend is running
    """
    return HealthResponse(
        status="ok",
        message="RAG Document Q&A backend is running"
    )


@router.get("/health/ollama", response_model=OllamaHealthResponse)
async def ollama_health_check():
    """
    Check Ollama availability and model presence.

    Returns:
        OllamaHealthResponse: Ollama and model availability status
    """
    llm_service = get_llm_service()
    model_available = llm_service.check_health()

    return OllamaHealthResponse(
        status="ok" if model_available else "degraded",
        ollama_available=True,  # If we got here, Ollama responded
        model_available=model_available,
        model_name=llm_service.model_name
    )