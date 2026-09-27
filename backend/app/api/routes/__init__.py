"""API routes package initialization."""
from .health import router as health_router
from .documents import router as documents_router
from .chat import router as chat_router

__all__ = ["health_router", "documents_router", "chat_router"]