"""
Service factory for creating and managing singleton instances of RAG services.
"""
from typing import Optional
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.document_processor import DocumentProcessor


class ServiceFactory:
    """Factory for creating singleton service instances."""

    _instance: Optional["ServiceFactory"] = None
    _embedding_service: Optional[EmbeddingService] = None
    _vector_store: Optional[VectorStore] = None
    _llm_service: Optional[LLMService] = None
    _rag_service: Optional[RAGService] = None
    _document_processor: Optional[DocumentProcessor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create EmbeddingService singleton."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        """Get or create VectorStore singleton."""
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def llm_service(self) -> LLMService:
        """Get or create LLMService singleton."""
        if self._llm_service is None:
            self._llm_service = LLMService(model_name="phi3:mini")
        return self._llm_service

    @property
    def rag_service(self) -> RAGService:
        """Get or create RAGService singleton."""
        if self._rag_service is None:
            self._rag_service = RAGService(
                embedding_service=self.embedding_service,
                vector_store_service=self.vector_store,
                llm_service=self.llm_service,
                top_k=3
            )
        return self._rag_service

    @property
    def document_processor(self) -> DocumentProcessor:
        """Get or create DocumentProcessor singleton."""
        if self._document_processor is None:
            self._document_processor = DocumentProcessor()
        return self._document_processor

    def reset(self):
        """Reset all service instances (useful for testing)."""
        self._embedding_service = None
        self._vector_store = None
        self._llm_service = None
        self._rag_service = None
        self._document_processor = None


# Global factory instance
_service_factory = ServiceFactory()


def get_embedding_service() -> EmbeddingService:
    """Dependency function for embedding service."""
    return _service_factory.embedding_service


def get_vector_store() -> VectorStore:
    """Dependency function for vector store."""
    return _service_factory.vector_store


def get_llm_service() -> LLMService:
    """Dependency function for LLM service."""
    return _service_factory.llm_service


def get_rag_service() -> RAGService:
    """Dependency function for RAG service."""
    return _service_factory.rag_service


def get_document_processor() -> DocumentProcessor:
    """Dependency function for document processor."""
    return _service_factory.document_processor


def get_service_factory() -> ServiceFactory:
    """Get the global service factory."""
    return _service_factory