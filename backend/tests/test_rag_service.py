"""
Tests for RAG Service.
"""

import pytest
from unittest.mock import Mock, MagicMock

from app.services.rag_service import RAGService


class TestRAGService:
    """Test cases for RAG service."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        embedding_service = Mock()
        vector_store_service = Mock()
        llm_service = Mock()

        return embedding_service, vector_store_service, llm_service

    @pytest.fixture
    def rag_service(self, mock_services):
        """Create RAG service with mocked dependencies."""
        embedding_service, vector_store_service, llm_service = mock_services
        return RAGService(
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            llm_service=llm_service,
            top_k=3
        )

    def test_initialization(self, rag_service, mock_services):
        """Test RAG service initialization."""
        embedding_service, vector_store_service, llm_service = mock_services

        assert rag_service.embedding_service == embedding_service
        assert rag_service.vector_store_service == vector_store_service
        assert rag_service.llm_service == llm_service
        assert rag_service.top_k == 3
        assert "provided document context" in rag_service.system_prompt

    def test_answer_question_success(self, rag_service, mock_services):
        """Test successful question answering."""
        embedding_service, vector_store_service, llm_service = mock_services

        # Mock embedding
        embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]

        # Mock search results
        vector_store_service.search.return_value = [
            {
                "document": "Python is a programming language.",
                "metadata": {
                    "source": "python_guide.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            {
                "document": "Python was created by Guido van Rossum.",
                "metadata": {
                    "source": "python_guide.pdf",
                    "page": 2,
                    "chunk_index": 1
                }
            }
        ]

        # Mock LLM response
        llm_service.generate.return_value = {
            "response": "Python is a programming language created by Guido van Rossum.",
            "success": True
        }

        # Test question answering
        result = rag_service.answer_question("What is Python?")

        assert result["answer"] == "Python is a programming language created by Guido van Rossum."
        assert result["question"] == "What is Python?"
        assert result["retrieved_chunks"] == 2
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "python_guide.pdf"
        assert result["sources"][0]["page"] == 1

        # Verify calls
        embedding_service.embed_text.assert_called_once_with("What is Python?")
        vector_store_service.search.assert_called_once()
        llm_service.generate.assert_called_once()

    def test_answer_question_no_results(self, rag_service, mock_services):
        """Test question answering when no results are found."""
        embedding_service, vector_store_service, llm_service = mock_services

        # Mock embedding
        embedding_service.embed_text.return_value = [0.1, 0.2, 0.3]

        # Mock empty search results
        vector_store_service.search.return_value = []

        # Test question answering
        result = rag_service.answer_question("What is quantum computing?")

        assert "cannot find this information" in result["answer"].lower()
        assert result["sources"] == []
        assert result["retrieved_chunks"] == 0

        # LLM should not be called if no context
        llm_service.generate.assert_not_called()

    def test_build_context(self, rag_service):
        """Test context building from search results."""
        results = [
            {
                "document": "First chunk of text.",
                "metadata": {
                    "source": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            {
                "document": "Second chunk of text.",
                "metadata": {
                    "source": "doc1.pdf",
                    "page": 2,
                    "chunk_index": 1
                }
            }
        ]

        context = rag_service._build_context(results)

        assert "SOURCE 1" in context
        assert "SOURCE 2" in context
        assert "FILE: doc1.pdf" in context
        assert "PAGE: 1" in context
        assert "PAGE: 2" in context
        assert "First chunk of text." in context
        assert "Second chunk of text." in context

    def test_build_context_without_page(self, rag_service):
        """Test context building when page number is not available."""
        results = [
            {
                "document": "Text without page number.",
                "metadata": {
                    "source": "doc.txt",
                    "chunk_index": 0
                }
            }
        ]

        context = rag_service._build_context(results)

        assert "FILE: doc.txt" in context
        assert "PAGE:" not in context or "PAGE: None" not in context
        assert "Text without page number." in context

    def test_extract_sources(self, rag_service):
        """Test source extraction from search results."""
        results = [
            {
                "document": "Text 1",
                "metadata": {
                    "source": "doc1.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            {
                "document": "Text 2",
                "metadata": {
                    "source": "doc1.pdf",
                    "page": 2,
                    "chunk_index": 1
                }
            },
            {
                "document": "Text 3",
                "metadata": {
                    "source": "doc2.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            }
        ]

        sources = rag_service._extract_sources(results)

        assert len(sources) == 3
        assert sources[0]["source"] == "doc1.pdf"
        assert sources[0]["page"] == 1
        assert sources[1]["page"] == 2
        assert sources[2]["source"] == "doc2.pdf"

    def test_extract_sources_without_page(self, rag_service):
        """Test source extraction when page is not available."""
        results = [
            {
                "document": "Text",
                "metadata": {
                    "source": "doc.txt",
                    "chunk_index": 0
                }
            }
        ]

        sources = rag_service._extract_sources(results)

        assert len(sources) == 1
        assert sources[0]["source"] == "doc.txt"
        assert "page" not in sources[0]
        assert sources[0]["chunk_index"] == 0

    def test_answer_question_custom_top_k(self, rag_service, mock_services):
        """Test question answering with custom top_k."""
        embedding_service, vector_store_service, llm_service = mock_services

        embedding_service.embed_text.return_value = [0.1, 0.2]
        vector_store_service.search.return_value = [
            {"document": "Text", "metadata": {"source": "file.pdf"}}
        ]
        llm_service.generate.return_value = {
            "response": "Answer",
            "success": True
        }

        rag_service.answer_question("Question?", top_k=10)

        # Verify search was called with custom top_k
        call_args = vector_store_service.search.call_args
        assert call_args[1]["top_k"] == 10

    def test_check_health(self, rag_service, mock_services):
        """Test health check."""
        embedding_service, vector_store_service, llm_service = mock_services

        vector_store_service.collection_exists.return_value = True
        llm_service.check_health.return_value = True

        health = rag_service.check_health()

        assert health["embedding_service"] is True
        assert health["vector_store"] is True
        assert health["llm_service"] is True

    def test_system_prompt_guidelines(self, rag_service):
        """Test that system prompt contains key guidelines."""
        system_prompt = rag_service.system_prompt

        # Check for key instructions
        assert "ONLY" in system_prompt or "only" in system_prompt
        assert "provided" in system_prompt.lower()
        assert "context" in system_prompt.lower()
        assert "cannot find" in system_prompt.lower() or "not found" in system_prompt.lower()
        assert "do not" in system_prompt.lower() or "don't" in system_prompt.lower()


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests with real services."""

    def test_real_rag_pipeline(self):
        """Test real RAG pipeline (requires all services running)."""
        pytest.skip("Integration test - run manually with demo_rag.py")
