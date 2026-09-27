"""
Tests for embedding service functionality.
"""

import pytest
from app.services.embeddings import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create an EmbeddingService instance for testing."""
    return EmbeddingService()


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    def test_service_initialization(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert service.model is None  # Lazy loading

    def test_model_loading(self, embedding_service):
        """Test that model loads correctly."""
        embedding_service.load_model()
        assert embedding_service.model is not None

    def test_embedding_dimension(self, embedding_service):
        """Test that embedding dimension is correct."""
        dimension = embedding_service.get_embedding_dimension()
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert dimension == 384

    def test_embed_single_text(self, embedding_service):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedding_service.embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_multiple_texts(self, embedding_service):
        """Test embedding multiple texts in batch."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedding_service.embed_texts(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert all(isinstance(x, float) for emb in embeddings for x in emb)

    def test_embed_empty_list(self, embedding_service):
        """Test embedding empty list returns empty list."""
        embeddings = embedding_service.embed_texts([])
        assert embeddings == []

    def test_different_texts_different_embeddings(self, embedding_service):
        """Test that different texts produce different embeddings."""
        text1 = "The cat sits on the mat."
        text2 = "The dog runs in the park."

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)

        # Embeddings should be different
        assert emb1 != emb2

    def test_same_text_same_embedding(self, embedding_service):
        """Test that same text produces same embedding."""
        text = "This is a consistent test."

        emb1 = embedding_service.embed_text(text)
        emb2 = embedding_service.embed_text(text)

        # Embeddings should be identical
        assert emb1 == emb2

    def test_similar_texts_similar_embeddings(self, embedding_service):
        """Test that semantically similar texts have similar embeddings."""
        text1 = "The cat sits on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "Quantum physics is complex."

        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)

        # Calculate cosine similarity
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_1_2 = cosine_similarity(emb1, emb2)
        sim_1_3 = cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity than unrelated texts
        assert sim_1_2 > sim_1_3

    def test_embedding_values_normalized(self, embedding_service):
        """Test that embeddings are in reasonable range."""
        text = "Test sentence for value range."
        embedding = embedding_service.embed_text(text)

        # Values should be in reasonable float range (typically -1 to 1 for normalized)
        assert all(-10 < x < 10 for x in embedding)
