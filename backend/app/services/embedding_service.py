"""
Embedding service using Sentence Transformers for generating text embeddings.
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    """Handles text embedding generation using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._embedding_dimension = None

    def load_model(self):
        """Load the embedding model. Called lazily on first use."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension from model
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Integer dimension of embeddings
        """
        if self._embedding_dimension is None:
            self.load_model()
        return self._embedding_dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        self.load_model()

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Convert to list for JSON serialization and ChromaDB compatibility
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self.load_model()

        # Batch encode for efficiency
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Convert to list of lists
        return embeddings.tolist()
