"""
Vector store service using ChromaDB for storing and retrieving document embeddings.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from app.services.document_processor import DocumentChunk
from app.services.embedding_service import EmbeddingService


class VectorStore:
    """Manages document embeddings in ChromaDB."""

    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_service = EmbeddingService()

    def initialize(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """
        Generate a deterministic ID for a chunk based on content and metadata.

        Args:
            chunk: Document chunk

        Returns:
            Unique hash-based ID
        """
        # Create ID from source, chunk_index, and text hash
        # This ensures same document chunks get same IDs (deduplication)
        content = f"{chunk.metadata.get('source', '')}_{chunk.metadata.get('chunk_index', 0)}_{chunk.text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def add_chunks(self, chunks: List[DocumentChunk]):
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to store
        """
        if not chunks:
            return

        self.initialize()

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embedding_service.embed_texts(texts)

        # Generate IDs
        ids = [self._generate_chunk_id(chunk) for chunk in chunks]

        # Prepare metadata (ChromaDB requires flat dictionaries)
        metadatas = []
        for chunk in chunks:
            # Convert all metadata values to strings for ChromaDB compatibility
            metadata = {}
            for key, value in chunk.metadata.items():
                metadata[key] = str(value)
            metadatas.append(metadata)

        # Add to ChromaDB (upsert to handle duplicates)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(
        self,
        query: str = None,
        query_embedding: List[float] = None,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            query: Search query text (will generate embedding internally if provided)
            query_embedding: Pre-computed query embedding (alternative to query)
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            collection_name: Optional collection to search (uses default if not provided)

        Returns:
            List of results with text, metadata, and distance
        """
        self.initialize()

        # Check for empty/whitespace query
        if query and not query.strip():
            return []

        if not query and not query_embedding:
            return []

        # Get the collection to search
        if collection_name and collection_name != self.collection_name:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = self.collection

        # Generate query embedding if not provided
        if not query_embedding and query:
            query_embedding = self.embedding_service.embed_text(query)

        if not query_embedding:
            return []

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata  # Optional metadata filtering
        )

        # Format results - ChromaDB returns 'documents' key
        formatted_results = []
        if results.get('ids') and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc_text = results['documents'][0][i] if results.get('documents') else ''
                result = {
                    'id': results['ids'][0][i],
                    'text': doc_text,  # For backward compatibility with tests
                    'document': doc_text,  # For RAGService
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                    'distance': results['distances'][0][i] if 'distances' in results and results['distances'] else None
                }
                formatted_results.append(result)

        return formatted_results

    def get_collection_count(self) -> int:
        """
        Get the number of chunks in the collection.

        Returns:
            Count of stored chunks
        """
        self.initialize()
        return self.collection.count()

    def delete_collection(self):
        """Delete the entire collection. Use with caution."""
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = None

    def clear_collection(self):
        """Clear all documents from the collection without deleting it."""
        if self.collection:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        self.initialize()
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def create_collection(self, collection_name: str):
        """Create a new collection (or get existing one)."""
        self.initialize()
        self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_collection(self, collection_name: str = None):
        """Delete a collection by name."""
        if collection_name is None:
            collection_name = self.collection_name
        if self.client:
            self.client.delete_collection(collection_name)
            if self.collection_name == collection_name:
                self.collection = None

    def add_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection_name: str = None
    ):
        """Add a single document to the collection."""
        if collection_name is None:
            collection_name = self.collection_name

        self.initialize()

        # Get the specified collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Generate ID from metadata
        content = f"{metadata.get('source', '')}_{metadata.get('chunk_index', 0)}_{text}"
        doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()

        # Convert metadata values to strings
        flat_metadata = {k: str(v) for k, v in metadata.items()}

        # Add to ChromaDB
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[flat_metadata]
        )
