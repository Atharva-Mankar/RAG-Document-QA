"""
Document service for handling document upload, indexing, listing, and deletion.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil
import tempfile
import logging
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class DocumentService:
    """Service for managing document lifecycle in the RAG system."""

    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        collection_name: str = "documents"
    ):
        """
        Initialize document service.

        Args:
            document_processor: Service for extracting and chunking documents
            embedding_service: Service for generating embeddings
            vector_store: Vector store for storage and retrieval
            collection_name: ChromaDB collection to use
        """
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.collection_name = collection_name

    def validate_file(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """
        Validate uploaded file.

        Args:
            filename: Original filename
            file_size: Size of file in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file type
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {extension}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

        # Check file size
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large: {size_mb:.1f}MB. Maximum allowed: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"

        if file_size == 0:
            return False, "File is empty"

        return True, ""

    async def process_and_index_document(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Process and index an uploaded document.

        Args:
            file_content: Raw bytes of the uploaded file
            filename: Original filename

        Returns:
            Dict with success status, filename, file_type, and chunks_indexed

        Raises:
            ValueError: If file validation or processing fails
        """
        # Validate file
        is_valid, error_msg = self.validate_file(filename, len(file_content))
        if not is_valid:
            raise ValueError(error_msg)

        # Save to temporary file
        temp_path = None
        try:
            # Create temp file with proper extension
            extension = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(
                suffix=extension,
                delete=False,
                dir="uploads"
            ) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            # Process document
            chunks = self.document_processor.process_document(temp_path)

            if not chunks:
                raise ValueError("No content could be extracted from the document")

            # Index chunks in vector store
            for chunk in chunks:
                embedding = self.embedding_service.embed_text(chunk.text)
                self.vector_store.add_document(
                    text=chunk.text,
                    embedding=embedding,
                    metadata=chunk.metadata,
                    collection_name=self.collection_name
                )

            file_type = extension.lstrip(".")
            return {
                "success": True,
                "filename": filename,
                "file_type": file_type,
                "chunks_indexed": len(chunks),
                "message": f"Successfully indexed {len(chunks)} chunks from {filename}"
            }

        finally:
            # Clean up temp file
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents with their chunk counts.

        Returns:
            List of dicts with source, file_type, and chunk_count
        """
        try:
            self.vector_store.initialize()

            # Get all data from collection
            all_data = self.vector_store.collection.get(include=["metadatas"])

            if not all_data or not all_data.get("metadatas"):
                return []

            # Aggregate chunks by source
            documents = {}
            for metadata in all_data["metadatas"]:
                if not metadata:
                    continue

                source = metadata.get("source", "Unknown")
                file_type = metadata.get("file_type", "Unknown")

                if source not in documents:
                    documents[source] = {
                        "source": source,
                        "file_type": file_type,
                        "chunk_count": 0
                    }

                documents[source]["chunk_count"] += 1

            return list(documents.values())

        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def delete_document(self, document_name: str) -> Dict[str, Any]:
        """
        Delete all chunks belonging to a specific document.

        Args:
            document_name: Name of the source document to delete

        Returns:
            Dict with success status, deleted_source, and chunks_deleted

        Raises:
            ValueError: If document not found
        """
        try:
            self.vector_store.initialize()

            # Get all data from collection
            all_data = self.vector_store.collection.get(include=["metadatas"])

            if not all_data or not all_data.get("metadatas"):
                raise ValueError(f"Document '{document_name}' not found")

            # Find IDs belonging to this document
            ids_to_delete = []
            for idx, metadata in enumerate(all_data["metadatas"]):
                if metadata and metadata.get("source") == document_name:
                    ids_to_delete.append(all_data["ids"][idx])

            if not ids_to_delete:
                raise ValueError(f"Document '{document_name}' not found")

            # Delete the chunks
            self.vector_store.collection.delete(ids=ids_to_delete)

            return {
                "success": True,
                "deleted_source": document_name,
                "chunks_deleted": len(ids_to_delete),
                "message": f"Deleted {len(ids_to_delete)} chunks from {document_name}"
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise ValueError(f"Error deleting document: {str(e)}")