"""
Tests for vector store functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from app.services.vector_store import VectorStore
from app.services.document_processor import DocumentChunk


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_chroma_dir):
    """Create a VectorStore instance with temporary storage."""
    return VectorStore(persist_directory=temp_chroma_dir, collection_name="test_collection")


@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing."""
    chunks = [
        DocumentChunk(
            text="The quick brown fox jumps over the lazy dog.",
            metadata={"source": "test.txt", "file_type": "txt", "chunk_index": 0}
        ),
        DocumentChunk(
            text="Python is a popular programming language.",
            metadata={"source": "test.txt", "file_type": "txt", "chunk_index": 1}
        ),
        DocumentChunk(
            text="Machine learning models can process natural language.",
            metadata={"source": "test.txt", "file_type": "txt", "chunk_index": 2}
        ),
    ]
    return chunks


class TestVectorStore:
    """Test VectorStore functionality."""

    def test_initialization(self, temp_chroma_dir):
        """Test vector store initialization."""
        store = VectorStore(persist_directory=temp_chroma_dir, collection_name="test")
        assert store.persist_directory == temp_chroma_dir
        assert store.collection_name == "test"
        assert store.client is None  # Lazy initialization

    def test_initialize_creates_directory(self, temp_chroma_dir):
        """Test that initialize creates the persist directory."""
        persist_path = str(Path(temp_chroma_dir) / "subdir")
        store = VectorStore(persist_directory=persist_path)
        store.initialize()

        assert Path(persist_path).exists()
        assert store.client is not None
        assert store.collection is not None

    def test_add_chunks(self, vector_store, sample_chunks):
        """Test adding chunks to the vector store."""
        vector_store.add_chunks(sample_chunks)

        count = vector_store.get_collection_count()
        assert count == len(sample_chunks)

    def test_add_empty_chunks(self, vector_store):
        """Test adding empty list of chunks."""
        vector_store.add_chunks([])
        count = vector_store.get_collection_count()
        assert count == 0

    def test_chunk_id_generation(self, vector_store, sample_chunks):
        """Test that chunk IDs are generated deterministically."""
        chunk = sample_chunks[0]
        id1 = vector_store._generate_chunk_id(chunk)
        id2 = vector_store._generate_chunk_id(chunk)

        # Same chunk should produce same ID
        assert id1 == id2
        assert isinstance(id1, str)
        assert len(id1) == 32  # MD5 hash length

    def test_duplicate_ingestion(self, vector_store, sample_chunks):
        """Test that duplicate chunks are handled (upserted, not duplicated)."""
        # Add chunks first time
        vector_store.add_chunks(sample_chunks)
        count1 = vector_store.get_collection_count()

        # Add same chunks again
        vector_store.add_chunks(sample_chunks)
        count2 = vector_store.get_collection_count()

        # Count should remain the same (upsert behavior)
        assert count1 == count2 == len(sample_chunks)

    def test_search_basic(self, vector_store, sample_chunks):
        """Test basic semantic search."""
        vector_store.add_chunks(sample_chunks)

        # Search for something related to programming
        results = vector_store.search("programming languages", top_k=2)

        assert len(results) <= 2
        assert all('text' in r for r in results)
        assert all('metadata' in r for r in results)
        assert all('distance' in r for r in results)

    def test_search_relevance(self, vector_store, sample_chunks):
        """Test that search returns relevant results."""
        vector_store.add_chunks(sample_chunks)

        # Search for programming-related content
        results = vector_store.search("Python programming", top_k=3)

        # The chunk about Python should be in the results
        texts = [r['text'] for r in results]
        assert any("Python" in text for text in texts)

    def test_search_top_k(self, vector_store, sample_chunks):
        """Test that top_k parameter is respected."""
        vector_store.add_chunks(sample_chunks)

        results_k1 = vector_store.search("test query", top_k=1)
        results_k2 = vector_store.search("test query", top_k=2)

        assert len(results_k1) == 1
        assert len(results_k2) == 2

    def test_search_empty_query(self, vector_store, sample_chunks):
        """Test search with empty query."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("", top_k=5)
        assert results == []

        results = vector_store.search("   ", top_k=5)
        assert results == []

    def test_search_empty_collection(self, vector_store):
        """Test search on empty collection."""
        results = vector_store.search("test query", top_k=5)
        # Should return empty results without error
        assert isinstance(results, list)
        assert len(results) == 0

    def test_metadata_preservation(self, vector_store, sample_chunks):
        """Test that metadata is preserved in storage and retrieval."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("test", top_k=3)

        for result in results:
            metadata = result['metadata']
            assert 'source' in metadata
            assert 'file_type' in metadata
            assert 'chunk_index' in metadata
            assert metadata['source'] == 'test.txt'
            assert metadata['file_type'] == 'txt'

    def test_metadata_with_page_numbers(self, vector_store):
        """Test metadata preservation including page numbers (PDF scenario)."""
        chunks = [
            DocumentChunk(
                text="Content from page 1.",
                metadata={"source": "doc.pdf", "file_type": "pdf", "page": 1, "chunk_index": 0}
            ),
            DocumentChunk(
                text="Content from page 2.",
                metadata={"source": "doc.pdf", "file_type": "pdf", "page": 2, "chunk_index": 1}
            ),
        ]

        vector_store.add_chunks(chunks)
        results = vector_store.search("content", top_k=2)

        for result in results:
            assert 'page' in result['metadata']
            assert result['metadata']['page'] in ['1', '2']  # Stored as strings

    def test_get_collection_count(self, vector_store, sample_chunks):
        """Test getting collection count."""
        assert vector_store.get_collection_count() == 0

        vector_store.add_chunks(sample_chunks[:2])
        assert vector_store.get_collection_count() == 2

        vector_store.add_chunks(sample_chunks[2:])
        assert vector_store.get_collection_count() == 3

    def test_clear_collection(self, vector_store, sample_chunks):
        """Test clearing all documents from collection."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.get_collection_count() == 3

        vector_store.clear_collection()
        assert vector_store.get_collection_count() == 0

    def test_persistence(self, temp_chroma_dir, sample_chunks):
        """Test that data persists across VectorStore instances."""
        # Create first instance and add data
        store1 = VectorStore(persist_directory=temp_chroma_dir, collection_name="persist_test")
        store1.add_chunks(sample_chunks)
        count1 = store1.get_collection_count()

        # Create second instance with same directory
        store2 = VectorStore(persist_directory=temp_chroma_dir, collection_name="persist_test")
        count2 = store2.get_collection_count()

        # Data should persist
        assert count1 == count2 == len(sample_chunks)

    def test_distance_ordering(self, vector_store, sample_chunks):
        """Test that results are ordered by relevance (distance)."""
        vector_store.add_chunks(sample_chunks)

        results = vector_store.search("Python programming language", top_k=3)

        # Results should be ordered by distance (lower is better for cosine)
        if len(results) > 1:
            distances = [r['distance'] for r in results]
            assert distances == sorted(distances)  # Should be in ascending order


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_pipeline(self, temp_chroma_dir):
        """Test complete pipeline: document -> chunks -> embeddings -> storage -> retrieval."""
        from app.services.document_processor import DocumentProcessor
        import tempfile

        # Create a test document
        test_content = """
        Artificial intelligence is transforming technology.
        Machine learning is a subset of AI.
        Deep learning uses neural networks.
        Natural language processing helps computers understand text.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        try:
            # Process document
            processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
            chunks = processor.process_document(temp_file)

            assert len(chunks) > 0

            # Store in vector database
            store = VectorStore(persist_directory=temp_chroma_dir, collection_name="integration_test")
            store.add_chunks(chunks)

            assert store.get_collection_count() == len(chunks)

            # Retrieve relevant chunks
            results = store.search("What is machine learning?", top_k=2)

            assert len(results) > 0

            # Verify results contain relevant content
            combined_text = " ".join([r['text'] for r in results])
            assert "machine learning" in combined_text.lower() or "learning" in combined_text.lower()

            # Verify metadata is preserved
            for result in results:
                assert 'source' in result['metadata']
                assert 'file_type' in result['metadata']
                assert result['metadata']['file_type'] == 'txt'

        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)

    def test_multiple_documents(self, temp_chroma_dir):
        """Test ingesting and retrieving from multiple documents."""
        chunks1 = [
            DocumentChunk(
                text="Document 1 discusses cats and dogs.",
                metadata={"source": "doc1.txt", "file_type": "txt", "chunk_index": 0}
            ),
        ]

        chunks2 = [
            DocumentChunk(
                text="Document 2 discusses programming and Python.",
                metadata={"source": "doc2.txt", "file_type": "txt", "chunk_index": 0}
            ),
        ]

        store = VectorStore(persist_directory=temp_chroma_dir, collection_name="multi_doc_test")
        store.add_chunks(chunks1)
        store.add_chunks(chunks2)

        # Search should retrieve from both documents
        results = store.search("animals", top_k=1)
        assert any("doc1.txt" in r['metadata']['source'] for r in results)

        results = store.search("coding", top_k=1)
        assert any("doc2.txt" in r['metadata']['source'] for r in results)
