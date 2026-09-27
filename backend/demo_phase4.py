"""
Demonstration of the complete RAG document ingestion and retrieval pipeline.
"""

import tempfile
from pathlib import Path
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore


def main():
    print("=" * 70)
    print("RAG Document Q&A System - Phase 4 Demo")
    print("Embeddings + ChromaDB Integration")
    print("=" * 70)
    print()

    # Create a sample document
    sample_document = """
    Artificial Intelligence and Machine Learning

    Artificial intelligence (AI) is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that enables systems to learn from data.

    Deep Learning

    Deep learning is a type of machine learning that uses neural networks with
    multiple layers. It has been particularly successful in image recognition,
    natural language processing, and speech recognition tasks.

    Natural Language Processing

    Natural language processing (NLP) helps computers understand and generate
    human language. It powers chatbots, translation systems, and sentiment analysis.

    Applications

    AI and ML are used in healthcare for diagnosis, in finance for fraud detection,
    in transportation for autonomous vehicles, and in many other domains.
    """

    # Create temporary document
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(sample_document)
        temp_file = f.name

    try:
        print("Step 1: Document Processing")
        print("-" * 70)
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        chunks = processor.process_document(temp_file)
        print(f"✓ Extracted {len(chunks)} chunks from document")
        print(f"  Chunk size: {processor.chunk_size} characters")
        print(f"  Chunk overlap: {processor.chunk_overlap} characters")
        print()

        print("Step 2: Embedding Generation & Storage")
        print("-" * 70)
        # Use temporary directory for demo
        temp_dir = tempfile.mkdtemp()

        store = VectorStore(persist_directory=temp_dir, collection_name="demo")
        print(f"✓ Initialized ChromaDB at: {temp_dir}")

        store.add_chunks(chunks)
        print(f"✓ Generated embeddings using: sentence-transformers/all-MiniLM-L6-v2")
        print(f"✓ Stored {store.get_collection_count()} chunks in ChromaDB")
        print(f"  Embedding dimension: {store.embedding_service.get_embedding_dimension()}")
        print()

        print("Step 3: Semantic Search Demonstrations")
        print("-" * 70)

        # Demo query 1
        query1 = "What is deep learning?"
        print(f"\n🔍 Query: '{query1}'")
        print()
        results1 = store.search(query1, top_k=2)

        for i, result in enumerate(results1, 1):
            print(f"Result {i}:")
            print(f"  Text: {result['text'][:150]}...")
            print(f"  Source: {result['metadata']['source']}")
            print(f"  File Type: {result['metadata']['file_type']}")
            print(f"  Chunk Index: {result['metadata']['chunk_index']}")
            print(f"  Similarity Distance: {result['distance']:.4f}")
            print()

        # Demo query 2
        query2 = "How is AI used in healthcare?"
        print(f"🔍 Query: '{query2}'")
        print()
        results2 = store.search(query2, top_k=2)

        for i, result in enumerate(results2, 1):
            print(f"Result {i}:")
            print(f"  Text: {result['text'][:150]}...")
            print(f"  Metadata: {result['metadata']}")
            print(f"  Similarity Distance: {result['distance']:.4f}")
            print()

        # Demo query 3
        query3 = "natural language understanding"
        print(f"🔍 Query: '{query3}'")
        print()
        results3 = store.search(query3, top_k=1)

        for i, result in enumerate(results3, 1):
            print(f"Result {i}:")
            print(f"  Text: {result['text']}")
            print(f"  Metadata: {result['metadata']}")
            print(f"  Similarity Distance: {result['distance']:.4f}")
            print()

        print("=" * 70)
        print("✓ Phase 4 Complete: Embeddings + ChromaDB Integration")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  • Document chunks: {len(chunks)}")
        print(f"  • Embedding model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"  • Embedding dimension: 384")
        print(f"  • Vector database: ChromaDB (persistent)")
        print(f"  • Similarity metric: Cosine similarity")
        print(f"  • Duplicate handling: MD5-based deduplication")
        print()
        print("✓ System ready for LLM integration (Phase 5)")

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    finally:
        # Cleanup temp file
        Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
