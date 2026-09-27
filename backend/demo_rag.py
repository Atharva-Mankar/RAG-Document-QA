"""
RAG Demo Script

Demonstrates the complete RAG pipeline:
1. Document ingestion
2. Question answering with grounded responses
3. Handling questions not in the documents
"""

import sys
import os
from pathlib import Path
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.document_processor import DocumentProcessor


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run RAG demo."""
    print_section("RAG SYSTEM DEMO - Phase 5")

    # Initialize services
    print("Initializing services...")
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    llm_service = LLMService(model_name="phi3:mini")
    document_processor = DocumentProcessor()

    print("[OK] Embedding service initialized")
    print("[OK] Vector store initialized")
    print("[OK] LLM service initialized (phi3:mini)")
    print("[OK] Document processor initialized")

    # Check LLM health
    print("\nChecking Ollama connection...")
    if not llm_service.check_health():
        print("[FAIL] ERROR: Cannot connect to Ollama or phi3:mini not available")
        print("Please ensure Ollama is running and phi3:mini is installed.")
        return 1
    print("[OK] Ollama connection successful")

    # Initialize RAG service
    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store_service=vector_store,
        llm_service=llm_service,
        top_k=3
    )
    print("[OK] RAG service initialized")

    # Create a temporary demo collection
    demo_collection = "demo_rag"

    # Create demo document
    print_section("STEP 1: Creating Demo Document")

    demo_content = """
    # Python Programming Guide

    ## Introduction to Python
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability and simplicity.

    ## Key Features
    Python supports multiple programming paradigms including:
    - Object-oriented programming
    - Procedural programming
    - Functional programming

    Python uses dynamic typing and automatic memory management.

    ## Popular Applications
    Python is widely used for:
    - Web development (Django, Flask)
    - Data science and machine learning (NumPy, Pandas, TensorFlow)
    - Automation and scripting
    - Scientific computing

    ## Version Information
    As of 2024, Python 3.12 is the latest stable version.
    Python 2 reached end-of-life in January 2020 and is no longer supported.
    """

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(demo_content)
        demo_file_path = f.name

    print(f"[OK] Created demo document: {demo_file_path}")
    print(f"  Content length: {len(demo_content)} characters")

    # Process and ingest document
    print_section("STEP 2: Processing and Ingesting Document")

    try:
        # Process document
        chunks = document_processor.process_document(demo_file_path)
        print(f"[OK] Document processed into {len(chunks)} chunks")

        # Create collection
        vector_store.create_collection(demo_collection)
        print(f"[OK] Created collection: {demo_collection}")

        # Add chunks to vector store
        for chunk in chunks:
            embedding = embedding_service.embed_text(chunk.text)
            vector_store.add_document(
                text=chunk.text,
                embedding=embedding,
                metadata=chunk.metadata,
                collection_name=demo_collection
            )

        print(f"[OK] Added {len(chunks)} chunks to vector store")

    except Exception as e:
        print(f"[FAIL] ERROR during ingestion: {str(e)}")
        return 1

    # Test Question 1: Answer should be in the document
    print_section("STEP 3: Question with Answer in Document")

    question1 = "Who created Python and when was it first released?"
    print(f"QUESTION: {question1}\n")

    try:
        result1 = rag_service.answer_question(
            question=question1,
            collection_name=demo_collection,
            top_k=3
        )

        print("ANSWER:")
        print(result1["answer"])
        print(f"\nRETRIEVED CHUNKS: {result1['retrieved_chunks']}")
        print("\nSOURCES:")
        for idx, source in enumerate(result1["sources"], 1):
            print(f"  {idx}. {source['source']} (chunk {source['chunk_index']})")

    except Exception as e:
        print(f"[FAIL] ERROR: {str(e)}")
        return 1

    # Test Question 2: Answer NOT in the document
    print_section("STEP 4: Question NOT in Document")

    question2 = "What is the capital of France?"
    print(f"QUESTION: {question2}\n")

    try:
        result2 = rag_service.answer_question(
            question=question2,
            collection_name=demo_collection,
            top_k=3
        )

        print("ANSWER:")
        print(result2["answer"])
        print(f"\nRETRIEVED CHUNKS: {result2['retrieved_chunks']}")

        if result2["sources"]:
            print("\nSOURCES:")
            for idx, source in enumerate(result2["sources"], 1):
                print(f"  {idx}. {source['source']}")
        else:
            print("\nSOURCES: None (no relevant context found)")

    except Exception as e:
        print(f"[FAIL] ERROR: {str(e)}")
        return 1

    # Test Question 3: Another document question
    print_section("STEP 5: Another Question from Document")

    question3 = "What are some popular applications of Python?"
    print(f"QUESTION: {question3}\n")

    try:
        result3 = rag_service.answer_question(
            question=question3,
            collection_name=demo_collection,
            top_k=3
        )

        print("ANSWER:")
        print(result3["answer"])
        print(f"\nRETRIEVED CHUNKS: {result3['retrieved_chunks']}")
        print("\nSOURCES:")
        for idx, source in enumerate(result3["sources"], 1):
            print(f"  {idx}. {source['source']} (chunk {source['chunk_index']})")

    except Exception as e:
        print(f"[FAIL] ERROR: {str(e)}")
        return 1

    # Cleanup
    print_section("STEP 6: Cleanup")

    try:
        vector_store.delete_collection(demo_collection)
        print(f"[OK] Deleted collection: {demo_collection}")

        os.unlink(demo_file_path)
        print(f"[OK] Deleted temporary file")

    except Exception as e:
        print(f"[WARN] Warning during cleanup: {str(e)}")

    # Summary
    print_section("DEMO COMPLETE")

    print("[OK] Successfully demonstrated RAG pipeline:")
    print("  1. Document ingestion and chunking")
    print("  2. Embedding and vector storage")
    print("  3. Semantic search and retrieval")
    print("  4. Context-aware answer generation")
    print("  5. Source attribution")
    print("  6. Handling questions outside document scope")
    print("\n[OK] Local LLM (phi3:mini via Ollama) working correctly")
    print("[OK] All components integrated successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
