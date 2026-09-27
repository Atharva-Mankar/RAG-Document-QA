"""
Simple demonstration of semantic retrieval without Unicode output issues.
"""

from pathlib import Path
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
import tempfile
import shutil


# Create sample document
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

# Process document
print("Processing document...")
processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
chunks = processor.process_document(temp_file)
print(f"Extracted {len(chunks)} chunks")
print()

# Setup vector store
temp_dir = tempfile.mkdtemp()
store = VectorStore(persist_directory=temp_dir, collection_name="demo")
print(f"ChromaDB initialized")
print(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding dimension: {store.embedding_service.get_embedding_dimension()}")
print()

# Store chunks
print("Storing chunks with embeddings...")
store.add_chunks(chunks)
print(f"Stored {store.get_collection_count()} chunks")
print()

# Query 1
print("=" * 70)
query = "What is deep learning?"
print(f"Query: {query}")
print("-" * 70)
results = store.search(query, top_k=2)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Text: {result['text'][:120]}...")
    print(f"Source: {result['metadata']['source']}")
    print(f"Chunk: {result['metadata']['chunk_index']}")
    print(f"Distance: {result['distance']:.4f}")

# Query 2
print()
print("=" * 70)
query = "How is AI used in healthcare?"
print(f"Query: {query}")
print("-" * 70)
results = store.search(query, top_k=1)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Distance: {result['distance']:.4f}")

print()
print("=" * 70)
print("Demo complete - Phase 4 functional")
print("=" * 70)

# Cleanup
Path(temp_file).unlink(missing_ok=True)
shutil.rmtree(temp_dir, ignore_errors=True)
