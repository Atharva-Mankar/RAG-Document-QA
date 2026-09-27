"""
RAG Service for document question answering.

Combines embedding, vector search, and LLM generation for grounded Q&A.
"""

from typing import List, Dict, Any, Optional
import logging

from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStore,
        llm_service: LLMService,
        top_k: int = 5
    ):
        """
        Initialize RAG service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store_service: Vector store for similarity search
            llm_service: Service for LLM text generation
            top_k: Number of relevant chunks to retrieve
        """
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.llm_service = llm_service
        self.top_k = top_k

        # RAG system prompt
        self.system_prompt = """You are a helpful assistant that answers questions based on provided document context.

IMPORTANT INSTRUCTIONS:
1. Answer questions using ONLY the information in the provided document context below.
2. Do NOT use external knowledge or make assumptions beyond what is explicitly stated in the documents.
3. If the answer cannot be found in the provided context, clearly state: "I cannot find this information in the provided documents."
4. Distinguish between information directly stated in the documents and any uncertainty.
5. Keep answers clear, concise, and factual.
6. When relevant information is found, reference the source document in your answer.
7. Do not invent or fabricate information.

Answer based solely on the context provided below."""

    def answer_question(
        self,
        question: str,
        collection_name: str = "documents",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.

        Pipeline:
        1. Embed the question
        2. Search for relevant chunks in vector store
        3. Build context from retrieved chunks
        4. Generate answer using LLM with context
        5. Return answer with sources

        Args:
            question: User's question
            collection_name: ChromaDB collection to search
            top_k: Number of chunks to retrieve (overrides default)

        Returns:
            Dict containing:
                - answer: Generated answer
                - sources: List of source documents with metadata
                - retrieved_chunks: Number of chunks retrieved
                - question: Original question
        """
        logger.info(f"Processing question: {question[:100]}...")

        k = top_k or self.top_k

        # Step 1: Embed the question
        try:
            question_embedding = self.embedding_service.embed_text(question)
            logger.info("Question embedded successfully")
        except Exception as e:
            logger.error(f"Failed to embed question: {str(e)}")
            raise

        # Step 2: Retrieve relevant chunks
        try:
            results = self.vector_store_service.search(
                query_embedding=question_embedding,
                collection_name=collection_name,
                top_k=k
            )
            logger.info(f"Retrieved {len(results)} relevant chunks")

            if not results:
                return {
                    "answer": "I cannot find this information in the provided documents. No relevant context was retrieved.",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "question": question
                }

        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise

        # Step 3: Build context from retrieved chunks
        context = self._build_context(results)
        logger.info(f"Built context with {len(context)} characters")

        # Step 4: Generate answer with LLM
        try:
            prompt = f"""CONTEXT:

{context}

QUESTION: {question}

ANSWER:"""

            llm_response = self.llm_service.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.3  # Lower temperature for more factual responses
            )

            answer = llm_response["response"]
            logger.info("Answer generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise

        # Step 5: Extract sources
        sources = self._extract_sources(results)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(results),
            "question": question
        }

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build formatted context from retrieved chunks.

        Args:
            results: List of search results with documents and metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            document_text = result.get("document", "")

            # Extract metadata
            source_file = metadata.get("source", "Unknown")
            page_num = metadata.get("page")
            chunk_idx = metadata.get("chunk_index", idx - 1)

            # Build context block
            context_block = f"--- SOURCE {idx} ---\n"
            context_block += f"FILE: {source_file}\n"

            if page_num is not None:
                context_block += f"PAGE: {page_num}\n"

            context_block += f"CHUNK: {chunk_idx}\n\n"
            context_block += f"{document_text}\n"

            context_parts.append(context_block)

        return "\n".join(context_parts)

    def _extract_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source information from search results.

        Args:
            results: List of search results

        Returns:
            List of source dictionaries with metadata
        """
        sources = []
        seen_sources = set()

        for result in results:
            metadata = result.get("metadata", {})

            source_file = metadata.get("source", "Unknown")
            page_num = metadata.get("page")
            chunk_idx = metadata.get("chunk_index", 0)

            # Create unique identifier
            source_id = f"{source_file}:{page_num}:{chunk_idx}"

            if source_id not in seen_sources:
                source_info = {
                    "source": source_file,
                    "chunk_index": chunk_idx
                }

                # Only include page if it exists
                if page_num is not None:
                    source_info["page"] = page_num

                sources.append(source_info)
                seen_sources.add(source_id)

        return sources

    def check_health(self) -> Dict[str, bool]:
        """
        Check health of all RAG components.

        Returns:
            Dict with health status of each component
        """
        return {
            "embedding_service": True,  # EmbeddingService doesn't have health check
            "vector_store": self.vector_store_service.collection_exists("documents"),
            "llm_service": self.llm_service.check_health()
        }
