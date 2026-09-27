# RAG Document Q&A System

## Project Purpose
A complete Retrieval-Augmented Generation (RAG) application where users can upload PDF, DOCX, and TXT documents and ask questions about them. The system retrieves relevant document chunks and uses a locally running LLM to generate grounded answers with source/page citations.

## Planned Technology Stack
* **Frontend:** React + Vite
* **Backend:** Python + FastAPI
* **Vector Database:** ChromaDB
* **Embeddings:** Sentence Transformers (local)
* **LLM Engine:** Ollama (local)
* **Document Processing:** PyMuPDF (PDFs) and python-docx (DOCX)

## High-Level RAG Pipeline
1. **Document Upload** → Documents received via API
2. **Text Extraction & Cleaning** → Raw text retrieved, page numbers tracked
3. **Chunking** → Text split into manageable, overlapping chunks
4. **Embedding Generation** → Text chunks converted into vector representations
5. **ChromaDB Storage** → Embeddings indexed for fast retrieval
6. **Similarity Retrieval** → User queries mapped to relevant context chunks
7. **Local LLM via Ollama** → Generated answer grounded *strictly* in retrieved text
8. **Source Citations** → Delivered back to the UI indicating document & page origin

## Current Development Status
**Phase 1** - Environment Setup & Foundation (Ongoing)
* Initialized folder structure
* Established Python Virtual Environment
* Defined placeholder configuration and architecture
* No dependencies installed or models downloaded yet.
