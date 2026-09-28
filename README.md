# RAG Document Q&A System

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content.

The system retrieves the most relevant document chunks from a local vector database and uses a locally running LLM to generate grounded answers with source citations.

![RAG Document Q&A Dashboard](docs/dashboard.png)

---

## Overview

The RAG Document Q&A System is designed to make information retrieval from personal documents easier.

Users can upload:

- PDF
- DOCX
- TXT

documents and ask natural-language questions about their contents.

Instead of sending the entire document to the language model, the system:

1. Extracts the document text
2. Splits the text into smaller chunks
3. Converts chunks into vector embeddings
4. Stores the embeddings in ChromaDB
5. Retrieves the most relevant chunks for a question
6. Sends only the relevant context to the local LLM
7. Generates a grounded answer
8. Displays the document source and page information when available

The application is designed to answer questions based on the uploaded documents rather than relying on external information.

---

## Key Features

### Document Upload

Upload PDF, DOCX, and TXT documents through the web interface.

### Document Processing

Documents are automatically:

- Extracted
- Cleaned
- Split into overlapping chunks
- Converted into embeddings
- Stored in the vector database

### Semantic Search

User questions are converted into embeddings and compared against document embeddings to retrieve relevant information.

### Retrieval-Augmented Generation

The retrieved document context is provided to the LLM before generating the final answer.

### Grounded Answers

The RAG service instructs the model to answer using the supplied document context and avoid inventing information that is not present in the retrieved documents.

### Source Citations

Answers can include source information such as:

- Document name
- Page number for PDFs
- Chunk information

### Document Management

Users can:

- View indexed documents
- See document chunk counts
- Delete documents from the knowledge base

### Local AI

The application uses local AI components for embeddings and generation, allowing the core RAG workflow to run without a paid LLM API.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │      React UI       │
                    │      + Vite         │
                    └──────────┬──────────┘
                               │
                               │ REST API
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ Document   │  │ Embedding  │  │    RAG     │
        │ Processor  │  │  Service   │  │  Service   │
        └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
              │               │               │
              ▼               │               ▼
        ┌────────────┐        │        ┌────────────┐
        │ PDF/DOCX/  │        │        │  Ollama    │
        │ TXT Parser │        │        │ phi3:mini   │
        └────────────┘        │        └────────────┘
                              │
                              ▼
                       ┌────────────┐
                       │ ChromaDB   │
                       │ Vector DB  │
                       └────────────┘
