# RAG Document Q&A System Architecture

## Core Design Principles
1. **Separation of Concerns**: Clear boundaries between document processing, retrieval, and generation.
2. **Stateless API**: No session management in backend.
3. **Local-First**: Everything runs locally, no external APIs.
4. **Hallucination Prevention**: System prompts instruct LLM to only answer from context.
5. **Citation-Ready**: Track source documents and page numbers throughout the pipeline.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER (React)                     │
│  • Document Upload UI                                        │
│  • Chat Interface                                            │
│  • Document Management                                       │
│  • Source Citation Display                                   │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────▼────────────────────────────────────────┐
│                  API LAYER (FastAPI)                         │
│  • /upload endpoint                                          │
│  • /query endpoint                                           │
│  • /documents endpoint                                       │
│  • /health endpoint                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────┐         ┌────────▼────────┐
│   DOCUMENT   │         │   RAG ENGINE    │
│  PROCESSOR   │         │                 │
│              │         │  • Query        │
│ • PDF Reader │         │  • Retrieval    │
│ • DOCX Reader│         │  • Generation   │
│ • TXT Reader │         │  • Citation     │
│ • Chunking   │         └────────┬────────┘
│ • Cleaning   │                  │
└───────┬──────┘         ┌────────┴────────┐
        │                │                 │
        │         ┌──────▼──────┐   ┌──────▼──────┐
        │         │  ChromaDB   │   │   Ollama    │
        └────────►│  (Vector)   │   │  (LLM API)  │
                  └─────────────┘   └─────────────┘
```
