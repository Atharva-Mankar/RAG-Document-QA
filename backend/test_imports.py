import sys
import fastapi
import uvicorn
import fitz
import docx
import chromadb
import sentence_transformers
import requests
import pydantic
import dotenv

print("---")
print("Python version:", sys.version.split()[0])
print("FastAPI:", fastapi.__version__)
print("Uvicorn:", uvicorn.__version__)
print("PyMuPDF:", fitz.__version__)
print("python-docx:", docx.__version__)
print("ChromaDB:", chromadb.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)
print("Requests:", requests.__version__)
print("Pydantic:", pydantic.__version__)
try:
    print("Dotenv:", dotenv.__version__)
except AttributeError:
    print("Dotenv: installed (version not exposed)")
print("---")
print("ALL IMPORTS SUCCESSFUL")
