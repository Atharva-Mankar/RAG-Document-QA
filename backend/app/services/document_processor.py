"""
Document processing service for extracting and chunking text from PDF, DOCX, and TXT files.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from docx import Document


class DocumentChunk:
    """Represents a text chunk with metadata."""

    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }


class DocumentProcessor:
    """Handles document extraction, cleaning, and chunking."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize document processor.

        Args:
            chunk_size: Target size for each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document file and return text chunks with metadata.

        Args:
            file_path: Path to the document file

        Returns:
            List of DocumentChunk objects

        Raises:
            ValueError: If file type is unsupported or file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        file_extension = path.suffix.lower()

        if file_extension == ".pdf":
            return self._process_pdf(path)
        elif file_extension == ".docx":
            return self._process_docx(path)
        elif file_extension == ".txt":
            return self._process_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _process_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """Extract text from PDF with page-level metadata."""
        chunks = []

        try:
            doc = fitz.open(file_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Skip completely empty pages
                if not text.strip():
                    continue

                # Clean the extracted text
                cleaned_text = self._clean_text(text)

                if not cleaned_text.strip():
                    continue

                # Chunk the page text
                page_chunks = self._chunk_text(cleaned_text)

                # Add metadata to each chunk
                for chunk_index, chunk_text in enumerate(page_chunks):
                    metadata = {
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_num + 1,  # 1-indexed page numbers
                        "chunk_index": len(chunks) + chunk_index
                    }
                    chunks.append(DocumentChunk(chunk_text, metadata))

            doc.close()

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

        return chunks

    def _process_docx(self, file_path: Path) -> List[DocumentChunk]:
        """Extract text from DOCX file."""
        chunks = []

        try:
            doc = Document(file_path)

            # Extract all paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:  # Ignore empty paragraphs
                    paragraphs.append(text)

            # Combine paragraphs into full text
            full_text = "\n\n".join(paragraphs)

            # Clean the text
            cleaned_text = self._clean_text(full_text)

            if not cleaned_text.strip():
                return chunks

            # Chunk the text
            text_chunks = self._chunk_text(cleaned_text)

            # Add metadata to each chunk
            for chunk_index, chunk_text in enumerate(text_chunks):
                metadata = {
                    "source": file_path.name,
                    "file_type": "docx",
                    "chunk_index": chunk_index
                }
                chunks.append(DocumentChunk(chunk_text, metadata))

        except Exception as e:
            raise ValueError(f"Error processing DOCX: {str(e)}")

        return chunks

    def _process_txt(self, file_path: Path) -> List[DocumentChunk]:
        """Extract text from TXT file."""
        chunks = []

        try:
            # Read with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean the text
            cleaned_text = self._clean_text(text)

            if not cleaned_text.strip():
                return chunks

            # Chunk the text
            text_chunks = self._chunk_text(cleaned_text)

            # Add metadata to each chunk
            for chunk_index, chunk_text in enumerate(text_chunks):
                metadata = {
                    "source": file_path.name,
                    "file_type": "txt",
                    "chunk_index": chunk_index
                }
                chunks.append(DocumentChunk(chunk_text, metadata))

        except UnicodeDecodeError:
            raise ValueError("File encoding error: unable to decode as UTF-8")
        except Exception as e:
            raise ValueError(f"Error processing TXT: {str(e)}")

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by normalizing whitespace.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with double newline (paragraph breaks)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence ending (. ! ?) followed by space
                sentence_break = text.rfind('. ', start, end)
                if sentence_break == -1:
                    sentence_break = text.rfind('! ', start, end)
                if sentence_break == -1:
                    sentence_break = text.rfind('? ', start, end)

                if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                    end = sentence_break + 1
                else:
                    # Look for word boundary (space)
                    space_break = text.rfind(' ', start, end)
                    if space_break != -1 and space_break > start + self.chunk_size // 2:
                        end = space_break

            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Ensure we make progress
            if start <= chunks[-1].find(text[start:start+10]) if chunks else 0:
                start = end

        return chunks
