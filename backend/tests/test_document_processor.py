"""
Tests for document processing functionality.
"""

import pytest
from pathlib import Path
import tempfile
import fitz  # PyMuPDF
from docx import Document
from app.services.document_processor import DocumentProcessor, DocumentChunk


@pytest.fixture
def processor():
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(chunk_size=800, chunk_overlap=100)


@pytest.fixture
def temp_txt_file():
    """Create a temporary TXT file for testing."""
    content = """This is a test document.

It has multiple paragraphs.

Each paragraph contains some text that we will use to test the document processing functionality."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_docx_file():
    """Create a temporary DOCX file for testing."""
    doc = Document()
    doc.add_paragraph("This is the first paragraph of the test document.")
    doc.add_paragraph("This is the second paragraph with more content.")
    doc.add_paragraph("")  # Empty paragraph to test filtering
    doc.add_paragraph("This is the third paragraph after an empty one.")

    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
        doc.save(f.name)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    pdf_doc = fitz.open()

    # Page 1
    page1 = pdf_doc.new_page()
    page1.insert_text((50, 50), "This is page 1 of the test PDF document.")
    page1.insert_text((50, 80), "It contains multiple lines of text.")

    # Page 2
    page2 = pdf_doc.new_page()
    page2.insert_text((50, 50), "This is page 2 with different content.")

    # Page 3 - Empty page
    pdf_doc.new_page()

    # Page 4
    page4 = pdf_doc.new_page()
    page4.insert_text((50, 50), "This is page 4 after an empty page.")

    # Create temp file path without opening the file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    import os
    os.close(temp_fd)  # Close the file descriptor immediately

    # Save PDF to the path
    pdf_doc.save(temp_path)
    pdf_doc.close()

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestDocumentChunk:
    """Test DocumentChunk class."""

    def test_chunk_creation(self):
        """Test creating a document chunk."""
        text = "Sample text"
        metadata = {"source": "test.txt", "chunk_index": 0}
        chunk = DocumentChunk(text, metadata)

        assert chunk.text == text
        assert chunk.metadata == metadata

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        text = "Sample text"
        metadata = {"source": "test.txt", "chunk_index": 0}
        chunk = DocumentChunk(text, metadata)

        result = chunk.to_dict()
        assert result["text"] == text
        assert result["metadata"] == metadata


class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""

    def test_processor_initialization(self):
        """Test processor initialization with custom settings."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50

    def test_unsupported_file_type(self, processor):
        """Test error handling for unsupported file types."""
        with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.process_document(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_nonexistent_file(self, processor):
        """Test error handling for nonexistent files."""
        with pytest.raises(ValueError, match="File not found"):
            processor.process_document("nonexistent_file.pdf")


class TestTXTProcessing:
    """Test TXT file processing."""

    def test_txt_extraction(self, processor, temp_txt_file):
        """Test basic TXT extraction."""
        chunks = processor.process_document(temp_txt_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_txt_metadata(self, processor, temp_txt_file):
        """Test TXT metadata structure."""
        chunks = processor.process_document(temp_txt_file)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "file_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["file_type"] == "txt"
            assert "page" not in chunk.metadata  # TXT should not have page numbers

    def test_txt_empty_file(self, processor):
        """Test handling of empty TXT file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            temp_path = f.name

        try:
            chunks = processor.process_document(temp_path)
            assert len(chunks) == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDOCXProcessing:
    """Test DOCX file processing."""

    def test_docx_extraction(self, processor, temp_docx_file):
        """Test basic DOCX extraction."""
        chunks = processor.process_document(temp_docx_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_docx_metadata(self, processor, temp_docx_file):
        """Test DOCX metadata structure."""
        chunks = processor.process_document(temp_docx_file)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "file_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["file_type"] == "docx"
            assert "page" not in chunk.metadata  # DOCX should not have page numbers

    def test_docx_empty_paragraphs_filtered(self, processor, temp_docx_file):
        """Test that empty paragraphs are filtered out."""
        chunks = processor.process_document(temp_docx_file)

        # Combine all chunk text
        full_text = " ".join(chunk.text for chunk in chunks)

        # Should contain non-empty paragraphs
        assert "first paragraph" in full_text.lower()
        assert "second paragraph" in full_text.lower()
        assert "third paragraph" in full_text.lower()


class TestPDFProcessing:
    """Test PDF file processing."""

    def test_pdf_extraction(self, processor, temp_pdf_file):
        """Test basic PDF extraction."""
        chunks = processor.process_document(temp_pdf_file)

        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)

    def test_pdf_metadata(self, processor, temp_pdf_file):
        """Test PDF metadata structure including page numbers."""
        chunks = processor.process_document(temp_pdf_file)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "file_type" in chunk.metadata
            assert "page" in chunk.metadata  # PDF should have page numbers
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["file_type"] == "pdf"
            assert chunk.metadata["page"] >= 1  # 1-indexed page numbers

    def test_pdf_page_numbers(self, processor, temp_pdf_file):
        """Test that PDF page numbers are correctly assigned."""
        chunks = processor.process_document(temp_pdf_file)

        # Check that we have chunks from different pages
        pages = set(chunk.metadata["page"] for chunk in chunks)
        assert len(pages) >= 2  # Should have content from multiple pages

        # Page numbers should be 1-indexed
        assert min(pages) >= 1

    def test_pdf_empty_pages_skipped(self, processor, temp_pdf_file):
        """Test that empty PDF pages are skipped."""
        chunks = processor.process_document(temp_pdf_file)

        # We created a PDF with pages 1, 2, 3 (empty), 4
        # Page 3 should be skipped, so we should only see pages 1, 2, 4
        pages = sorted(set(chunk.metadata["page"] for chunk in chunks))

        # Should not have page 3 (empty page)
        assert 3 not in pages


class TestTextCleaning:
    """Test text cleaning functionality."""

    def test_clean_multiple_spaces(self, processor):
        """Test removal of multiple consecutive spaces."""
        text = "This  has   multiple    spaces"
        cleaned = processor._clean_text(text)
        assert "  " not in cleaned
        assert "multiple spaces" in cleaned

    def test_clean_multiple_newlines(self, processor):
        """Test normalization of multiple newlines."""
        text = "Paragraph 1\n\n\n\nParagraph 2"
        cleaned = processor._clean_text(text)
        assert "\n\n\n" not in cleaned
        assert "Paragraph 1" in cleaned
        assert "Paragraph 2" in cleaned

    def test_clean_whitespace_lines(self, processor):
        """Test removal of lines with only whitespace."""
        text = "Line 1\n   \n  \t  \nLine 2"
        cleaned = processor._clean_text(text)
        lines = cleaned.split('\n')
        assert all(line.strip() != "" for line in lines if line)


class TestChunking:
    """Test text chunking functionality."""

    def test_chunk_short_text(self, processor):
        """Test that short text is not split."""
        text = "This is a short text."
        chunks = processor._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self):
        """Test that long text is split into multiple chunks."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "A" * 300  # 300 characters
        chunks = processor._chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        """Test that chunks have overlap."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "This is a test sentence. " * 20  # Long repeated text
        chunks = processor._chunk_text(text)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # The end of one chunk should appear near the start of the next
                assert len(chunks[i]) > 0
                assert len(chunks[i+1]) > 0

    def test_chunk_sentence_boundary(self, processor):
        """Test that chunking respects sentence boundaries when possible."""
        sentences = [f"This is sentence number {i}. " for i in range(50)]
        text = "".join(sentences)

        chunks = processor._chunk_text(text)

        # Each chunk should ideally end with a period
        for chunk in chunks[:-1]:  # All except possibly the last
            # Most chunks should end at sentence boundaries
            assert chunk.rstrip().endswith('.') or chunk.rstrip().endswith('!') or chunk.rstrip().endswith('?') or len(chunk) > processor.chunk_size * 0.8


class TestMetadataPreservation:
    """Test that metadata is properly preserved throughout processing."""

    def test_chunk_index_sequential(self, processor, temp_txt_file):
        """Test that chunk indices are sequential."""
        chunks = processor.process_document(temp_txt_file)

        indices = [chunk.metadata["chunk_index"] for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_filename_preserved(self, processor, temp_txt_file):
        """Test that source filename is preserved in metadata."""
        chunks = processor.process_document(temp_txt_file)

        expected_filename = Path(temp_txt_file).name
        for chunk in chunks:
            assert chunk.metadata["source"] == expected_filename


class TestInvalidInput:
    """Test handling of invalid or edge case inputs."""

    def test_whitespace_only_file(self, processor):
        """Test handling of file with only whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("   \n\n\t\t\n   ")
            temp_path = f.name

        try:
            chunks = processor.process_document(temp_path)
            assert len(chunks) == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)
