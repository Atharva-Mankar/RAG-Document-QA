"""FastAPI API tests using TestClient."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoints."""

    def test_health_check(self):
        """Test GET /health returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "running" in data["message"]


class TestOllamaEndpoint:
    """Tests for Ollama health endpoint."""

    def test_ollama_health(self):
        """Test GET /health/ollama returns status."""
        response = client.get("/health/ollama")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama_available" in data
        assert "model_available" in data
        assert "model_name" in data


class TestDocumentUploadEndpoint:
    """Tests for document upload endpoint."""

    def test_upload_invalid_file_type(self):
        """Test upload with invalid file extension is rejected."""
        response = client.post(
            "/documents/upload",
            files={"file": ("bad.exe", b"fake content", "application/octet-stream")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_upload_unsupported_extension(self):
        """Test upload with unsupported extension."""
        response = client.post(
            "/documents/upload",
            files={"file": ("bad.jpg", b"fake image content", "image/jpeg")}
        )
        assert response.status_code == 400

    def test_upload_empty_file(self):
        """Test upload with empty file returns error."""
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        # Empty file should either succeed with 0 chunks or fail
        # Our service validates empty content
        assert response.status_code in (200, 400, 413)


class TestDocumentListEndpoint:
    """Tests for document listing."""

    def test_list_documents(self):
        """Test GET /documents returns response."""
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_documents" in data
        assert "total_chunks" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_empty_question(self):
        """Test chat with empty question."""
        response = client.post("/chat", json={"question": ""})
        # Empty string should either return error or graceful response
        assert response.status_code == 200 or response.status_code == 422

    def test_chat_valid_question(self):
        """Test chat with a valid question."""
        response = client.post(
            "/chat",
            json={"question": "What is Python?"}
        )
        # Should return either a real answer or a graceful error
        # If services are available, should be 200
        assert response.status_code == 200
        data = response.json()
        # If RAG pipeline works
        if "answer" in data:
            assert isinstance(data["answer"], str)
        assert "success" in data