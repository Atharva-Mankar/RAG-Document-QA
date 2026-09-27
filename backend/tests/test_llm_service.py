"""
Tests for LLM Service.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from app.services.llm_service import LLMService


class TestLLMService:
    """Test cases for LLM service."""

    def test_initialization(self):
        """Test LLM service initialization."""
        service = LLMService(model_name="phi3:mini", base_url="http://localhost:11434")

        assert service.model_name == "phi3:mini"
        assert service.base_url == "http://localhost:11434"
        assert service.generate_url == "http://localhost:11434/api/generate"

    @patch('requests.post')
    def test_generate_success(self, mock_post):
        """Test successful text generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "The answer is 42.",
            "model": "phi3:mini"
        }
        mock_post.return_value = mock_response

        service = LLMService()
        result = service.generate(prompt="What is the answer?")

        assert result["success"] is True
        assert result["response"] == "The answer is 42."
        assert result["model"] == "phi3:mini"
        assert result["error"] is None

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "phi3:mini"
        assert "What is the answer?" in call_args[1]["json"]["prompt"]

    @patch('requests.post')
    def test_generate_with_system_prompt(self, mock_post):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Based on the context, the answer is 42.",
            "model": "phi3:mini"
        }
        mock_post.return_value = mock_response

        service = LLMService()
        result = service.generate(
            prompt="What is the answer?",
            system_prompt="You are a helpful assistant."
        )

        assert result["success"] is True

        # Verify system prompt was included
        call_args = mock_post.call_args
        prompt_sent = call_args[1]["json"]["prompt"]
        assert "You are a helpful assistant" in prompt_sent
        assert "What is the answer?" in prompt_sent

    @patch('requests.post')
    def test_generate_with_parameters(self, mock_post):
        """Test generation with custom parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response

        service = LLMService()
        service.generate(
            prompt="Test",
            temperature=0.5,
            max_tokens=100
        )

        call_args = mock_post.call_args
        options = call_args[1]["json"]["options"]
        assert options["temperature"] == 0.5
        assert options["num_predict"] == 100

    @patch('requests.post')
    def test_generate_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = requests.exceptions.ConnectionError()

        service = LLMService()

        with pytest.raises(ConnectionError) as exc_info:
            service.generate(prompt="Test")

        assert "Cannot connect to Ollama" in str(exc_info.value)

    @patch('requests.post')
    def test_generate_timeout(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = requests.exceptions.Timeout()

        service = LLMService()

        with pytest.raises(TimeoutError) as exc_info:
            service.generate(prompt="Test", timeout=30)

        assert "timed out" in str(exc_info.value)

    @patch('requests.post')
    def test_generate_model_not_found(self, mock_post):
        """Test handling of model not found error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "model not found"
        mock_post.return_value = mock_response
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)

        service = LLMService(model_name="nonexistent:model")

        with pytest.raises(ValueError) as exc_info:
            service.generate(prompt="Test")

        assert "not found" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)

    @patch('requests.get')
    def test_check_health_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "phi3:mini"},
                {"name": "llama2:latest"}
            ]
        }
        mock_get.return_value = mock_response

        service = LLMService(model_name="phi3:mini")
        assert service.check_health() is True

    @patch('requests.get')
    def test_check_health_model_not_available(self, mock_get):
        """Test health check when model is not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "other:model"}]
        }
        mock_get.return_value = mock_response

        service = LLMService(model_name="phi3:mini")
        assert service.check_health() is False

    @patch('requests.get')
    def test_check_health_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()

        service = LLMService()
        assert service.check_health() is False


@pytest.mark.integration
class TestLLMServiceIntegration:
    """Integration tests with real Ollama instance."""

    def test_real_generation(self):
        """Test real generation with local Ollama (requires phi3:mini installed)."""
        service = LLMService(model_name="phi3:mini")

        # Check if Ollama is available
        if not service.check_health():
            pytest.skip("Ollama or phi3:mini model not available")

        # Test simple generation
        result = service.generate(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.1
        )

        assert result["success"] is True
        assert result["response"] is not None
        assert len(result["response"]) > 0
        assert "4" in result["response"]
