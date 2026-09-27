"""
LLM Service for local Ollama integration.

Communicates with Ollama running locally to generate responses.
"""

import requests
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with local Ollama LLM."""

    def __init__(self, model_name: str = "phi3:mini", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM service.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Generate a response from the local LLM.

        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to guide the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds

        Returns:
            Dict containing:
                - response: Generated text
                - model: Model name used
                - success: Whether generation succeeded
                - error: Error message if failed

        Raises:
            ConnectionError: If cannot connect to Ollama
            TimeoutError: If request times out
        """
        try:
            # Build the full prompt with system prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            logger.info(f"Sending request to Ollama model: {self.model_name}")

            # Make request to Ollama
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()

            # Parse response
            result = response.json()

            generated_text = result.get("response", "").strip()

            logger.info(f"Successfully generated response ({len(generated_text)} chars)")

            return {
                "response": generated_text,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

        except requests.exceptions.Timeout as e:
            error_msg = f"Request to Ollama timed out after {timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg) from e

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_msg = f"Model '{self.model_name}' not found. Please run: ollama pull {self.model_name}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
            else:
                error_msg = f"Ollama API error: {e.response.status_code} - {e.response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = f"Unexpected error during LLM generation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def check_health(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]

            if self.model_name in model_names or f"{self.model_name}:latest" in model_names:
                logger.info(f"Health check passed: {self.model_name} is available")
                return True
            else:
                logger.warning(f"Model {self.model_name} not found in available models: {model_names}")
                return False

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
