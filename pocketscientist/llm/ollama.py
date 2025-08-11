"""
Ollama LLM provider implementation.
"""

import json
from typing import Dict, Any, List, Optional
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout

from .base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model_name: str = "llama3.1",
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any
    ) -> None:
        super().__init__(endpoint, model_name, api_key, **kwargs)
        self.timeout = timeout
        
        # Ensure endpoint doesn't end with slash
        self.endpoint = self.endpoint.rstrip('/')
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response from Ollama."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature, max_tokens, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Have a chat conversation with Ollama."""
        url = f"{self.endpoint}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        # Add max_tokens if specified (Ollama uses num_predict)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"].strip()
            else:
                raise ValueError(f"Unexpected response format from Ollama: {result}")
                
        except ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.endpoint}. "
                "Make sure Ollama is running and accessible."
            )
        except Timeout:
            raise TimeoutError(
                f"Request to Ollama timed out after {self.timeout} seconds."
            )
        except RequestException as e:
            raise RuntimeError(f"Error communicating with Ollama: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Ollama: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available and responding."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if the specific model is available
            tags = response.json()
            available_models = [model["name"] for model in tags.get("models", [])]
            
            # Check if our model is in the list (handle version tags)
            model_available = any(
                self.model_name in model or model.startswith(f"{self.model_name}:")
                for model in available_models
            )
            
            if not model_available:
                # Try to pull the model if it's not available
                self._pull_model()
                
            return True
            
        except Exception:
            return False
    
    def _pull_model(self) -> None:
        """Attempt to pull the model if it's not available."""
        try:
            url = f"{self.endpoint}/api/pull"
            payload = {"name": self.model_name}
            
            # This is a synchronous pull - in production you might want to make this async
            response = requests.post(
                url,
                json=payload,
                timeout=300,  # 5 minutes for model pull
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
        except Exception:
            # If pull fails, we'll let the model generation fail with a clearer error
            pass
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            
            tags = response.json()
            return [model["name"] for model in tags.get("models", [])]
            
        except Exception as e:
            raise RuntimeError(f"Error listing Ollama models: {e}")