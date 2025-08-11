"""
Base LLM provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Have a chat conversation with the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and responding."""
        pass
    
    def validate_connection(self) -> Dict[str, Any]:
        """Validate the connection to the LLM provider."""
        try:
            available = self.is_available()
            return {
                "success": available,
                "endpoint": self.endpoint,
                "model": self.model_name,
                "error": None if available else "Connection failed"
            }
        except Exception as e:
            return {
                "success": False,
                "endpoint": self.endpoint,
                "model": self.model_name,
                "error": str(e)
            }