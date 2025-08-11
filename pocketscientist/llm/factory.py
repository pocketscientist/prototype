"""
Factory for creating LLM providers.
"""

from typing import Optional
from urllib.parse import urlparse

from .base import BaseLLMProvider
from .ollama import OllamaProvider


def create_llm_provider(
    endpoint: str,
    model_name: str,
    api_key: Optional[str] = None,
    provider_type: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create an LLM provider based on the endpoint URL or explicit provider type.
    
    Args:
        endpoint: The LLM endpoint URL
        model_name: The model name to use
        api_key: Optional API key
        provider_type: Explicit provider type ('ollama', 'openai', etc.)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        An instance of the appropriate LLM provider
    
    Raises:
        ValueError: If the provider type cannot be determined or is unsupported
    """
    if provider_type:
        provider_type = provider_type.lower()
    else:
        # Try to infer provider from endpoint
        provider_type = _infer_provider_from_endpoint(endpoint)
    
    if provider_type == "ollama":
        return OllamaProvider(
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
    elif provider_type == "openai":
        # Placeholder for future OpenAI implementation
        raise NotImplementedError("OpenAI provider not yet implemented")
    elif provider_type == "anthropic":
        # Placeholder for future Anthropic implementation
        raise NotImplementedError("Anthropic provider not yet implemented")
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. "
            f"Supported providers: ollama"
        )


def _infer_provider_from_endpoint(endpoint: str) -> str:
    """
    Infer the provider type from the endpoint URL.
    
    Args:
        endpoint: The LLM endpoint URL
    
    Returns:
        The inferred provider type
    """
    parsed = urlparse(endpoint)
    
    # Common patterns for different providers
    if "11434" in endpoint or "ollama" in parsed.hostname or parsed.hostname == "localhost":
        return "ollama"
    elif "api.openai.com" in endpoint:
        return "openai"
    elif "api.anthropic.com" in endpoint:
        return "anthropic"
    else:
        # Default to Ollama for unknown endpoints (common self-hosted pattern)
        return "ollama"


def list_supported_providers() -> list[str]:
    """Return a list of supported provider types."""
    return ["ollama"]