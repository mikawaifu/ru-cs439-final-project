"""LLM backend implementations for candidate generation."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM backend.
        
        Args:
            config: Backend configuration
        """
        self.config = config
        self.model = config.get("model")
    
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a response.
        
        Args:
            system_prompt: System message
            user_prompt: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Returns:
            Generated text
        """
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get API key from environment
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        # Get API base (optional)
        api_base_env = config.get("api_base_env", "OPENAI_API_BASE")
        self.api_base = os.getenv(api_base_env, "https://api.openai.com/v1")
        
        # Import OpenAI library
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        except ImportError:
            raise ImportError("openai library not installed. Run: pip install openai")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens or self.config.get("max_tokens", 256),
            temperature=temperature or self.config.get("temperature", 0.8),
            top_p=top_p or self.config.get("top_p", 0.95),
        )
        
        return response.choices[0].message.content


class XAIBackend(LLMBackend):
    """xAI Grok API backend (compatible with OpenAI API format)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get API key from environment
        api_key_env = config.get("api_key_env", "XAI_API_KEY")
        self.api_key = os.getenv(api_key_env)
        
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        
        # Get API base
        api_base_env = config.get("api_base_env", "XAI_API_BASE")
        self.api_base = os.getenv(api_base_env, "https://api.x.ai/v1")
        
        # Use OpenAI client with custom base URL
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        except ImportError:
            raise ImportError("openai library not installed. Run: pip install openai")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate response using xAI Grok API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens or self.config.get("max_tokens", 256),
            temperature=temperature or self.config.get("temperature", 0.8),
            top_p=top_p or self.config.get("top_p", 0.95),
        )
        
        return response.choices[0].message.content


class VLLMBackend(LLMBackend):
    """vLLM server backend (OpenAI-compatible API)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get endpoint from environment
        endpoint_env = config.get("endpoint_env", "VLLM_ENDPOINT")
        self.endpoint = os.getenv(endpoint_env)
        
        if not self.endpoint:
            raise ValueError(f"Environment variable {endpoint_env} not set")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate response using vLLM server."""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens or self.config.get("max_tokens", 256),
            "temperature": temperature or self.config.get("temperature", 0.8),
            "top_p": top_p or self.config.get("top_p", 0.95),
        }
        
        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


class OllamaBackend(LLMBackend):
    """Ollama backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get host from environment
        host_env = config.get("host_env", "OLLAMA_HOST")
        self.host = os.getenv(host_env, "http://localhost:11434")
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Generate response using Ollama."""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens or self.config.get("max_tokens", 256),
                "temperature": temperature or self.config.get("temperature", 0.8),
                "top_p": top_p or self.config.get("top_p", 0.95),
            },
        }
        
        response = requests.post(
            f"{self.host}/api/chat",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]


def create_backend(name: str, config: Dict[str, Any]) -> LLMBackend:
    """
    Create an LLM backend instance.
    
    Args:
        name: Backend name
        config: Backend configuration
        
    Returns:
        LLM backend instance
    """
    provider = config.get("provider", "").lower()
    
    if provider == "openai":
        return OpenAIBackend(config)
    elif provider == "xai":
        return XAIBackend(config)
    elif provider == "vllm":
        return VLLMBackend(config)
    elif provider == "ollama":
        return OllamaBackend(config)
    else:
        raise ValueError(f"Unknown backend provider: {provider}")


