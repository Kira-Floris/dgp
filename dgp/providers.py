from typing import Protocol, runtime_checkable, Optional
import os

import openai
import groq

from dgp.config import ModelConfig

@runtime_checkable
class ModelProvider(Protocol):
    """Protocol for translation providers."""
    
    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig
    ) -> str:
        """Translate text from source to target language."""
        ...
    
    def get_provider_name(self) -> str:
        """Return the name of the translation provider."""
        ...

class OpenAIProvider:
    """OpenAI-based translation provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        self.client = openai.OpenAI(api_key=self.api_key)
    
    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI translation error: {e}")
    
    def get_provider_name(self) -> str:
        return "OpenAI"

class GroqProvider:
    """Groq-based translation provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key must be provided or set in GROQ_API_KEY environment variable"
            )

        self.client = groq.Groq(api_key=self.api_key)

    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig
    ) -> str:

        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Groq translation error: {e}")

    def get_provider_name(self) -> str:
        return "Groq"