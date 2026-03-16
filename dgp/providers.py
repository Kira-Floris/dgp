from typing import Protocol, runtime_checkable, Optional
import os

import openai
import groq
from together import Together
from google import genai
from transformers import pipeline
import torch

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
        if "-5" in config.model_name:
            config.temperature = 1.0
        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                temperature=config.temperature,
                max_completion_tokens=config.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")
    
    def get_provider_name(self) -> str:
        return "OpenAI"

class TogetherAIProvider:
    """TogetherAI-based translation provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Together API key must be provided or set in TOGETHER_API_KEY environment variable"
            )

        # Initialize Together client
        self.client = Together(api_key=self.api_key)

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
            raise RuntimeError(f"TogetherAI error: {e}")

    def get_provider_name(self) -> str:
        return "TogetherAI"

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
            raise RuntimeError(f"Groq error: {e}")

    def get_provider_name(self) -> str:
        return "Groq"

class VLLMProvider:
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        # vLLM doesn't require an API key by default, but setting a dummy one
        # or using provided key if server has authentication enabled
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        
        # Create OpenAI client pointing to vLLM server
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=3
        )
    
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
            raise RuntimeError(f"vLLM error: {e}")
    
    def get_provider_name(self) -> str:
        return "vLLM"

class GeminiProvider:
    """Google Gemini-based translation provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key must be provided or set in GEMINI_API_KEY environment variable"
            )

        # genai.configure(api_key=self.api_key)

    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig
    ) -> str:
        try:
            # model = genai.Client(config.model_name)
            client = genai.Client(api_key=self.api_key)

            prompt = f"{system}\n\n{text}"

            response = client.models.generate_content(
                model=config.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens
                )
            )

            return response.text.strip()

        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")

    def get_provider_name(self) -> str:
        return "Gemini"

# ###################################################
# Translation Models
# ###################################################

class NLLBProvider:
    """NLLB-based translation provider using HuggingFace pipeline."""

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "kin_Latn",
        device: int = -1
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            src_lang: source language code
            tgt_lang: target language code
            device: -1 for CPU, >=0 for GPU
        """

        self.translator = pipeline(
            "translation",
            model=model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=device,
            # device_map="auto",
            dtype=torch.float16
        )

    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig
    ) -> str:
        """
        Run translation using the NLLB pipeline.
        """

        try:
            result = self.translator(
                text,
                max_length=config.max_tokens or 512,
                # max_new_tokens=config.max_tokens or 512,
                truncation=True,
            )

            return result[0]["translation_text"].strip()

        except Exception as e:
            raise RuntimeError(f"NLLB error: {e}")
        
    def get_provider_name(self) -> str:
        return "Facebook"

class TranslateGemmaProvider:
    """
    TranslateGemma translation provider using HuggingFace's image-text-to-text pipeline.

    Args:
        model_name: HuggingFace model identifier.
        src_lang:   ISO 639-1 source language code (e.g. "en", "rw", "fr").
        tgt_lang:   ISO 639-1 target language code.
        device:     "cpu", "cuda", "cuda:0", etc. Defaults to CUDA if available.
        dtype:      Torch dtype. Defaults to bfloat16 on GPU, float32 on CPU.
    """

    def __init__(
        self,
        model_name: str = "google/translategemma-4b-it",
        src_lang: str = "en",
        tgt_lang: str = "rw",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype is None:
            dtype = torch.bfloat16 if device != "cpu" else torch.float32

        self.translator = pipeline(
            "image-text-to-text",
            model=model_name,
            # device=device,
            device_map="auto",
            torch_dtype=dtype,
        )

    def _build_messages(self, text: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.src_lang,
                        "target_lang_code": self.tgt_lang,
                        "text": text,
                    }
                ],
            }
        ]

    def invoke(
        self,
        text: str,
        system: str,
        config: ModelConfig,
    ) -> str:
        try:
            output = self.translator(
                text=self._build_messages(text),
                max_new_tokens=config.max_tokens or 512,
            )
            return output[0]["generated_text"][-1]["content"].strip()

        except Exception as e:
            raise RuntimeError(f"TranslateGemma error: {e}")

    def get_provider_name(self) -> str:
        return "Google"