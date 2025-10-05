# providers/openai_provider.py
import logging
from typing import Optional, Any
import io
import base64

# Langchain imports for OpenAI LLM and Embedding models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Direct OpenAI client for audio services
from openai import AsyncOpenAI, APIError, APIConnectionError

# Import our abstract base classes (the contracts)
from providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseTranscriptionProvider,
    BaseTTSProvider
)

logger = logging.getLogger(__name__)

# This file contains the concrete implementations for OpenAI services.

class OpenAILLMProvider(BaseLLMProvider):
    """Concrete LLM provider for OpenAI models."""

    def get_chat_model(self, api_key: str, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        """Initializes a Langchain ChatOpenAI model."""
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        try:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Failed to initialize ChatOpenAI model '{model_name}': {e}")
            raise

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Concrete embedding provider for OpenAI models."""

    def get_embedding_model(self, api_key: str, model_name: str, **kwargs: Any) -> Embeddings:
        """Initializes a Langchain OpenAIEmbeddings model."""
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        try:
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Failed to initialize OpenAIEmbeddings model '{model_name}': {e}")
            raise

class OpenAITranscriptionProvider(BaseTranscriptionProvider):
    """Concrete transcription provider using OpenAI's audio endpoint."""

    async def transcribe_audio(
        self,
        api_key: str,
        audio_file_bytes: bytes,
        filename: str,
        language: Optional[str],
        model_name: str
    ) -> Optional[str]:
        """Transcribes audio using a specialized OpenAI model."""
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        logger.info(f"Transcribing '{filename}' using OpenAI model '{model_name}'...")
        try:
            client = AsyncOpenAI(api_key=api_key)
            audio_file_object = io.BytesIO(audio_file_bytes)
            
            transcription_response = await client.audio.transcriptions.create(
                model=model_name,
                file=(filename, audio_file_object),
                language=language
            )
            return transcription_response.text
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error during transcription: {e.__cause__}")
            return None
        except APIError as e:
            logger.error(f"OpenAI API error during transcription: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during OpenAI transcription: {e}")
            return None

class OpenAITTSProvider(BaseTTSProvider):
    """Concrete TTS provider using OpenAI's audio endpoint."""

    async def synthesize_speech(
        self,
        api_key: str,
        text: str,
        model_name: str,
        voice_name: str,
        **kwargs: Any
    ) -> Optional[bytes]:
        """Synthesizes speech using a specialized OpenAI model."""
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        logger.info(f"Synthesizing speech with OpenAI model '{model_name}'...")
        try:
            client = AsyncOpenAI(api_key=api_key)

            # CORRECTED: Build the API parameters dictionary and only add
            # the 'instructions' key if it has a non-empty value.
            api_params = {
                "model": model_name,
                "voice": voice_name,
                "input": text,
                "response_format": "mp3"
            }
            if kwargs.get("instructions"):
                api_params["instructions"] = kwargs["instructions"]

            async with client.audio.speech.with_streaming_response.create(**api_params) as response:
                if response.status_code != 200:
                    error_content = await response.content()
                    logger.error(f"OpenAI TTS API error: {response.status_code} - {error_content.decode(errors='ignore')}")
                    return None
                return await response.read()
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error during TTS: {e.__cause__}")
            return None
        except APIError as e:
            error_message = f"Status: {getattr(e, 'status_code', 'N/A')} - Message: {e.message}"
            logger.error(f"OpenAI API error during TTS: {error_message}")
            return None
        except Exception as e:
            logger.exception(f"An unexpected error occurred during OpenAI TTS: {e}")
            return None
