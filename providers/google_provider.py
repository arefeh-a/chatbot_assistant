# providers/google_provider.py
import logging
import os
from typing import Optional, Any
import io
import wave

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# GenAI SDK imports (for transcription & TTS)
from google import genai
from google.genai.types import Part, GenerateContentConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig

# Your abstract base classes
from providers.base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseTranscriptionProvider,
    BaseTTSProvider
)

logger = logging.getLogger(__name__)


def _set_proxy_env():
    """If a proxy is configured in HTTPS_PROXY or HTTP_PROXY, inject it into the process."""
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


class GoogleLLMProvider(BaseLLMProvider):
    """This class is not used in the current test but is kept for completeness."""
    def get_chat_model(self, api_key: str, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        pass # Implementation omitted as it's not part of the current issue

class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """This class is not used in the current test but is kept for completeness."""
    def get_embedding_model(self, api_key: str, model_name: str, **kwargs: Any) -> Embeddings:
        pass # Implementation omitted


class GoogleTranscriptionProvider(BaseTranscriptionProvider):
    """Gemini transcription (v1beta). This provider is working correctly."""

    async def transcribe_audio(
        self,
        api_key: str,
        audio_file_bytes: bytes,
        filename: str,
        language: Optional[str],
        model_name: str
    ) -> Optional[str]:
        if not api_key:
            raise ValueError("Google API key is required for transcription provider.")
        _set_proxy_env()
        client = genai.Client(api_key=api_key) # Removed http_options=None for default proxy behavior
        prompt = "Generate a transcript of the speech in a paragraph, avoid background-noise labels, and polish the text. but i emphasis that all the speech is in transcribe!"
        audio_part = Part.from_bytes(data=audio_file_bytes, mime_type="audio/mpeg")
        try:
            resp = await client.aio.models.generate_content(
                model=model_name,
                contents=[prompt, audio_part]
            )
            return resp.text
        except Exception as e:
            logger.exception(f"GoogleTranscriptionProvider error: {e}")
            return None


class GoogleTTSProvider(BaseTTSProvider):
    """Gemini Text-to-Speech, with WAV file formatting."""

    async def synthesize_speech(
        self,
        api_key: str,
        text: str,
        model_name: str,
        voice_name: str,
        **kwargs: Any
    ) -> Optional[bytes]:
        if not api_key:
            raise ValueError("Google API key is required for TTS provider.")
        _set_proxy_env()
        client = genai.Client(api_key=api_key)
        tts_cfg = GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=voice_name)
                ),\
            )
        )
        try:
            resp = await client.aio.models.generate_content(
                model=model_name,
                contents=text,
                config=tts_cfg
            )
            raw_audio_data = resp.candidates[0].content.parts[0].inline_data.data

            # CORRECTED: Format the raw PCM data into a valid WAV file in memory
            # The Gemini TTS API returns audio at a 24000 Hz sample rate.
            sample_rate = 24000
            
            # Use an in-memory buffer to build the WAV file
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(raw_audio_data)
            
            # Get the complete WAV file content from the buffer
            wav_bytes = buffer.getvalue()
            logger.info(f"Successfully formatted raw TTS data into a WAV file ({len(wav_bytes)} bytes).")
            return wav_bytes

        except Exception as e:
            logger.exception(f"GoogleTTSProvider error: {e}")
            return None