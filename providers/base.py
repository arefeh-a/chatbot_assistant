from abc import ABC, abstractmethod
from typing import Optional, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class BaseLLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self, api_key: str, model_name: str, temperature: float, **kwargs: Any) -> BaseChatModel:
        pass

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding_model(self, api_key: str, model_name: str, **kwargs: Any) -> Embeddings:
        pass

class BaseTranscriptionProvider(ABC):
    @abstractmethod
    async def transcribe_audio(
        self, api_key: str, audio_file_bytes: bytes,
        filename: str, language: Optional[str], model_name: str
    ) -> Optional[str]:
        pass

class BaseTTSProvider(ABC):
    @abstractmethod
    async def synthesize_speech(
        self, api_key: str, text: str,
        model_name: str, voice_name: str, **kwargs: Any
    ) -> Optional[bytes]:
        pass
