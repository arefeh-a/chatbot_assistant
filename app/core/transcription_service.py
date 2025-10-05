# app/core/transcription_service.py
import logging
import asyncio
from typing import Optional
from pathlib import Path
import sys

# --- Path Correction for Direct Execution ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# --- Core Application & Provider Imports ---
from app import config
from providers.base import BaseTranscriptionProvider

logger = logging.getLogger(__name__)


class TranscriptionService:
    """A service to handle audio transcription via an injected provider."""

    def __init__(self, provider: BaseTranscriptionProvider):
        """
        Initializes the TranscriptionService by injecting a provider dependency.
        """
        logger.info(f"Initializing TranscriptionService with provider: {provider.__class__.__name__}")
        self.provider = provider

    async def transcribe_audio(
        self,
        api_key: str,
        model_name: str,
        audio_file_bytes: bytes,
        filename: str,
        language: Optional[str] = None
    ) -> Optional[str]:
        """
        Transcribes audio by calling the injected provider's method.
        """
        # Call the method on the injected provider instance with the provided details.
        return await self.provider.transcribe_audio(
            api_key=api_key,
            audio_file_bytes=audio_file_bytes,
            filename=filename,
            language=language,
            model_name=model_name
        )


if __name__ == '__main__':
    # This block simulates how app/main.py will assemble and inject dependencies.
    from providers.openai_provider import OpenAITranscriptionProvider
    from providers.google_provider import GoogleTranscriptionProvider

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    async def run_test():
        print("--- TranscriptionService Standalone Test (Dependency Injection) ---")
        try:
            # --- 1. Assembly Logic (This will live in `app/main.py`) ---
            provider_choice = config.TRANSCRIPTION_PROVIDER
            print(f"Provider selected in config: {provider_choice.value}")

            if provider_choice == config.Provider.OPENAI:
                provider_instance = OpenAITranscriptionProvider()
                api_key_for_test = config.OPENAI_API_KEY
                model_name_for_test = config.OPENAI_TRANSCRIPTION_MODEL
            else:
                provider_instance = GoogleTranscriptionProvider()
                api_key_for_test = config.GOOGLE_API_KEY
                model_name_for_test = config.GOOGLE_TRANSCRIPTION_MODEL
            
            # --- 2. Inject the provider instance into the TranscriptionService ---
            transcription_service = TranscriptionService(provider=provider_instance)
            
            # --- 3. Run Test ---
            test_audio_path = PROJECT_ROOT_DIR / "forTest.mp3"
            if not test_audio_path.exists():
                logger.error(f"Test audio file not found: {test_audio_path}")
                return

            with open(test_audio_path, "rb") as f:
                audio_bytes = f.read()

            print(f"Transcribing '{test_audio_path.name}'...")
            
            # The calling code is now responsible for passing the specific config values.
            # CORRECTED: Use the two-letter ISO-639-1 code for OpenAI
            language_hint = "en" if provider_choice == config.Provider.OPENAI else "en-US"
            
            transcript = await transcription_service.transcribe_audio(
                api_key=api_key_for_test,
                model_name=model_name_for_test,
                audio_file_bytes=audio_bytes,
                filename=test_audio_path.name,
                language=language_hint
            )

            if transcript is not None:
                print("\n--- TRANSCRIPTION RESULT ---")
                print(transcript)
                print("--------------------------\n")
            else:
                logger.error("Test transcription failed and returned None.")

        except Exception as e:
            logging.exception(f"Test failed: {e}")

    asyncio.run(run_test())
