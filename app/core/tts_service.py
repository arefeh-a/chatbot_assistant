# app/core/tts_service.py
import logging
import asyncio
from typing import Optional, Any
from pathlib import Path
import sys

# --- Path Correction for Direct Execution ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# --- Core Application & Provider Imports ---
from app import config
from providers.base import BaseTTSProvider

logger = logging.getLogger(__name__)


class TTSService:
    """A service to handle text-to-speech via an injected provider."""

    def __init__(self, provider: BaseTTSProvider):
        """
        Initializes the TTSService by injecting a provider dependency.
        """
        logger.info(f"Initializing TTSService with provider: {provider.__class__.__name__}")
        self.provider = provider
        # The service is now "dumb" about the main config settings.

    async def synthesize_speech(
        self,
        api_key: str,
        model_name: str,
        voice_name: str,
        text: str,
        **kwargs: Any
    ) -> Optional[bytes]:
        """
        Synthesizes audio by calling the injected provider's method.
        It uses the parameters passed directly to it, without knowledge of config.
        """
        # Call the method on the injected provider instance with the provided details.
        return await self.provider.synthesize_speech(
            api_key=api_key,
            text=text,
            model_name=model_name,
            voice_name=voice_name,
            **kwargs  # Pass along any extra arguments like 'instructions'
        )


if __name__ == '__main__':
    # This block simulates how app/main.py will assemble and inject dependencies.
    from providers.openai_provider import OpenAITTSProvider
    from providers.google_provider import GoogleTTSProvider

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    async def run_test():
        print("--- TTSService Standalone Test (Pure Dependency Injection) ---")
        try:
            # --- 1. Assembly Logic (This will live in `app/main.py`) ---
            provider_choice = config.TTS_PROVIDER
            print(f"Provider selected in config: {provider_choice.value}")

            if provider_choice == config.Provider.OPENAI:
                provider_instance = OpenAITTSProvider()
                api_key_for_test = config.OPENAI_API_KEY
                model_name_for_test = config.OPENAI_TTS_MODEL
                voice_name_for_test = config.OPENAI_TTS_VOICE
            else:
                provider_instance = GoogleTTSProvider()
                api_key_for_test = config.GOOGLE_API_KEY
                model_name_for_test = config.GOOGLE_TTS_MODEL
                voice_name_for_test = config.GOOGLE_TTS_VOICE
            
            # --- 2. Inject the provider instance into the TTSService ---
            tts_service = TTSService(provider=provider_instance)
            
            # --- 3. Run Test ---
            test_text = "سلام، من سینابات هستم. این یک آزمایش برای تولید گفتار است."
            test_output_path = PROJECT_ROOT_DIR / "tts_test_output.wav"
            
            print(f"Synthesizing speech for text: '{test_text}'")
            # The calling code is now responsible for passing the specific config values.
            audio_bytes = await tts_service.synthesize_speech(
                api_key=api_key_for_test,
                model_name=model_name_for_test,
                voice_name=voice_name_for_test,
                text=test_text,
                instructions="Speak clearly and naturally and friendly. try to be a persian native speaker "
            )

            if audio_bytes:
                with open(test_output_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"\n--- SUCCESS ---")
                print(f"Audio content successfully saved to '{test_output_path}'")
                print("----------------\n")
            else:
                logger.error("Test TTS failed and returned None.")

        except Exception as e:
            logging.exception(f"Test failed: {e}")

    asyncio.run(run_test())