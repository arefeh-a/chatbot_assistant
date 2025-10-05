# app/api/v1/endpoints.py
import logging
from typing import AsyncIterator

from fastapi import (
    APIRouter, 
    HTTPException, 
    Request, 
    status, 
    Depends,
    UploadFile, 
    File,
    Form
)
from fastapi.responses import StreamingResponse, Response

from app.api.v1.schemas import ChatRequest, TranscribeResponse, SpeechRequest
from app.core.rag_service import RAGService
from app.core.transcription_service import TranscriptionService
from app.core.tts_service import TTSService
from app import config

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1", 
    tags=["Generic RAG and Voice API"], 
)

# --- Generic User-Facing Error Messages ---
ERROR_MSG_SERVICE_UNAVAILABLE = "A required service is temporarily unavailable. Please try again later."
ERROR_MSG_UNEXPECTED = "An unexpected server error occurred."
ERROR_MSG_TRANSCRIPTION_FAILED = "Failed to transcribe the provided audio."
ERROR_MSG_TTS_FAILED = "Failed to synthesize speech from the provided text."


# --- Dependency Functions to get Services ---
# These retrieve the initialized service instances that will be created and stored
# in `app.state` by the main application entrypoint (main.py).

def get_rag_service(request: Request) -> RAGService:
    rag_service = getattr(request.app.state, 'rag_service', None)
    if not isinstance(rag_service, RAGService):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, ERROR_MSG_SERVICE_UNAVAILABLE)
    return rag_service

def get_transcription_service(request: Request) -> TranscriptionService:
    transcription_service = getattr(request.app.state, 'transcription_service', None)
    if not isinstance(transcription_service, TranscriptionService):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, ERROR_MSG_SERVICE_UNAVAILABLE)
    return transcription_service

def get_tts_service(request: Request) -> TTSService:
    tts_service = getattr(request.app.state, 'tts_service', None)
    if not isinstance(tts_service, TTSService):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, ERROR_MSG_SERVICE_UNAVAILABLE)
    return tts_service


# --- API Endpoints ---

@router.post("/chat")
async def process_chat_request(
    chat_input: ChatRequest, 
    rag_service: RAGService = Depends(get_rag_service) 
):
    """Handles streaming chat responses."""
    history_dicts = [msg.model_dump() for msg in chat_input.history]
    
    async def stream_generator() -> AsyncIterator[str]:
        try:
            async for chunk in rag_service.get_response(chat_input.query, history_dicts):
                yield chunk
        except Exception:
            logger.exception("Error during RAG stream generation.")
            yield ERROR_MSG_UNEXPECTED

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_file(
    file: UploadFile = File(...),
    language: str = Form(None),
    transcription_service: TranscriptionService = Depends(get_transcription_service)
):
    """Handles audio transcription."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded audio file is empty.")

    try:
        provider = config.TRANSCRIPTION_PROVIDER
        if provider == config.Provider.OPENAI:
            api_key, model_name = config.OPENAI_API_KEY, config.OPENAI_TRANSCRIPTION_MODEL
        else:
            api_key, model_name = config.GOOGLE_API_KEY, config.GOOGLE_TRANSCRIPTION_MODEL

        transcript = await transcription_service.transcribe_audio(
            api_key=api_key, model_name=model_name, audio_file_bytes=audio_bytes,
            filename=file.filename, language=language
        )

        if transcript is None:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, ERROR_MSG_TRANSCRIPTION_FAILED)
        return TranscribeResponse(transcript=transcript)

    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, ERROR_MSG_UNEXPECTED)


@router.post("/speech")
async def synthesize_speech_from_text(
    speech_input: SpeechRequest,
    tts_service: TTSService = Depends(get_tts_service)
):
    """Handles text-to-speech synthesis."""
    try:
        provider = config.TTS_PROVIDER
        if provider == config.Provider.OPENAI:
            api_key, model_name, voice = config.OPENAI_API_KEY, config.OPENAI_TTS_MODEL, config.OPENAI_TTS_VOICE
        else:
            api_key, model_name, voice = config.GOOGLE_API_KEY, config.GOOGLE_TTS_MODEL, config.GOOGLE_TTS_VOICE

        audio_bytes = await tts_service.synthesize_speech(
            api_key=api_key, model_name=model_name, voice_name=voice,
            text=speech_input.text, instructions=speech_input.instructions
        )

        if audio_bytes is None:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, ERROR_MSG_TTS_FAILED)
        
        # Both providers are now configured to return MP3 data.
        return Response(content=audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        logger.exception(f"Unexpected error during speech synthesis: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, ERROR_MSG_UNEXPECTED)


@router.get("/health", summary="Health check for the API and its services")
async def health_check(request: Request):
    """Checks the status of the API and its core injected services."""
    services_status = {}
    is_healthy = True

    # Check RAG Service
    rag_service_instance = getattr(request.app.state, 'rag_service', None)
    if rag_service_instance and all([rag_service_instance.llm, rag_service_instance.embedding_model]):
        services_status["rag_service"] = "healthy"
    else:
        services_status["rag_service"] = "unavailable"
        is_healthy = False
            
    # Check audio services
    for service_name in ['transcription_service', 'tts_service']:
        service_instance = getattr(request.app.state, service_name, None)
        if service_instance and hasattr(service_instance, 'provider'):
            services_status[service_name] = "healthy"
        else:
            services_status[service_name] = "unavailable"
            is_healthy = False
            
    if is_healthy:
        return {"status": "ok", "services": services_status}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "degraded", "services": services_status}
        )