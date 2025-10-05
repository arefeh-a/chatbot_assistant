# app/main.py
import logging
from contextlib import asynccontextmanager
import uvicorn
import sys
from pathlib import Path

from fastapi import FastAPI

# --- Path Correction ---
# Ensures that the 'providers' directory can be imported
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# --- Core Application & Provider Imports ---
from app import config
from app.core.rag_service import RAGService
from app.core.transcription_service import TranscriptionService
from app.core.tts_service import TTSService
from app.api.v1 import endpoints as api_v1_endpoints

# --- Provider Class Imports ---
# Import all possible provider classes. The lifespan manager will choose which ones to use.
from providers.openai_provider import (
    OpenAILLMProvider, OpenAIEmbeddingProvider, OpenAITranscriptionProvider, OpenAITTSProvider
)
from providers.google_provider import (
    GoogleLLMProvider, GoogleEmbeddingProvider, GoogleTranscriptionProvider, GoogleTTSProvider
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(module)s] - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup logic. This is the central "assembly line"
    where all services are created and injected.
    """
    logger.info("Application startup: Assembling and initializing services...")
    
    # --- Assemble RAG Service Dependencies ---
    try:
        if config.LLM_PROVIDER == config.Provider.OPENAI:
            llm_provider = OpenAILLMProvider()
            llm_api_key = config.OPENAI_API_KEY
            chat_model_name = config.OPENAI_LLM_CHAT_MODEL
            rewrite_model_name = config.OPENAI_QUERY_REWRITE_MODEL
        else: # Google
            llm_provider = GoogleLLMProvider()
            llm_api_key = config.GOOGLE_API_KEY
            chat_model_name = config.GOOGLE_LLM_CHAT_MODEL
            rewrite_model_name = config.GOOGLE_QUERY_REWRITE_MODEL

        if config.EMBEDDING_PROVIDER == config.Provider.OPENAI:
            embedding_provider = OpenAIEmbeddingProvider()
            embedding_api_key = config.OPENAI_API_KEY
            embedding_model_name = config.OPENAI_EMBEDDING_MODEL
        else: # Google
            embedding_provider = GoogleEmbeddingProvider()
            embedding_api_key = config.GOOGLE_API_KEY
            embedding_model_name = config.GOOGLE_EMBEDDING_MODEL

        # Create the final model instances to be injected
        main_llm_instance = llm_provider.get_chat_model(llm_api_key, chat_model_name, config.LLM_TEMPERATURE)
        rewrite_llm_instance = llm_provider.get_chat_model(llm_api_key, rewrite_model_name, config.QUERY_REWRITE_TEMPERATURE)
        embedding_instance = embedding_provider.get_embedding_model(embedding_api_key, embedding_model_name)

        # Inject components into RAGService
        app.state.rag_service = RAGService(
            llm=main_llm_instance,
            llm_rewrite=rewrite_llm_instance,
            embedding_model=embedding_instance
        )
        logger.info("RAGService initialized and stored in app.state.")

    except Exception as e:
        logger.critical(f"CRITICAL ERROR during RAGService initialization: {e}", exc_info=True)
        app.state.rag_service = None

    # --- Assemble Transcription Service ---
    try:
        if config.TRANSCRIPTION_PROVIDER == config.Provider.OPENAI:
            transcription_provider = OpenAITranscriptionProvider()
        else: # Google
            transcription_provider = GoogleTranscriptionProvider()
        
        # Inject provider into TranscriptionService
        app.state.transcription_service = TranscriptionService(provider=transcription_provider)
        logger.info("TranscriptionService initialized and stored in app.state.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during TranscriptionService initialization: {e}", exc_info=True)
        app.state.transcription_service = None

    # --- Assemble TTS Service ---
    try:
        if config.TTS_PROVIDER == config.Provider.OPENAI:
            tts_provider = OpenAITTSProvider()
        else: # Google
            tts_provider = GoogleTTSProvider()
            
        # Inject provider into TTSService
        app.state.tts_service = TTSService(provider=tts_provider)
        logger.info("TTSService initialized and stored in app.state.")
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during TTSService initialization: {e}", exc_info=True)
        app.state.tts_service = None
    
    yield
    
    logger.info("Application shutdown.")


# --- FastAPI App Instance ---
# Create the main FastAPI application with a generic title and description.
app = FastAPI(
    title="Generic RAG and Voice API",
    description="A flexible API for Retrieval-Augmented Generation, Transcription, and Text-to-Speech, supporting multiple AI providers.",
    version="2.0.0",
    lifespan=lifespan 
)

# Mount the API router
app.include_router(api_v1_endpoints.router, prefix="/api")

# --- Root Endpoint ---
@app.get("/", tags=["Root"], summary="API Root")
async def read_root():
    """Provides basic information and links to the API documentation."""
    return {
        "message": "Welcome to the Generic RAG and Voice API!",
        "docs": "/docs",
        "api_version": "/api/v1",
        "health_check": "/api/v1/health"
    }

# --- Uvicorn Runner ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on http://{config.APP_HOST}:{config.APP_PORT}")
    uvicorn.run(
        "app.main:app", 
        host=config.APP_HOST,
        port=config.APP_PORT,
        log_level="info", 
        reload=True
    )
