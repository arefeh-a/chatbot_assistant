# app/config.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Root Definition ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Load .env File ---
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    logger.warning(f".env file not found at {env_path}. Key settings may be missing.")

# --- Provider Selection Enum ---
class Provider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"

# --- MASTER SWITCHES: CHOOSE YOUR PROVIDERS HERE ---
LLM_PROVIDER: Provider = Provider.OPENAI
EMBEDDING_PROVIDER: Provider = Provider.OPENAI
TRANSCRIPTION_PROVIDER: Provider = Provider.GOOGLE
TTS_PROVIDER: Provider = Provider.GOOGLE

# --- API Keys (Loaded from environment) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Smart Validation for API Keys ---
# Checks if a provider is used anywhere and logs an error if its key is missing.
USED_PROVIDERS = {LLM_PROVIDER, EMBEDDING_PROVIDER, TRANSCRIPTION_PROVIDER, TTS_PROVIDER}
if Provider.OPENAI in USED_PROVIDERS and not OPENAI_API_KEY:
    logger.critical("OpenAI is selected, but OPENAI_API_KEY is not set.")
if Provider.GOOGLE in USED_PROVIDERS and not GOOGLE_API_KEY:
    logger.critical("Google is selected, but GOOGLE_API_KEY is not set.")

# --- Model & Service Configurations ---


# RAG - LLM & Query Rewriting
OPENAI_LLM_CHAT_MODEL = "gpt-4o"
OPENAI_QUERY_REWRITE_MODEL = "gpt-4o"
GOOGLE_LLM_CHAT_MODEL = "gemini-2.5-flash-preview-05-20"
GOOGLE_QUERY_REWRITE_MODEL = "gemini-2.5-flash-preview-05-20"

LLM_TEMPERATURE: float = 0.0
QUERY_REWRITE_TEMPERATURE: float = 0.0

# RAG - Embeddings
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-exp"

# RAG - General Parameters
RETRIEVER_K: int = 3
MAX_HISTORY_TURNS: int = 3
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 0
DOC_BATCH_SIZE = 100

# Audio - Transcription
OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
GOOGLE_TRANSCRIPTION_MODEL = "gemini-2.5-flash-preview-05-20"

# Audio - Text-to-Speech
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "nova"
GOOGLE_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GOOGLE_TTS_VOICE = "Sulafat"
TTS_RESPONSE_FORMAT_DEFAULT: str = "mp3"

# --- Path Configurations ---
# Note: These paths are relative to the PROJECT_ROOT.
SOURCE_DOCS_DIR = PROJECT_ROOT / "data"
APP_PROFILE_DIR = PROJECT_ROOT / "app" / "profile"
QUERY_REWRITING_FILE_PATH = APP_PROFILE_DIR / "query_rewriting_prompt.txt"
SYSTEM_PROMPT_FILE_PATH = APP_PROFILE_DIR / "system_prompt.txt"
VECTOR_DB_DIR = APP_PROFILE_DIR / "vector_database"
FAISS_INDEX_NAME = "kb_index"
CHUNKED_DOCS_JSON_FILENAME = "chunked_documents.json"
CHUNKED_DOCS_JSON_PATH = VECTOR_DB_DIR / CHUNKED_DOCS_JSON_FILENAME

# --- Application Host and Port ---
APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT: int = int(os.getenv("APP_PORT", "8008"))