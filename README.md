# Generic RAG and Voice API

A flexible, production-ready FastAPI application for **Retrieval-Augmented Generation (RAG)**, **Speech-to-Text (Transcription)**, and **Text-to-Speech (TTS)** services. The system supports **multiple AI providers** (OpenAI and Google Gemini) with a clean, provider-agnostic architecture.

## 🌟 Features

### Core Capabilities

- **🤖 RAG Chat System**: Conversational AI with context retrieval from a custom knowledge base
- **🎤 Audio Transcription**: Convert speech to text using state-of-the-art models
- **🔊 Text-to-Speech**: Generate natural-sounding audio from text
- **📚 Knowledge Base Management**: Build and maintain vector databases from text documents

### Architecture Highlights

- **Provider-Agnostic Design**: Switch between OpenAI and Google AI with configuration changes only
- **Dependency Injection**: Clean service architecture with pure DI pattern
- **Streaming Responses**: Real-time chat responses for better UX
- **FAISS Vector Store**: Fast, efficient semantic search for RAG
- **Production Ready**: Includes systemd service file and comprehensive logging

### Supported Providers

- **OpenAI**: GPT-4o, GPT-4o-mini, Embeddings, Whisper, TTS
- **Google AI**: Gemini 2.5 Flash, Gemini Embeddings, Audio transcription/synthesis

---

## 📁 Project Structure

```tree
chatbot/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints.py       # API route handlers
│   │       └── schemas.py         # Request/response models
│   ├── core/
│   │   ├── rag_service.py        # RAG engine with DI
│   │   ├── transcription_service.py
│   │   └── tts_service.py
│   ├── profile/
│   │   ├── system_prompt.txt     # System instructions for chatbot
│   │   ├── query_rewriting_prompt.txt
│   │   └── vector_database/      # FAISS index storage
│   ├── scripts/
│   │   ├── build_kb.py           # Knowledge base builder
│   │   └── lib/
│   │       └── data_loader.py    # Custom document loader
│   ├── config.py                 # Centralized configuration
│   └── main.py                   # Application entry point
├── providers/
│   ├── base.py                   # Abstract provider interfaces
│   ├── openai_provider.py        # OpenAI implementations
│   └── google_provider.py        # Google AI implementations
├── data/                         # Source documents (*.txt)
├── logs/                         # Application logs
├── requirement.txt               # Python dependencies
├── .env                          # Environment variables (not in repo)
└── sinabot.service              # systemd service configuration
```

---

## 🚀 Quick Start

See **[QUICK_START.md](QUICK_START.md)** for detailed setup instructions.

### Prerequisites

- Python 3.9+
- API keys for OpenAI and/or Google AI
- Virtual environment tool (venv or virtualenv)

### Installation Overview

```bash
# 1. Clone and navigate to project
cd /home/erfan/Projects/chatbot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Configure environment variables
cp .env.example .env  # Edit with your API keys

# 5. Build knowledge base
python -m app.scripts.build_kb

# 6. Run the application
python -m app.main
```

---

## 🔧 Configuration

### Provider Selection (`app/config.py`)

Configure which AI provider to use for each service:

```python
# Master switches - choose providers
LLM_PROVIDER: Provider = Provider.OPENAI          # or Provider.GOOGLE
EMBEDDING_PROVIDER: Provider = Provider.OPENAI    
TRANSCRIPTION_PROVIDER: Provider = Provider.GOOGLE
TTS_PROVIDER: Provider = Provider.GOOGLE
```

### Environment Variables (`.env`)

```bash
# API Keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=8008

# Optional: Proxy settings for Google AI
HTTPS_PROXY=http://your-proxy:port
HTTP_PROXY=http://your-proxy:port
```

### Model Configuration

Fine-tune model parameters in `app/config.py`:

```python
# OpenAI Models
OPENAI_LLM_CHAT_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"

# Google Models
GOOGLE_LLM_CHAT_MODEL = "gemini-2.5-flash-preview-05-20"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-exp"
GOOGLE_TRANSCRIPTION_MODEL = "gemini-2.5-flash-preview-05-20"
GOOGLE_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# RAG Parameters
RETRIEVER_K: int = 3              # Number of documents to retrieve
MAX_HISTORY_TURNS: int = 3        # Conversation history length
CHUNK_SIZE: int = 800             # Document chunk size
CHUNK_OVERLAP: int = 0            # Overlap between chunks
```

---

## 📡 API Endpoints

### Base URL: `http://localhost:8008`

### 1. **Chat** (Streaming RAG)

```bash
POST /api/v1/chat
Content-Type: application/json

{
  "query": "What is retrieval-augmented generation?",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Response**: Server-Sent Events (SSE) stream

### 2. **Transcribe Audio**

```bash
POST /api/v1/transcribe
Content-Type: multipart/form-data

file: <audio_file.mp3>
language: en (optional)
```

**Response**:

```json
{
  "transcript": "This is the transcribed text...",
  "error_message": null
}
```

### 3. **Synthesize Speech**

```bash
POST /api/v1/speech
Content-Type: application/json

{
  "text": "Hello, this is a test.",
  "instructions": "Speak in a calm voice" (optional)
}
```

**Response**: Audio file (MP3 or WAV)

### 4. **Health Check**

```bash
GET /api/v1/health
```

**Response**:

```json
{
  "status": "ok",
  "services": {
    "rag_service": "healthy",
    "transcription_service": "healthy",
    "tts_service": "healthy"
  }
}
```

### 5. **API Documentation**

- **Interactive Docs**: `http://localhost:8008/docs`
- **ReDoc**: `http://localhost:8008/redoc`

---

## 🏗️ Building the Knowledge Base

The system uses FAISS for semantic search. Build the vector database from your documents:

```bash
# Add .txt files to the data/ directory
cp your_documents/*.txt data/

# Run the knowledge base builder
python -m app.scripts.build_kb
```

### Knowledge Base Builder Features

- Loads all `.txt` files from `data/` directory
- Chunks documents intelligently
- Generates embeddings using configured provider
- Saves FAISS index to `app/profile/vector_database/`
- Exports chunked documents to JSON for inspection

---

## 🧪 Testing

### Test RAG Service

```bash
# Direct test of RAG service
python -m app.core.rag_service
```

### Test via API

```bash
# Start server
python -m app.main

# In another terminal, test with curl
curl -X POST http://localhost:8008/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Test question", "history": []}'
```

---

## 🚀 Production Deployment

### Using systemd

1. Edit `sinabot.service` with your paths:

    ```ini
    [Service]
    User=your_username
    WorkingDirectory=/path/to/chatbot/
    EnvironmentFile=/path/to/chatbot/.env
    ExecStart=/path/to/chatbot/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8008
    ```

2. Install and start the service:

    ```bash
    sudo cp sinabot.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable sinabot
    sudo systemctl start sinabot
    ```

3. Check status:

    ```bash
    sudo systemctl status sinabot
    sudo journalctl -u sinabot -f  # View logs
    ```

---

## 🔍 Architecture Details

### Dependency Injection Pattern

The application uses pure dependency injection for flexibility:

```python
# main.py - Assembly line
if config.LLM_PROVIDER == config.Provider.OPENAI:
    llm_provider = OpenAILLMProvider()
    llm_instance = llm_provider.get_chat_model(api_key, model_name, temp)
else:
    llm_provider = GoogleLLMProvider()
    llm_instance = llm_provider.get_chat_model(api_key, model_name, temp)

# Inject into service
app.state.rag_service = RAGService(llm=llm_instance, ...)
```

### Provider System

All providers implement abstract base classes:

- `BaseLLMProvider` - Language model interface
- `BaseEmbeddingProvider` - Embedding model interface
- `BaseTranscriptionProvider` - Speech-to-text interface
- `BaseTTSProvider` - Text-to-speech interface

This allows adding new providers (e.g., Anthropic, Cohere) by implementing these interfaces.

---

## 📊 Logging

Logs are configured with structured output:

```python
# Console logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - [%(module)s] - %(message)s'
)

# File logging (via systemd)
StandardOutput=append:/path/to/logs/app.log
StandardError=append:/path/to/logs/error.log
```

---

## 🛠️ Troubleshooting

### Common Issues

#### **1. FAISS index not found**

```error
FileNotFoundError: FAISS index not found
```

**Solution**: Run `python -m app.scripts.build_kb` to build the knowledge base.

#### **2. API key errors**

```error
CRITICAL: OpenAI is selected, but OPENAI_API_KEY is not set
```

**Solution**: Ensure `.env` file contains valid API keys.

#### **3. Import errors**

```error
ModuleNotFoundError: No module named 'langchain'
```

**Solution**: Install dependencies: `pip install -r requirement.txt`

#### **4. Google AI proxy issues**

```error
Connection error during TTS
```

**Solution**: Set `HTTPS_PROXY` in `.env` if behind a corporate firewall.

---

## 📦 Dependencies

### Core Libraries

- **FastAPI**: Web framework
- **LangChain**: LLM orchestration
- **FAISS**: Vector similarity search
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

### AI Providers

- **openai**: OpenAI API client
- **google-genai**: Google AI SDK
- **langchain-openai**: LangChain OpenAI integration
- **langchain-google-genai**: LangChain Google integration

### Utilities

- **python-dotenv**: Environment variable management
- **tiktoken**: Token counting for OpenAI models

---

## 🤝 Contributing

### Adding a New Provider

1. Create a new provider file: `providers/newprovider_provider.py`
2. Implement all four base classes from `providers/base.py`
3. Add provider enum to `app/config.py`
4. Update imports in `app/main.py`
5. Test thoroughly with knowledge base and API endpoints

### Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Add docstrings to all classes and methods
- Keep functions focused and single-purpose

---

## 📝 License

[Add your license information here]

---

## 👥 Contact

[Add contact information or project maintainer details]

---

## 🙏 Acknowledgments

- **LangChain** for the excellent LLM orchestration framework
- **FAISS** for efficient similarity search
- **OpenAI** and **Google** for their powerful AI models

---

## Happy Coding! 🚀
