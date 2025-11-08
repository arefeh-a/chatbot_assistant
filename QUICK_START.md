# 🚀 Quick Start Guide

This guide will help you get the **Generic RAG and Voice API** up and running in minutes.

---

## 📋 Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.9 or higher** installed
- ✅ **pip** (Python package manager)
- ✅ **API Keys** for at least one provider:
  - OpenAI API key (from <https://platform.openai.com/api-keys>)
  - OR Google AI API key (from <https://makersuite.google.com/app/apikey>)
- ✅ **Virtual environment tool** (built-in `venv` or `virtualenv`)
- ✅ **Git** (optional, for cloning)

---

## 🔧 Step-by-Step Setup

### Step 1: Navigate to Project Directory

```bash
cd /home/your/Projects/chatbot
```

If you're cloning the project:

```bash
git clone <your-repo-url>
cd chatbot
```

---

### Step 2: Create Virtual Environment

Create an isolated Python environment to avoid dependency conflicts:

```bash
# Using venv (recommended)
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# On Windows (if applicable):
# .venv\Scripts\activate
```

**Verify activation**: Your terminal prompt should now show `(.venv)`.

---

### Step 3: Install Dependencies

Install all required Python packages:

```bash
pip install --upgrade pip
pip install -r requirement.txt
```

**Installation time**: ~2-5 minutes depending on your connection.

**Key packages installed**:

- FastAPI & Uvicorn (web server)
- LangChain (LLM orchestration)
- FAISS (vector database)
- OpenAI & Google AI SDKs
- Pydantic, python-dotenv, tiktoken

---

### Step 4: Configure Environment Variables

Create your `.env` file with API keys and settings:

```bash
# Create .env file
touch .env
```

Edit `.env` with your favorite editor and add:

```bash
# ======================
# API Keys (Required)
# ======================
# Get OpenAI key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Get Google AI key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=AIza-your-google-api-key-here

# ======================
# Application Settings
# ======================
APP_HOST=0.0.0.0
APP_PORT=8008

# ======================
# Optional: Proxy Settings
# ======================
# Uncomment if behind a corporate firewall
# HTTPS_PROXY=http://your-proxy-server:port
# HTTP_PROXY=http://your-proxy-server:port
```

**Security Note**:

- ⚠️ Never commit `.env` to version control
- Add `.env` to your `.gitignore` file

---

### Step 5: Configure Providers

Edit `app/config.py` to select which AI provider to use for each service:

```python
# app/config.py (lines 28-31)

# Choose your providers (OPENAI or GOOGLE)
LLM_PROVIDER: Provider = Provider.OPENAI          # Chat model
EMBEDDING_PROVIDER: Provider = Provider.OPENAI    # Embeddings for RAG
TRANSCRIPTION_PROVIDER: Provider = Provider.GOOGLE # Speech-to-text
TTS_PROVIDER: Provider = Provider.GOOGLE          # Text-to-speech
```

**Provider Recommendations**:

- **For RAG (Chat + Embeddings)**: OpenAI (GPT-4o + text-embedding-3-large)
- **For Audio (Transcription + TTS)**: Google (Gemini 2.5 Flash)
- **Mix and match** based on your use case and budget

---

### Step 6: Build the Knowledge Base

The RAG system needs a vector database of your documents. Add your source documents and build the knowledge base:

#### 6.1. Add Source Documents

```bash
# Your .txt files should be in the data/ directory
# Example: Copy your documents
cp /path/to/your/documents/*.txt data/

# Or create a test document
cat > data/test_knowledge.txt << 'EOF'
Topic: Artificial Intelligence
AI is the simulation of human intelligence by machines.

Topic: Machine Learning
Machine learning is a subset of AI that enables systems to learn from data.

Topic: Neural Networks
Neural networks are computing systems inspired by biological neural networks.
EOF
```

#### 6.2. Run the Knowledge Base Builder

```bash
python -m app.scripts.build_kb
```

**Expected output**:

```bash
2025-10-05 16:40:15 - INFO - [build_kb] - Step 1: Loading documents from: '/home/your/Projects/chatbot/data'
2025-10-05 16:40:15 - INFO - [build_kb] -   - Processing file: '1.txt'
2025-10-05 16:40:15 - INFO - [build_kb] -   - Processing file: '2.txt'
...
2025-10-05 16:40:16 - INFO - [build_kb] - Loaded a total of 245 topic-block document(s).
2025-10-05 16:40:16 - INFO - [build_kb] - Step 2: Saving 245 documents to JSON...
2025-10-05 16:40:16 - INFO - [build_kb] - Step 3: Initializing embedding model from provider: 'openai'
2025-10-05 16:40:17 - INFO - [build_kb] - Step 4: Building FAISS vector store from 245 documents...
2025-10-05 16:40:25 - INFO - [build_kb] - FAISS vector store saved successfully to 'app/profile/vector_database'
```

**Build time**: ~30 seconds to several minutes depending on:

- Number of documents
- Embedding provider speed
- API rate limits

**Output files**:

- `app/profile/vector_database/kb_index.faiss` - FAISS index
- `app/profile/vector_database/kb_index.pkl` - FAISS metadata
- `app/profile/vector_database/chunked_documents.json` - Inspection file

---

### Step 7: Configure System Prompts (Optional)

Customize the chatbot's behavior by editing prompt files:

```bash
# Edit system prompt (defines bot personality and rules)
nano app/profile/system_prompt.txt

# Edit query rewriting prompt (for conversational context)
nano app/profile/query_rewriting_prompt.txt
```

**Example system prompt**:

```txt
You are a helpful AI assistant specializing in [your domain].
Always provide accurate, concise answers based on the provided context.
If you don't know something, say so honestly.
Use a professional yet friendly tone.
```

---

### Step 8: Start the Application

Launch the FastAPI server:

```bash
python -m app.main
```

**Expected output**:

```bash
2025-10-05 16:40:30 - INFO - [main] - Application startup: Assembling and initializing services...
2025-10-05 16:40:31 - INFO - [rag_service] - Loading FAISS vector store from: app/profile/vector_database
2025-10-05 16:40:32 - INFO - [main] - RAGService initialized and stored in app.state.
2025-10-05 16:40:32 - INFO - [main] - TranscriptionService initialized and stored in app.state.
2025-10-05 16:40:32 - INFO - [main] - TTSService initialized and stored in app.state.
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8008 (Press CTRL+C to quit)
```

**Access the application**:

- 🌐 **API Root**: <http://localhost:8008>
- 📚 **Interactive Docs**: <http://localhost:8008/docs>
- 📖 **ReDoc**: <http://localhost:8008/redoc>

---

## 🧪 Test Your Installation

### Test 1: Health Check

```bash
curl http://localhost:8008/api/v1/health
```

**Expected response**:

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

### Test 2: Chat API (RAG)

```bash
curl -X POST http://localhost:8008/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "history": []
  }'
```

**Expected**: Streaming response with AI-generated answer based on your knowledge base.

### Test 3: Transcription API

```bash
# Prepare a test audio file (e.g., test.mp3)
curl -X POST http://localhost:8008/api/v1/transcribe \
  -F "file=@test.mp3" \
  -F "language=en"
```

**Expected**:

```json
{
  "transcript": "This is the transcribed text from your audio file.",
  "error_message": null
}
```

### Test 4: Text-to-Speech API

```bash
curl -X POST http://localhost:8008/api/v1/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the text to speech system."}' \
  --output test_output.mp3
```

**Expected**: Audio file saved as `test_output.mp3`.

---

## 🌐 Using the Interactive API Documentation

FastAPI provides automatic, interactive API documentation:

1. **Open your browser** to <http://localhost:8008/docs>
2. **Explore endpoints** with full schemas and examples
3. **Test directly** in the browser using "Try it out" buttons
4. **View request/response** formats with automatic validation

**Features**:

- ✅ Interactive testing
- ✅ Schema visualization
- ✅ Authentication support (if added)
- ✅ Download OpenAPI spec (JSON/YAML)

---

## 🎯 Common Use Cases

### Use Case 1: Chat with Your Documents

```python
import requests

response = requests.post(
    "http://localhost:8008/api/v1/chat",
    json={
        "query": "Explain neural networks",
        "history": []
    },
    stream=True
)

for chunk in response.iter_content(decode_unicode=True):
    print(chunk, end="", flush=True)
```

### Use Case 2: Multi-Turn Conversation

```python
history = []
queries = ["What is AI?", "How is it different from ML?"]

for query in queries:
    response = requests.post(
        "http://localhost:8008/api/v1/chat",
        json={"query": query, "history": history}
    ).text
    
    # Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": response})
```

### Use Case 3: Voice Interface

```python
# 1. Transcribe user's audio
transcribe_response = requests.post(
    "http://localhost:8008/api/v1/transcribe",
    files={"file": open("user_audio.mp3", "rb")}
).json()

user_text = transcribe_response["transcript"]

# 2. Get chat response
chat_response = requests.post(
    "http://localhost:8008/api/v1/chat",
    json={"query": user_text, "history": []}
).text

# 3. Convert response to speech
tts_response = requests.post(
    "http://localhost:8008/api/v1/speech",
    json={"text": chat_response}
)

with open("assistant_response.mp3", "wb") as f:
    f.write(tts_response.content)
```

---

## 🔄 Updating the Knowledge Base

When you add or modify documents:

```bash
# 1. Add/update .txt files in data/
cp new_documents/*.txt data/

# 2. Rebuild the knowledge base
python -m app.scripts.build_kb

# 3. Restart the application
# (Press CTRL+C to stop, then run again)
python -m app.main
```

**Note**: If running in production with systemd:

```bash
sudo systemctl restart Chatbot
```

---

## 🐛 Troubleshooting

### Problem: "FAISS index not found"

**Cause**: Knowledge base not built yet.

**Solution**:

```bash
python -m app.scripts.build_kb
```

---

### Problem: "API key is required"

**Cause**: Missing or incorrect API keys in `.env`.

**Solution**:

1. Check `.env` file exists: `ls -la .env`
2. Verify keys are set: `cat .env`
3. Ensure no extra spaces around `=` in `.env`
4. Restart the application after editing `.env`

---

### Problem: Import or module errors

**Cause**: Dependencies not installed or wrong Python version.

**Solution**:

```bash
# Verify Python version (must be 3.9+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirement.txt
```

---

### Problem: Port already in use

**Cause**: Another process is using port 8008.

**Solution**:

```bash
# Option 1: Change port in .env
echo "APP_PORT=8009" >> .env

# Option 2: Kill the process using port 8008
# Find the process
lsof -i :8008

# Kill it (replace PID with actual process ID)
kill -9 <PID>
```

---

### Problem: Slow response times

**Possible causes and solutions**:

1. **Large knowledge base**: Reduce `RETRIEVER_K` in `app/config.py`
2. **API rate limits**: Use exponential backoff or upgrade plan
3. **Network issues**: Check proxy settings in `.env`
4. **Cold start**: First request after restart is slower (model loading)

---

## 🚀 Next Steps

### Production Deployment

For production use, see the **Production Deployment** section in [README.md](README.md):

- Set up systemd service
- Configure logging and monitoring
- Set up reverse proxy (nginx/caddy)
- Enable HTTPS/SSL
- Set up rate limiting and authentication

### Customization

Explore advanced features:

- **Add new providers**: Implement `providers/base.py` interfaces
- **Fine-tune prompts**: Edit files in `app/profile/`
- **Adjust RAG parameters**: Modify `app/config.py`
- **Add authentication**: Use FastAPI's security utilities
- **Implement caching**: Add Redis for response caching

### Integration

Integrate with your applications:

- **Web frontend**: Build React/Vue.js interface
- **Mobile apps**: Use REST API from iOS/Android
- **Slack/Discord bots**: Connect via webhooks
- **Voice assistants**: Integrate with Alexa/Google Home

---

## 📚 Additional Resources

- **Full Documentation**: [README.md](README.md)
- **FastAPI Docs**: <https://fastapi.tiangolo.com>
- **LangChain Docs**: <https://python.langchain.com>
- **OpenAI API**: <https://platform.openai.com/docs>
- **Google AI Studio**: <https://ai.google.dev>

---

## 💡 Tips & Best Practices

1. **Start small**: Test with a few documents before scaling up
2. **Version control**: Keep your prompts and config in Git
3. **Monitor costs**: Track API usage for both providers
4. **Test thoroughly**: Use the `/health` endpoint for monitoring
5. **Keep updated**: Regularly update dependencies for security patches

---

## ✅ Quick Checklist

Before considering your setup complete:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list` shows langchain, fastapi, etc.)
- [ ] `.env` file created with valid API keys
- [ ] Provider selection configured in `app/config.py`
- [ ] Knowledge base built successfully (FAISS index exists)
- [ ] Application starts without errors
- [ ] `/health` endpoint returns all services "healthy"
- [ ] At least one successful API call to each endpoint

---

## You're all set! 🎉

If you encounter any issues not covered here, check the logs in `logs/` directory or enable debug logging by setting `log_level="debug"` in `app/main.py`.

Happy building! 🚀
