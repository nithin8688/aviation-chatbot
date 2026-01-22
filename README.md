# ✈️ Aviation Chatbot - Streamlit Web App

A production-ready RAG (Retrieval-Augmented Generation) chatbot for aviation documentation with dynamic PDF upload capability.

## 🎯 Features

- ✅ **Interactive Chat Interface** - Ask questions about aviation documents
- ✅ **PDF Upload** - Upload new PDFs and instantly query them
- ✅ **Source Citations** - See which documents answered your question
- ✅ **Document Management** - View and delete documents from the database
- ✅ **Real-time Statistics** - Track database size and document counts
- ✅ **Chat History** - Save and review past conversations

## 🏗️ Architecture

```
PostgreSQL (pgvector) → Embedding Search → Gemini LLM → Answer
         ↑
    PDF Upload → Chunking → Embeddings → Database
```

## 📋 Prerequisites

- Python 3.11+
- Docker Desktop (for PostgreSQL)
- Gemini API Key (free from Google AI Studio)

## 🚀 Quick Start

### 1. Ensure Docker Container is Running

```bash
# Check if container exists
docker ps -a

# Start the container
docker start aviation-postgres

# Verify it's running
docker ps
```

### 2. Update Configuration

Open `src/config.py` and ensure:

```python
GEMINI_API_KEY = "your-actual-api-key-here"
GEMINI_MODEL = "gemini-2.5-flash"
```

### 3. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Asking Questions

1. Type your question in the chat input at the bottom
2. Press Enter or click Send
3. View the answer with source citations
4. Expand "View Sources" to see which documents were used

**Example Questions:**
- "What is SCADA and how does it work?"
- "What are the responsibilities of airport apron control?"
- "Explain runway safety procedures"

### Uploading PDFs

1. Go to the sidebar → "📤 Upload New PDF"
2. Click "Choose a PDF file"
3. Select your aviation PDF document
4. Click "🚀 Process PDF"
5. Wait for processing to complete
6. Ask questions about the newly uploaded document!

### Managing Documents

1. Go to sidebar → "🗑️ Document Management"
2. Select a document from the dropdown
3. Click "Delete Document"
4. Click again to confirm

### Adjusting Settings

- **Number of sources**: Use the slider to retrieve 1-10 sources per query
- **Clear Chat**: Remove all messages from current session
- **Save Chat**: Export conversation to JSON file

## 📁 Project Structure

```
aviation-chatbot/
├── app.py                      # Streamlit application
├── src/
│   ├── config.py              # Configuration
│   ├── db_utils.py            # Database operations
│   ├── rag_engine.py          # RAG logic
│   └── ingest.py              # PDF processing
├── data/
│   ├── raw_pdfs/              # Initial PDFs
│   ├── uploaded_pdfs/         # User-uploaded PDFs
│   ├── query_history.json     # Query logs
│   └── chat_history.json      # Chat sessions
├── notebooks/                  # Development notebooks
└── requirements.txt           # Python dependencies
```

## ⚙️ Configuration Options

Edit `src/config.py` to customize:

```python
# Retrieval settings
TOP_K_RETRIEVAL = 5           # Number of chunks to retrieve
CHUNK_SIZE = 400              # Size of text chunks
CHUNK_OVERLAP = 100           # Overlap between chunks

# LLM settings
GEMINI_MODEL = "gemini-2.5-flash"  # Gemini model to use
LLM_TEMPERATURE = 0.1         # Lower = more deterministic

# Database settings
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "aviation_chatbot"
```

## 🐛 Troubleshooting

### Error: "Connection refused"

**Problem:** PostgreSQL container is not running

**Solution:**
```bash
docker start aviation-postgres
```

### Error: "API key invalid"

**Problem:** Gemini API key not configured

**Solution:**
1. Get API key from https://aistudio.google.com/app/apikey
2. Update `src/config.py` with your key

### Error: "Module not found"

**Problem:** Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

### Performance Issues

**Problem:** Slow response times

**Solution:**
- Reduce `TOP_K_RETRIEVAL` in settings slider
- Use `gemini-2.5-flash-lite` model for faster responses
- Check database has proper indexes (run `setup_database.ipynb`)

## 📊 Database Statistics

View real-time statistics in the sidebar:
- Total number of chunks in database
- Number of documents indexed
- Chunks per document

## 💾 Data Persistence

All data is stored in:
- **PostgreSQL**: Knowledge chunks and embeddings
- **JSON files**: Chat history and query logs
- **File system**: Uploaded PDF files

## 🔒 Security Notes

- API keys should be in environment variables (not hardcoded)
- Consider adding authentication for production deployment
- Implement rate limiting for API calls
- Validate uploaded files before processing

## 🎓 Next Steps (Phase 2)

After the app is working:
1. Refactor code into clean modules
2. Add unit tests
3. Implement evaluation metrics
4. Add batch processing for multiple PDFs
5. Deploy to cloud (Streamlit Cloud, Heroku, AWS)

## 📝 License

MIT License - Feel free to use and modify

## 🤝 Support

For issues:
1. Check troubleshooting section
2. Review notebook outputs
3. Verify Docker container status
4. Check API key configuration