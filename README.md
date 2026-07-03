# RAG Document Intelligence System

A production-ready Retrieval-Augmented Generation (RAG) system that lets you query documents using natural language. Built with LangChain, OpenAI, FAISS, and FastAPI.

## How it works

1. Documents are loaded, split into chunks, and embedded using OpenAI's embedding model
2. Embeddings are stored in a FAISS vector index on disk
3. When a question is asked, the most relevant chunks are retrieved via semantic search
4. Retrieved chunks are sent to GPT-4o-mini as context to generate a grounded answer
5. Responses are cached to avoid redundant API calls

## Tech Stack

- **LangChain** — document loading, text splitting, retrieval pipeline
- **OpenAI** — embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
- **FAISS** — vector similarity search
- **FastAPI** — REST API with auto-generated docs
- **Python 3.11**

## Setup

```bash
# Create and activate environment
conda create -n rag-doc-intel python=3.11
conda activate rag-doc-intel
pip install -r requirements.txt

# Add your OpenAI key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# Ingest documents
python app/services/ingestion.py

# Start the server
uvicorn app.main:app --reload
```

## API

- `POST /query` — ask a question, get an answer with token usage
- Swagger docs at `http://localhost:8000/docs`