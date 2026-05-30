RAG with LangChain

  A Retrieval-Augmented Generation (RAG) pipeline that answers questions about IRS Publication 15 using LangChain, Qdrant, and Ollama.

  Architecture

  1. Load — Parse PDF (IRS Pub 15) with PyPDFLoader
  2. Chunk — Split into 500-char chunks with 50-char overlap
  3. Embed — Generate embeddings with sentence-transformers/all-MiniLM-L6-v2
  4. Store — Index chunks in Qdrant vector database
  5. Retrieve — Find top-3 relevant chunks for a query
  6. Generate — Answer using Llama 3.2 via Ollama with a custom tax-advisor prompt

  - Python 3.11+
  - Qdrant (https://qdrant.tech/) running on localhost:6333
  - Ollama (https://ollama.ai/) with llama3.2 model pulled

  Setup

  python3 -m venv .venv
  source .venv/bin/activate
  pip install langchain langchain-community langchain-qdrant pypdf sentence-transformers qdrant-client

  Start Qdrant:

  docker run -p 6333:6333 qdrant/qdrant

  Pull the Ollama model:

  ollama pull llama3.2

  Usage

  source .venv/bin/activate
  python3 langchain_rag.py

  The script loads p15.pdf, builds the vector store, then runs sample tax questions through the RAG chain.

  Project Structure

  .
  ├── langchain_rag.py   # RAG pipeline
  ├── p15.pdf            # IRS Publication 15 (source document)
  └── README.md
