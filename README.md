# AI Customer Support Assistant (RAG)

This project is a simple Retrieval-Augmented Generation (RAG) application.

## Features
- Upload PDF documents
- Ask questions based on document
- Uses embeddings + FAISS for retrieval
- Generates answers using LLM

## Tech Stack
- Python
- Streamlit
- Sentence Transformers
- FAISS
- OpenAI API

## How it works
1. Extract text from PDF
2. Split into chunks
3. Convert chunks into embeddings
4. Store in FAISS index
5. Retrieve relevant chunks based on query
6. Send context + query to LLM
7. Display answer

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py