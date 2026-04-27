# rag_pipeline.py

# This file handles:
# - Loading documents
# - Splitting into chunks
# - Creating embeddings
# - Storing & retrieving using FAISS

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader

# Load embedding model (converts text -> vector)
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(file):
    """Extract text from uploaded PDF"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def create_embeddings(chunks):
    """Convert text chunks into vectors"""
    return model.encode(chunks)


def create_faiss_index(embeddings):
    """Store embeddings in FAISS index"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype(np.float32))
    return index


def retrieve(query, chunks, index, k=3):
    """Find top-k similar chunks for a query"""
    query_vector = model.encode([query]).astype(np.float32)
    # k shouldn't exceed the number of chunks we have (prevents some out of bounds)
    distances, indices = index.search(query_vector, min(k, len(chunks)))
    return [chunks[i] for i in indices[0] if i != -1]