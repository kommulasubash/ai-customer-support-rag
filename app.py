# app.py

# This file handles:
# - User interface
# - File upload
# - Question input
# - Displaying answers

import streamlit as st
from rag_pipeline import load_pdf, chunk_text, create_embeddings, create_faiss_index, retrieve
from llm import generate_answer

st.title("📄 AI Customer Support Assistant (RAG)")

# Upload PDF
uploaded_file = st.file_uploader("Upload company document (PDF)", type="pdf")

if uploaded_file:
    st.success("Document uploaded successfully!")

    # Store processing state to avoid recalculating embeddings on every Streamlit rerun
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        with st.spinner("Processing document... this might take a moment."):
            # Step 1: Load text
            text = load_pdf(uploaded_file)

            # Step 2: Chunk text
            chunks = chunk_text(text)

            # Step 3: Create embeddings
            embeddings = create_embeddings(chunks)

            # Step 4: Create FAISS index
            index = create_faiss_index(embeddings)
            
            # Save into session state
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.last_uploaded_file = uploaded_file.name

    # User input
    query = st.text_input("Ask a question:")

    if query:
        # Step 5: Retrieve relevant chunks mapping to our session_state elements
        results = retrieve(query, st.session_state.chunks, st.session_state.index)

        context = " ".join(results)

        # Step 6: Generate answer
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, context)

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Top Retrieved Context:")
        for r in results:
            st.write("-", r[:200], "...")