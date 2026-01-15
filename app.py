import streamlit as st
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

# Configure Google Gemini
genai.configure(api_key=api_key)

# Page Config
st.set_page_config(page_title="Pure RAG Chatbot (No LangChain)", layout="wide")

# --- Helper Functions (අපිම ලියන Logic එක) ---

def get_pdf_text(pdf_docs):
    """PDF එක කියවලා Text ගන්න Function එක"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_chunks(text, chunk_size=1000, overlap=200):
    """LangChain නැතුව Python වලින්ම Text එක කඩන Function එක"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap # Overlap logic
    return chunks

def get_embeddings(text_chunks):
    """Google Gemini Embedding Model එක කෙලින්ම Call කිරීම"""
    embeddings = []
    # Batch processing to handle API limits (optional logic handled simply here)
    for chunk in text_chunks:
        # 'models/embedding-001' තමයි Google Free embedding model එක
        result = genai.embed_content(
            model="models/embedding-001",
            content=chunk,
            task_type="retrieval_document",
            title="Embedding of PDF chunk"
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings, dtype='float32')

def create_vector_store(text_chunks, embeddings):
    """FAISS index එක හදන එක (Database Creation)"""
    dimension = embeddings.shape[1] # 768 dimensions for Gemini
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def get_answer(query, index, text_chunks):
    """ප්‍රශ්නයට අදාල කොටස් හොයලා උත්තරේ දෙන එක"""
    
    # 1. Query එක Embed කරනවා
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_vec = np.array([query_embedding['embedding']], dtype='float32')
    
    # 2. FAISS එකෙන් සමානම කොටස් 3ක් හොයනවා
    D, I = index.search(query_vec, k=3) # Top 3 results
    
    relevant_context = ""
    for idx in I[0]:
        if idx < len(text_chunks):
            relevant_context += text_chunks[idx] + "\n\n"
            
    # 3. Gemini Pro Model එකට Prompt එක යවනවා
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't know based on this document."
    
    Context:
    {relevant_context}
    
    Question: {query}
    """
    
    response = model.generate_content(prompt)
    return response.text, relevant_context

# --- UI Application ---

def main():
    st.title("🚀 Pure RAG Chatbot (No LangChain)")
    st.caption("Powered by Google Gemini & FAISS | Zero Dependency Hell")

    # Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None # Holds (index, chunks)

    # Sidebar
    with st.sidebar:
        st.header("📂 Documents")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing without LangChain..."):
                    # 1. Get Text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Chunking
                    chunks = get_chunks(raw_text)
                    st.write(f"Created {len(chunks)} text chunks.")
                    
                    # 3. Embeddings (Google API)
                    embeddings = get_embeddings(chunks)
                    
                    # 4. Vector Store (FAISS)
                    index = create_vector_store(chunks, embeddings)
                    
                    # Save to session state
                    st.session_state.vector_store = (index, chunks)
                    st.success("Done! Ready to chat.")
            else:
                st.error("Upload a file first.")

    # Chat Interface
    user_query = st.chat_input("Ask something...")

    if user_query:
        # User Message
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Assistant Message
        if st.session_state.vector_store:
            index, chunks = st.session_state.vector_store
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, context = get_answer(user_query, index, chunks)
                    st.write(answer)
                    
                    with st.expander("Show Evidence"):
                        st.write(context)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please process a PDF first.")

if __name__ == "__main__":
    main()