import streamlit as st
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 1. Setup Environment
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

# Configure Google Gemini (Used only for generating the final answer)
genai.configure(api_key=api_key)

# Page Configuration
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# --- Helper Functions ---

@st.cache_resource
def load_embedding_model():
    """
    Loads the local embedding model from HuggingFace.
    Using @st.cache_resource ensures the model is loaded only once,
    improving app performance.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_pdf_text(pdf_docs):
    """Extracts text content from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_chunks(text, chunk_size=1000, overlap=200):
    """
    Splits the extracted text into manageable chunks.
    overlap ensures context is preserved between chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(text_chunks):
    """
    Generates embeddings for text chunks using the Local Model.
    This runs on your machine's CPU, avoiding Google API rate limits.
    """
    model = load_embedding_model()
    # Encode the text chunks into vectors
    embeddings = model.encode(text_chunks)
    return np.array(embeddings, dtype='float32')

def create_vector_store(text_chunks, embeddings):
    """
    Creates a FAISS index (Vector Database) from the embeddings.
    """
    # all-MiniLM-L6-v2 creates 384-dimensional vectors
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def get_answer(query, index, text_chunks):
    """
    1. Embeds the user query locally.
    2. Searches the FAISS index for relevant context.
    3. Sends context + query to Google Gemini for the answer.
    """
    
    # 1. Embed the user query using the SAME local model
    embedding_model = load_embedding_model()
    query_vec = embedding_model.encode([query])
    query_vec = np.array(query_vec, dtype='float32')
    
    # 2. Search FAISS for the top 3 most similar chunks
    D, I = index.search(query_vec, k=3)
    
    relevant_context = ""
    for idx in I[0]:
        if idx < len(text_chunks):
            relevant_context += text_chunks[idx] + "\n\n"
            
    # 3. Send the prompt to Google Gemini
    # Using 'gemini-1.5-flash' as it is faster and currently supported
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't know based on this document."
    
    Context:
    {relevant_context}
    
    Question: {query}
    """
    
    response = model.generate_content(prompt)
    return response.text, relevant_context

# --- Main Application Logic ---

def main():
    st.title("🚀 Hybrid RAG Chatbot")
    st.caption("Embeddings: Local (HuggingFace) | Chat: Cloud (Gemini) | Status: No Rate Limits")

    # Initialize Session State to store history and vector store
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Sidebar for File Upload
    with st.sidebar:
        st.header("📂 Documents")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            if pdf_docs:
                # Initialize Progress Bar
                progress_bar = st.progress(0, text="Starting process...")
                
                try:
                    # Step 1: Text Extraction
                    progress_bar.progress(20, text="📖 Reading PDF text...")
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Step 2: Chunking
                    progress_bar.progress(40, text="✂️ Splitting text into chunks...")
                    chunks = get_chunks(raw_text)
                    st.write(f"Created {len(chunks)} text chunks.")
                    
                    # Step 3: Embeddings (Local)
                    # This might take a moment depending on CPU speed
                    progress_bar.progress(60, text="🧠 Generating Embeddings (Local CPU)...")
                    embeddings = get_embeddings(chunks)
                    
                    # Step 4: Vector Store
                    progress_bar.progress(90, text="💾 Saving to Vector Database...")
                    index = create_vector_store(chunks, embeddings)
                    
                    # Store in session state
                    st.session_state.vector_store = (index, chunks)
                    
                    # Completion
                    progress_bar.progress(100, text="✅ Done!")
                    st.success("Documents processed successfully! You can now chat.")
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Please upload a PDF file first.")

    # Chat Interface
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Display User Message
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate Assistant Response
        if st.session_state.vector_store:
            index, chunks = st.session_state.vector_store
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, context = get_answer(user_query, index, chunks)
                        st.write(answer)
                        
                        # Expandable section to show the source evidence
                        with st.expander("🔍 View Source Evidence"):
                            st.write(context)
                            
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please upload and process a PDF document first.")

if __name__ == "__main__":
    main()