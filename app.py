import streamlit as st
import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("API Key not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# --- Helper Functions ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def process_pdfs_with_metadata(pdf_docs):
    """
    Reads PDFs and splits text while KEEPING track of Page Numbers and Filenames.
    Returns a list of dictionaries: [{'text': '...', 'source': 'doc.pdf', 'page': 1}, ...]
    """
    chunked_data = []
    chunk_size = 1000
    overlap = 200

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name
        
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if not text:
                continue
                
            # Split page text into chunks
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                # Stotre chunk with metadata
                chunked_data.append({
                    'text': chunk_text,
                    'source': pdf_name,
                    'page': i + 1  # Page numbers start at 1
                })
                
                start += chunk_size - overlap
                
    return chunked_data

def get_embeddings(chunked_data):
    """Extracts text from the dictionaries and generates embeddings."""
    model = load_embedding_model()
    # Only encode the 'text' part
    texts = [item['text'] for item in chunked_data]
    embeddings = model.encode(texts)
    return np.array(embeddings, dtype='float32')

def create_vector_store(embeddings):
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def get_answer(query, index, chunked_data):
    # 1. Embed Query
    embedding_model = load_embedding_model()
    query_vec = embedding_model.encode([query])
    query_vec = np.array(query_vec, dtype='float32')
    
    # 2. Search FAISS
    D, I = index.search(query_vec, k=3)
    
    relevant_context_text = ""
    sources_info = []

    # 3. Retrieve Text AND Metadata
    for idx in I[0]:
        if idx < len(chunked_data):
            data = chunked_data[idx]
            relevant_context_text += data['text'] + "\n\n"
            
            # Collect Source Info for display
            sources_info.append(f"📄 **{data['source']}** (Page {data['page']})")
            
    # 4. Generate Answer with Gemini
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a helpful assistant. Answer the question based ONLY on the following context.
    If the answer is not in the context, say "I don't know based on this document."
    
    Context:
    {relevant_context_text}
    
    Question: {query}
    """
    
    response = model.generate_content(prompt)
    return response.text, sources_info, relevant_context_text

# --- UI Application ---

def main():
    st.title("🚀 Hybrid RAG Chatbot")
    st.caption("Embeddings: Local | Chat: Gemini | Status: No Rate Limits | Metadata: Enabled")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.header("📂 Documents")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            if pdf_docs:
                progress_bar = st.progress(0, text="Starting...")
                
                try:
                    # Step 1 & 2: Read & Chunk with Metadata
                    progress_bar.progress(30, text="📖 Reading & Chunking with Metadata...")
                    chunked_data = process_pdfs_with_metadata(pdf_docs)
                    st.write(f"Created {len(chunked_data)} chunks with page numbers.")
                    
                    # Step 3: Embeddings
                    progress_bar.progress(60, text="🧠 Generating Embeddings...")
                    embeddings = get_embeddings(chunked_data)
                    
                    # Step 4: Vector Store
                    progress_bar.progress(90, text="💾 Saving to Database...")
                    index = create_vector_store(embeddings)
                    
                    # Store Index AND the Data List (for metadata lookup)
                    st.session_state.vector_store = (index, chunked_data)
                    
                    progress_bar.progress(100, text="✅ Done!")
                    st.success("Processed! Now ask questions.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Upload a file first.")

    user_query = st.chat_input("Ask something...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        if st.session_state.vector_store:
            index, chunked_data = st.session_state.vector_store
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get answer AND source info
                        answer, sources, raw_context = get_answer(user_query, index, chunked_data)
                        
                        st.write(answer)
                        
                        # --- Display Sources Nicely ---
                        st.markdown("---")
                        st.subheader("📚 Sources Used:")
                        for source in sources:
                            st.caption(source)
                        
                        # Expandable raw text
                        with st.expander("🔍 View Exact Context Text"):
                            st.write(raw_context)
                            
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Please upload a PDF first.")

if __name__ == "__main__":
    main()