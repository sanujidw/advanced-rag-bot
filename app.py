import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Page Config
st.set_page_config(page_title="Advanced RAG Assistant 🧠", layout="wide")

# Custom CSS for Source Boxes
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.title("Enterprise-Grade RAG Chatbot")
st.caption("Advanced Features: Source Citations | Metadata Tracking | Hallucination Control")

def main():
    # Load Environment Variables
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key missing! Please check your .env file.")
        
    st.write("Welcome! Upload a PDF to start.")

if __name__ == "__main__":
    main()
    
def get_pdf_text_with_metadata(pdf_docs):
    """
    Extracts text from PDFs along with page numbers (metadata).
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        pdf_name = pdf.name
        
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                # Create document chunks with metadata
                chunks = text_splitter.create_documents(
                    texts=[text], 
                    metadatas=[{"source": pdf_name, "page": i + 1}]
                )
                documents.extend(chunks)
    return documents 
def get_vector_store(documents):
    """
    Creates a FAISS vector store using Free HuggingFace embeddings.
    """
    # Using a lightweight, free model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore   