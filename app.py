import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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
def get_rag_chain(vectorstore):
    """
    Creates the Retrieval Chain with a custom system prompt.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # System Prompt: Strict instructions to prevent hallucinations
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Think step by step before answering.
    If the answer is not in the context, say "I don't have enough information in the documents to answer that."
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Chain 1: Generate Answer
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Chain 2: Retrieve + Generate
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain 