import streamlit as st
import os
from dotenv import load_dotenv

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