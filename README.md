# 🧠 Advanced RAG Chatbot (Google Gemini + LangChain)

This is an Enterprise-Grade RAG (Retrieval-Augmented Generation) application built with Streamlit. It allows users to chat with their PDF documents with **hallucination control** and **precise source citations**.

## 🌟 Key Features
*   **Metadata Tracking:** Identifies exact page numbers for every answer.
*   **Zero Cost:** Uses Google Gemini Pro (Free Tier) and HuggingFace Local Embeddings.
*   **Prompt Engineering:** Custom system prompts to reduce false information.
*   **Interactive UI:** View source text snippets within the chat.

## 🛠 Tech Stack
*   **Python**
*   **LangChain** (Orchestration)
*   **Google Gemini** (LLM)
*   **FAISS** (Vector Database)
*   **Streamlit** (Frontend)

## 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to a `.env` file.
4. Run: `streamlit run app.py`