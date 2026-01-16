# 🚀 Hybrid RAG Chatbot 

A production-ready **Retrieval-Augmented Generation (RAG)** application built with a **Hybrid Architecture** to solve API rate limiting and cost issues.

This project combines **Local Embeddings (HuggingFace)** for document processing with **Google Gemini (Cloud)** for reasoning, ensuring a fast, free, and robust experience.

## 🏗 Architecture (The Hybrid Approach)
Unlike standard RAG apps that rely entirely on paid/rate-limited APIs, this system uses a dual approach:
1.  **Embeddings:** Generated locally using `Sentence-Transformers` (all-MiniLM-L6-v2).
    *   *Benefit:* Zero cost, privacy-focused, and **NO Rate Limits**.
2.  **Vector DB:** Stored locally using **FAISS** (Facebook AI Similarity Search).
3.  **Generation:** Powered by **Google Gemini 1.5 Flash** via API.
    *   *Benefit:* High-speed reasoning with a large context window.

## 🛠 Tech Stack
- **Frontend:** Streamlit
- **LLM:** Google Gemini 1.5 Flash
- **Embeddings:** Sentence-Transformers (Local CPU/GPU)
- **Vector Store:** FAISS
- **PDF Processing:** PyPDF2
- **Language:** Python 3.10+

## 🌟 Key Features
- **Zero Dependency Hell:** Built without LangChain to avoid version conflicts.
- **Robustness:** Handles large PDFs (e.g., textbooks) without hitting API quotas.
- **Cost Efficient:** Uses free-tier friendly architecture.
- **Progress Tracking:** Real-time progress bar for document processing.

## 🚀 How to Run locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```
  
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
  
3. **Set up Google API Key:**
   - Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Add it to `.env`:
     ```env
     GOOGLE_API_KEY=your_api_key_here
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

