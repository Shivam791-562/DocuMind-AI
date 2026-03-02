# 🧠 DocuMind AI: Enterprise RAG Analyst

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-orange.svg)

DocuMind AI is a full-stack Retrieval-Augmented Generation (RAG) application designed to ingest complex PDF documents and provide highly accurate, context-aware answers using Google's Gemini 2.5 LLM. Built with a modern **LangChain Expression Language (LCEL)** architecture and **FAISS** vector search, this tool prevents AI hallucinations by strictly grounding its responses in the uploaded data.

---

## 🚀 Live Demo
**👉 [Click here to launch the Live Demo](https://documind-ai.streamlit.app/)**

---

## ⚡ Key Features
- **Intelligent Document Ingestion:** Extracts text from single or multiple PDF files simultaneously using `PyPDF2`.
- **Semantic Chunking:** Utilizes `RecursiveCharacterTextSplitter` to divide massive documents into optimized overlapping chunks for vectorization.
- **Local Vector Database:** Converts text into HuggingFace embeddings (`all-MiniLM-L6-v2`) and stores them locally via FAISS for lightning-fast similarity search.
- **Modern LCEL Architecture:** Implements the latest LangChain Expression Language (`prompt | model | parser`) for streamlined, enterprise-grade LLM routing.
- **Hallucination Prevention:** Custom prompt engineering strictly limits the LLM to only answer based on the retrieved context.
- **Stateful UI:** Built with Streamlit, featuring a dynamic chat interface that disables input until documents are fully processed to prevent runtime errors.

---

## 🛠️ Technology Stack
- **Frontend / UI:** Streamlit
- **LLM Framework:** LangChain (Core & Community)
- **Generative AI Model:** Google Gemini 2.5 Flash
- **Embeddings Model:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Document Parsing:** PyPDF2

---

## 💻 Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a .env file in the root directory and add your Google API key
```env
GOOGLE_API_KEY="your_api_key_here"
```
### 5. Run the Application
```bash
streamlit run app.py
```

## 🧠 How it Works (The RAG Pipeline)
1. Load: Users upload PDF documents via the Streamlit sidebar.

2. Split: The text is extracted and split into 10,000-character chunks with a 1,000-character overlap to preserve context.

3. Embed & Store: HuggingFace generates vector embeddings for each chunk, which are saved in a local FAISS index.

4. Retrieve: When a user asks a question, the app converts the query into a vector and retrieves the top 4 most semantically similar chunks.

5. Generate: The retrieved chunks are passed as context to Gemini 2.5 Flash, which generates a natural language response.