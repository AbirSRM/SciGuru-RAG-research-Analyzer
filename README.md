[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-100000?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**SciGuru** is an AI-powered research assistant that analyzes recent academic papers (from ArXiv, uploaded PDFs, or your own local folder) to identify **research gaps, limitations, and future opportunities**. It uses a Retrieval-Augmented Generation (RAG) pipeline with local embeddings, a FAISS vector store, and a Groq LLM for reasoning.

---

## 🚀 Quick Setup (Docker - Recommended)

The easiest and most reliable way to run SciGuru is using Docker. This ensures all system dependencies (like FAISS and PyTorch) are isolated and perfectly configured.

### Prerequisites
- Docker & Docker Compose installed.
- A free [Groq API Key](https://console.groq.com/keys).

### 1. Configure API Key
Copy the example environment file and add your Groq API key:
```bash
cp .env.example .env
# Open .env and add your key: GROQ_API_KEY=your_key_here
```

### 2. Run the Application
Start the application effortlessly using the provided run script:
```bash
chmod +x run.sh
./run.sh
```
*(Note: The very first time you run this, it will take a minute to download the Docker image. Every subsequent run will start instantly in < 2 seconds!)*

A browser window will automatically be accessible at **http://localhost:8501**.

---

## 🌐 Live Demonstrations (Ngrok)
If you wish to share the application externally (e.g., during a technical interview or presentation), ensure the app is running locally, open a **new terminal**, and run:
```bash
ngrok http 8501
```
Share the generated `ngrok.app` URL with your audience! They can access your locally-hosted app directly from their phone or browser.

---

## ✨ Features
- **Three Input Modes**
  - Upload individual PDFs (temporary analysis)
  - Search ArXiv by research domain (with caching)
  - **Scan a local folder (`data/papers/`)** – build a persistent, reusable index of your own paper collection

- **Intelligent Analysis**
  - Extracts and chunks document content
  - Builds a vector index using `all-MiniLM-L6-v2` (CPU-friendly)
  - Re-ranks retrieved passages using a cross-encoder for semantic relevance
  - Generates structured analysis: **Current State, Research Gaps, Areas for Improvement, New Opportunities**

- **Self-Evaluation**
  - Includes a hallucination check with confidence scoring
  - Compares generated analysis against provided context

- **Offline Caching**
  - Reuses previously built indexes for repeated queries
  - Speeds up repeated searches dramatically (persisted across Docker restarts via Named Volumes!)

---

## 🛠️ Tech Stack
| Component | Tool |
|-----------|------|
| Frontend | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local CPU) |
| Vector DB | FAISS (persistent caching) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Containerization | Docker & Docker Compose |
| Document Processing | PyPDF, LangChain |

---

## 🧠 How It Works (High‑Level)
1. **Input** – User chooses one of three sources: upload, ArXiv, or local folder.
2. **Processing** – Text is extracted and split into overlapping chunks.
3. **Embedding** – Chunks are embedded locally with `all-MiniLM-L6-v2`.
4. **Indexing** – FAISS builds a vector index (cached on disk for reuse).
5. **Retrieval** – For a research question, the most relevant chunks are retrieved.
6. **Re‑ranking** – A cross‑encoder re‑orders chunks by semantic similarity.
7. **Generation** – Groq LLM synthesises a structured gap analysis.
8. **Self‑Eval** – The LLM also evaluates its own output against the context to flag hallucinations.

---

## 📁 Project Structure (Key Files)
```
SciGuru/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition (optimized for CPU PyTorch)
├── docker-compose.yml      # Docker orchestration & volume mapping
├── run.sh                  # One-click start script
├── instructions.txt        # Interview/Demo talking points guide
├── .env.example            # Template for API key
├── .gitignore              # Recommended ignore patterns
└── data/
    ├── indexes/            # Cached FAISS indexes (auto‑generated)
    └── papers/             # Place your own PDFs here for the local database
```

---

## 🛠️ Manual Local Setup (Without Docker)

If you prefer *not* to use Docker, you can run the app natively on your machine:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AbirSRM/SciGuru-RAG-research-Analyzer.git
   cd SciGuru-RAG-research-Analyzer
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Your Groq API Key**
   ```bash
   cp .env.example .env
   # Edit .env and paste your Groq API key
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is open-source and available under the MIT License.

## 🙏 Acknowledgements
- Built with [LangChain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/).
- Embeddings and re‑ranker models from [Sentence‑Transformers](https://www.sbert.net/).
- LLM inference powered by [Groq](https://groq.com/).
