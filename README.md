[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-100000?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**SciGuru** is an advanced, AI-driven research intelligence engine designed to autonomously parse, index, and analyze academic literature. By leveraging a multi-stage Retrieval-Augmented Generation (RAG) architecture, SciGuru dynamically identifies epistemological research gaps, methodological limitations, and future innovation opportunities across any scientific domain.

---

## 🧠 System Architecture & Technical Pipeline

SciGuru implements a robust, localized NLP pipeline orchestrated via LangChain to ensure maximum context relevance while minimizing external API dependencies. 

1. **Multi-Modal Data Ingestion**: The system programmatically interfaces with the ArXiv API for dynamic literature retrieval, whilst also supporting asynchronous local PDF parsing via PyPDFLoader.
2. **Deterministic Chunking Strategy**: Documents are parsed and recursively partitioned into highly cohesive, overlapping context windows to preserve semantic integrity during vectorization.
3. **Local Embedding Projection**: Text chunks are projected into a dense semantic vector space utilizing the HuggingFace `all-MiniLM-L6-v2` transformer model operating entirely locally on CPU, ensuring zero data-egress for proprietary documents.
4. **FAISS Vector Indexing**: High-dimensional vectors are stored and indexed using Meta's FAISS (Facebook AI Similarity Search), enabling sub-millisecond approximate nearest neighbor (ANN) retrieval. The index state is persistently cached via Docker volumes.
5. **Two-Stage Retrieval & Cross-Encoding**: Initial context retrieval via cosine similarity is subsequently re-ranked using a dedicated Cross-Encoder (`ms-marco-MiniLM-L-6-v2`). This radically improves Precision@K by explicitly scoring query-document pairs before passing them to the generative model.
6. **LLM Synthesis & Reasoning**: The re-ranked context payload is injected into a specialized prompt template and evaluated by `llama-3.3-70b-versatile` (via Groq's high-throughput LPU inference engine) to synthesize structured gap analyses.
7. **Autonomous Hallucination Mitigation**: SciGuru implements a deterministic self-reflection protocol. The LLM acts as an independent evaluator to score its own generative output strictly against the retrieved grounding context, flagging unsupported claims and outputting a confidence metric.

---

## 🚀 Quick Setup (Docker)

The entire microservice architecture is fully containerized. The provided bash script handles dependency provisioning, environment orchestration, and volume mounting for persistent vector storage.

### Prerequisites
- Docker & Docker Compose installed and running.
- A free [Groq API Key](https://console.groq.com/keys) for Llama-3 inference.

### 1. Configure the Environment
Clone the repository and set up your environment variables:
```bash
git clone https://github.com/AbirSRM/SciGuru-RAG-research-Analyzer.git
cd SciGuru-RAG-research-Analyzer

cp .env.example .env
# Open .env and add your Groq API key: GROQ_API_KEY=your_key_here
```

### 2. Run the Engine
A unified executable is provided to build the image and spin up the container:
```bash
chmod +x run.sh
./run.sh
```

The application will bind to port 8501 and is accessible immediately at: **http://localhost:8501**.

---

## ✨ Core Capabilities

- **Dynamic Literature Indexing**: On-the-fly vectorization of ArXiv repositories based on complex domain queries.
- **Local Knowledge Base**: Persistent FAISS indexing of custom PDF libraries stored in `data/papers/`, enabling offline semantic search against proprietary datasets.
- **Structured Output Generation**: Synthesizes literature into four critical dimensions: *Current State*, *Research Gaps*, *Areas for Improvement*, and *New Opportunities*.
- **Hardware Optimized**: Built to run gracefully on consumer hardware by offloading heavy LLM inference to Groq while handling embeddings locally.

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend UI** | Streamlit |
| **LLM Engine** | `llama-3.3-70b-versatile` (via Groq) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Re-Ranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Database** | FAISS |
| **Orchestration** | LangChain |
| **Containerization** | Docker & Docker Compose (Named Volumes) |

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the issues page if you want to contribute.

## 📄 License
This project is open-source and available under the MIT License.
