import streamlit as st
import os
import tempfile
import arxiv
import requests
import hashlib
import pickle
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="SciGuru - Research Gap Analyzer", page_icon="🔬", layout="wide")
st.title("🔬 SciGuru: Research Gap & Opportunity Analyzer")
st.caption("Ask a research question and get answers from ArXiv, your uploaded PDFs, or your local papers folder.")

# ---------------- CONFIG ----------------
BASE_INDEX_DIR = "data/indexes"
PAPERS_DIR = "data/papers"
os.makedirs(BASE_INDEX_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR, exist_ok=True)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing from your .env file!")
    st.stop()

llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.3)

# Embeddings – change device to 'cuda' if you have a GPU
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# ---------------- CORE FUNCTIONS ----------------

def process_documents(pages, metadata_source, url_source):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(pages)
    for chunk in chunks:
        chunk.metadata["source"] = metadata_source
        chunk.metadata["url"] = url_source
    return chunks

def build_vectorstore(chunks):
    st.info("🧠 Building vector space...")
    return FAISS.from_documents(chunks, embeddings)

def get_folder_hash(folder_path):
    """Hash the list of files and their modification times to detect changes."""
    if not os.path.exists(folder_path):
        return ""
    hasher = hashlib.md5()
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith('.pdf'):
            path = os.path.join(folder_path, fname)
            hasher.update(fname.encode())
            hasher.update(str(os.path.getmtime(path)).encode())
    return hasher.hexdigest()

def index_local_papers():
    """Index all PDFs in data/papers/ with caching."""
    folder_hash = get_folder_hash(PAPERS_DIR)
    cache_name = f"local_{folder_hash}" if folder_hash else "local_empty"
    cache_folder = os.path.join(BASE_INDEX_DIR, cache_name)
    index_path = os.path.join(cache_folder, "index.faiss")
    metadata_path = os.path.join(cache_folder, "metadata.pkl")

    # Check cache
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            vectorstore = FAISS.load_local(cache_folder, embeddings, allow_dangerous_deserialization=True)
            with open(metadata_path, "rb") as f:
                paper_metadata = pickle.load(f)
            st.success(f"✅ Loaded cached local index ({len(paper_metadata)} papers).")
            return vectorstore, paper_metadata
        except Exception as e:
            st.warning(f"⚠️ Failed to load cache. Rebuilding due to: {e}")

    # Build fresh
    all_chunks = []
    paper_metadata = []
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith('.pdf')]

    if not pdf_files:
        st.warning("📂 No PDFs found in 'data/papers/' folder. Please add some.")
        return None, []

    st.info(f"📄 Found {len(pdf_files)} PDFs. Indexing...")
    for filename in pdf_files:
        file_path = os.path.join(PAPERS_DIR, filename)
        paper_info = {"title": filename, "url": "Local File", "authors": ["Unknown"]}
        paper_metadata.append(paper_info)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        all_chunks.extend(process_documents(pages, filename, "Local File"))

    if not all_chunks:
        return None, []

    vectorstore = build_vectorstore(all_chunks)
    vectorstore.save_local(cache_folder)
    with open(metadata_path, "wb") as f:
        pickle.dump(paper_metadata, f)
    st.success(f"💾 Cached local index for {len(pdf_files)} papers.")
    return vectorstore, paper_metadata

def fetch_and_index_arxiv(domain_query, max_results=3):
    query_hash = hashlib.md5(domain_query.encode()).hexdigest()
    cache_folder = os.path.join(BASE_INDEX_DIR, query_hash)

    index_path = os.path.join(cache_folder, "index.faiss")
    metadata_path = os.path.join(cache_folder, "metadata.pkl")

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            vectorstore = FAISS.load_local(cache_folder, embeddings, allow_dangerous_deserialization=True)
            with open(metadata_path, "rb") as f:
                paper_metadata = pickle.load(f)
            st.success(f"✅ Loaded cached index for '{domain_query}'")
            return vectorstore, paper_metadata
        except Exception as e:
            st.warning(f"⚠️ Cache load failed. Rebuilding: {e}")

    all_chunks = []
    paper_metadata = []
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info(f"🔎 Searching ArXiv for '{domain_query}'...")
        client = arxiv.Client()
        search = arxiv.Search(query=domain_query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

        for result in client.results(search):
            paper_info = {"title": result.title, "url": result.entry_id, "authors": [a.name for a in result.authors]}
            paper_metadata.append(paper_info)

            pdf_id = result.get_short_id() if hasattr(result, 'get_short_id') else result.entry_id.split('/')[-1]
            pdf_path = os.path.join(temp_dir, f"{pdf_id}.pdf")
            try:
                result.download_pdf(dirpath=temp_dir, filename=f"{pdf_id}.pdf")
            except AttributeError:
                pdf_url = result.pdf_url
                if not pdf_url:
                    st.warning(f"⚠️ No PDF URL for {result.title}")
                    continue
                response = requests.get(pdf_url)
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_chunks.extend(process_documents(pages, result.title, result.entry_id))

        if not all_chunks:
            return None, []

        vectorstore = build_vectorstore(all_chunks)
        vectorstore.save_local(cache_folder)
        with open(metadata_path, "wb") as f:
            pickle.dump(paper_metadata, f)
        st.success("💾 Cached ArXiv results.")
        return vectorstore, paper_metadata

def index_uploaded_pdfs(uploaded_files):
    all_chunks = []
    paper_metadata = []
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info("📥 Processing uploaded PDFs...")
        for uploaded_file in uploaded_files:
            paper_info = {"title": uploaded_file.name, "url": "Local Upload", "authors": ["Unknown"]}
            paper_metadata.append(paper_info)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            all_chunks.extend(process_documents(pages, uploaded_file.name, "Local Upload"))
        if not all_chunks:
            return None, []
        return build_vectorstore(all_chunks), paper_metadata

def rerank_documents(query, docs, top_k=6):
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, doc.page_content] for doc in docs]
        scores = model.predict(pairs)
        sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        st.info(f"🔁 Re-ranked top {top_k} chunks using cross-encoder.")
        return sorted_docs[:top_k]
    except ImportError:
        st.warning("ℹ️ Re-ranker not installed (run `pip install sentence-transformers`). Using default vector search.")
        return docs[:top_k]

def generate_analysis(query, vectorstore, metadata, domain_label):
    """Generic function to generate analysis from a vectorstore."""
    if not vectorstore:
        return None

    # Retrieve and re-rank
    source_docs = vectorstore.similarity_search(query, k=10)
    source_docs = rerank_documents(query, source_docs, top_k=6)

    context_text = "\n\n".join(
        [f"--- Source: {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}"
         for doc in source_docs]
    )

    prompt = f"""
    You are an expert academic researcher analyzing the current state of a field. 
    Based ONLY on the provided excerpts from recent research papers in the domain of "{domain_label}", 
    provide a structured analysis.

    Context from recent papers:
    {context_text}

    Please provide your analysis in the following format:
    1. **Current State**: A brief 2-sentence summary of what these papers are focusing on.
    2. **Research Gaps (What's Missing)**: Identify what aspects are ignored, under-researched, or explicitly stated as limitations in the provided text.
    3. **Areas for Improvement**: Highlight methodologies, datasets, or approaches that the papers suggest need enhancement.
    4. **New Research Opportunities**: Propose 2-3 novel ideas for a new research paper based on the identified gaps.

    If the context does not contain enough information to form a complete answer, state that clearly.
    """

    try:
        response = llm.invoke(prompt)
        return response.content, context_text, source_docs
    except Exception as e:
        st.error(f"❌ An error occurred during LLM generation: {e}")
        return None, None, None

# ---------------- UI -------------------

# Choose data source
input_mode = st.radio("⚙️ Choose Data Source:", [
    "Upload Local PDFs",
    "Search the Internet (ArXiv)",
    "Use Local Database (data/papers)"
])

# Initialize session state for storing vectorstore and metadata
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.metadata = []
    st.session_state.domain_label = ""

# ---- Upload Mode ----
if input_mode == "Upload Local PDFs":
    uploaded_files = st.file_uploader("Drop your PDF files here:", type="pdf", accept_multiple_files=True)
    if st.button("Process Uploaded PDFs", type="primary") and uploaded_files:
        with st.spinner("Reading and analyzing your files..."):
            vectorstore, metadata = index_uploaded_pdfs(uploaded_files)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.metadata = metadata
                st.session_state.domain_label = "Uploaded Documents"
                st.success("✅ PDFs processed! Now ask a question below.")

# ---- ArXiv Mode ----
elif input_mode == "Search the Internet (ArXiv)":
    domain_query = st.text_input("🌐 Enter a Research Domain:", placeholder="e.g., Federated Learning in Healthcare")
    if st.button("Fetch & Analyze ArXiv Papers", type="primary") and domain_query:
        with st.spinner("Fetching papers and building index..."):
            vectorstore, metadata = fetch_and_index_arxiv(domain_query, max_results=3)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.metadata = metadata
                st.session_state.domain_label = domain_query
                st.success("✅ ArXiv papers fetched! Now ask a question below.")

# ---- Local Database Mode ----
else:
    st.info(f"📁 Your local papers folder: `{PAPERS_DIR}` (add PDFs there)")
    if st.button("🔍 Build/Refresh Local Index", type="primary"):
        with st.spinner("Indexing papers from data/papers/..."):
            vectorstore, metadata = index_local_papers()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.metadata = metadata
                st.session_state.domain_label = "Local Papers Database"
                st.success("✅ Local index ready! Now ask a question below.")

# ---- Shared Search Bar for All Modes ----
if st.session_state.vectorstore is not None:
    st.markdown("---")
    st.subheader("💬 Ask Your Research Question")
    user_query = st.text_input("What would you like to know about these papers?",
                               placeholder="e.g., What are the main limitations and future directions?")
    if st.button("🔍 Get Analysis", type="primary") and user_query:
        with st.spinner("Generating analysis..."):
            content, context, docs = generate_analysis(
                user_query,
                st.session_state.vectorstore,
                st.session_state.metadata,
                st.session_state.domain_label
            )
            if content:
                st.markdown("### 📊 Landscape Analysis")
                st.write(content)

                with st.expander("🔎 Show Self-Evaluation & Hallucination Check", expanded=False):
                    eval_prompt = f"""
                    You are an expert academic research evaluator. You will be given:
                    1. The original context excerpts.
                    2. The generated analysis provided below.

                    Context:
                    {context}

                    Generated Analysis:
                    {content}

                    Task 1: Give a factual accuracy "Confidence Score" (0-100) strictly based on how well the analysis is supported by the provided context. Do not use outside knowledge.
                    Task 2: List any specific claims, examples, or numbers in the analysis that are NOT directly stated or strongly implied in the provided context (Identify Hallucinations).
                    Task 3: Suggest 1 concrete improvement to make the analysis more grounded in the provided texts.

                    Format the response clearly with headings.
                    """
                    with st.spinner("Running quality check..."):
                        eval_response = llm.invoke(eval_prompt)
                        st.markdown("### 🔍 Self-Evaluation & Confidence Score")
                        st.write(eval_response.content)

                st.markdown("---")
                st.markdown("### 📚 Papers Analyzed (Sources)")
                for paper in st.session_state.metadata:
                    st.markdown(f"* **[{paper['title']}]({paper['url']})**")

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.markdown("### ⚙️ Architecture details")
    st.markdown("""
    * **Embeddings**: `all-MiniLM-L6-v2` (Local CPU)
    * **Vector DB**: FAISS (Persistent Caching)
    * **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    * **LLM Reasoning**: `openai/gpt-oss-120b` via Groq
    """)
    st.markdown("---")
    st.markdown("Cached indexes are saved in `data/indexes` to boost performance for repeated searches.")
    if st.session_state.vectorstore:
        st.success(f"✅ Index loaded with {len(st.session_state.metadata)} papers.")