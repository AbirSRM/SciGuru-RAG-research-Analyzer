import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from backend.ingestion.loader import load_documents

def build_index():
    """Creates a FAISS vector store and saves it to disk."""
    print("🚀 Starting to build the AI index...")
    
    # 1. Load the chunks
    chunks = load_documents()
    if not chunks:
        print("❌ No PDF chunks found. Add PDFs to data/papers/")
        return
    
    # 2. Load the embedding model (this runs on your CPU, very light)
    # "all-MiniLM-L6-v2" is small, fast, and perfect for your laptop.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Create the FAISS index from the chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. Save it locally so you don't have to rebuild it every time
    os.makedirs("data/indexes", exist_ok=True)
    vectorstore.save_local("data/indexes/sciguru_index")
    
    print("🎉 Index built and saved to data/indexes/sciguru_index!")
    return vectorstore

# This runs the builder when you execute the file directly
if __name__ == "__main__":
    build_index()