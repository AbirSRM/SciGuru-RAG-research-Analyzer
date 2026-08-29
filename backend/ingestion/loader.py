import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- FIXED

def load_documents():
    """Loads all PDFs from data/papers/ and splits them into chunks."""
    paper_dir = "data/papers"
    all_chunks = []
    
    if not os.path.exists(paper_dir):
        os.makedirs(paper_dir)
        print(f"📁 Created {paper_dir}. Please add PDFs there.")
        return []

    for filename in os.listdir(paper_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(paper_dir, filename)
            print(f"📄 Loading: {filename}")
            
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = text_splitter.split_documents(pages)
            
            for chunk in chunks:
                chunk.metadata["source"] = filename
                
            all_chunks.extend(chunks)
            
    print(f"✅ Loaded {len(all_chunks)} total chunks!")
    return all_chunks