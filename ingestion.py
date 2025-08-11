import os
from typing import List, Dict

DATA_DIR = "data"
CHUNK_SIZE = 500  # characters (could also be tokens if using a tokenizer)
CHUNK_OVERLAP = 50

def load_documents(data_dir: str = DATA_DIR) -> Dict[str, str]:
    """Load all .txt files from the specified data folder."""
    docs: Dict[str, str] = {}
    if not os.path.isdir(data_dir):
        return docs
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs[fname] = f.read().strip()
            except FileNotFoundError:
                # File could be removed between listdir and open; skip safely
                continue
    return docs

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += max(1, chunk_size - overlap)
    return chunks

def ingest_data(data_dir: str = DATA_DIR) -> Dict[str, List[str]]:
    """
    Load and chunk all documents under data_dir.
    Returns a mapping of filename -> list of chunks.
    """
    docs = load_documents(data_dir)
    chunked_docs = {fname: chunk_text(content) for fname, content in docs.items()}
    return chunked_docs

def get_chunk(doc_name: str, i: int = 0, data_dir: str = DATA_DIR) -> str:
    """Return the i-th chunk of a document by filename from data_dir."""
    chunked_docs = ingest_data(data_dir)
    if doc_name not in chunked_docs:
        raise FileNotFoundError(f"Document '{doc_name}' not found in {data_dir}")
    return chunked_docs[doc_name][i] if chunked_docs[doc_name] else ""

if __name__ == "__main__":
    chunked_docs = ingest_data()
    print("Loaded documents with chunks:")
    for fname, chunks in chunked_docs.items():
        print(f"📄 {fname}: {len(chunks)} chunks")

    # Example: get first chunk of doc1.txt
    try:
        first_chunk = get_chunk("doc_1.txt")
        print("\nFirst chunk of doc1.txt:\n", first_chunk)
    except FileNotFoundError:
        print("doc_1.txt not found in data/")