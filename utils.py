from dotenv import load_dotenv
load_dotenv()

import os
import shutil
from typing import List, Tuple, Dict, Any

import numpy as np
from vector_store import load_vector_store


# Base directories used by the new store-aware design
BASE_STORE_DIR = "chroma_store_versions"
BASE_DATA_DIR = os.path.join("data", "stores")


def _paths_for_store(store_id: str) -> Tuple[str, str]:
    """Return (persist_dir, data_dir) for a given store_id."""
    persist_dir = os.path.join(BASE_STORE_DIR, store_id)
    data_dir = os.path.join(BASE_DATA_DIR, store_id)
    return persist_dir, data_dir


def count_rows_in_vector_db(store_id: str) -> int:
    """Return number of stored chunks in the Chroma vector DB for a store."""
    persist_dir, _ = _paths_for_store(store_id)
    if not os.path.isdir(persist_dir):
        return 0
    db = load_vector_store(persist_dir=persist_dir)
    return len(db.get().get("ids", []))


def print_vector_norms(store_id: str, limit: int = 5) -> None:
    """Print vector norms for the first N embeddings in the store."""
    persist_dir, _ = _paths_for_store(store_id)
    if not os.path.isdir(persist_dir):
        print(f"Store '{store_id}' not found at {persist_dir}")
        return
    db = load_vector_store(persist_dir=persist_dir)
    store_data = db.get(include=["embeddings"])  # dict with 'embeddings', 'documents', etc.
    embeddings = store_data.get("embeddings", [])

    print(f"Total vectors: {len(embeddings)}\n")
    for i, vec in enumerate(embeddings[:limit]):
        norm = np.linalg.norm(vec)
        print(f"Vector {i} norm: {norm:.6f}")


def print_chunks_for_file(store_id: str, filename: str) -> List[Tuple[int, str, Dict[str, Any]]]:
    """
    Print (and return) all chunks in the store that belong to a given source file.

    Assumes each stored Document has metadata:
        - "source": original filename (e.g., "giraffes.txt")
        - "chunk_index": integer index assigned at ingestion time

    Returns:
        List of tuples (chunk_index, text, metadata) sorted by chunk_index.
    """
    persist_dir, _ = _paths_for_store(store_id)
    if not os.path.isdir(persist_dir):
        print(f"Store '{store_id}' not found at {persist_dir}")
        return []

    db = load_vector_store(persist_dir=persist_dir)
    data = db.get(include=["documents", "metadatas"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    items: List[Tuple[int, str, Dict[str, Any]]] = []
    for text, meta in zip(documents, metadatas):
        meta = meta or {}
        if meta.get("source") == filename:
            idx = int(meta.get("chunk_index", -1))
            items.append((idx, text, meta))

    if not items:
        print(f"No chunks found for file: {filename}")
        return []

    items.sort(key=lambda x: x[0])

    print(f"Found {len(items)} chunks for '{filename}' in store '{store_id}':\n")
    for idx, text, meta in items:
        print(f"--- chunk_index = {idx} ---")
        print(text)
        print()

    return items


def reset_rag(store_id: str) -> None:
    """Delete both the vector DB directory and the data directory for a store."""
    persist_dir, data_dir = _paths_for_store(store_id)
    shutil.rmtree(persist_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    demo_store = "default"
    #print(f"Number of chunks in DB '{demo_store}': {count_rows_in_vector_db(demo_store)}")
    # print_vector_norms(demo_store, limit=5)
    print_chunks_for_file(demo_store, "frogs.txt")