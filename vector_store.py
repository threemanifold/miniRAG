import os
import shutil
from typing import List, Dict, Tuple
import logging
from logging_utils import log_event

import warnings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from ingestion import ingest_data
from langchain.docstore.document import Document

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "rag"

# Suppress LangChain deprecation warning about Chroma integration for this challenge project
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning  # type: ignore
    warnings.filterwarnings(
        "ignore",
        message=r".*Chroma.*deprecated.*",
        category=LangChainDeprecationWarning,
    )
except Exception:
    warnings.filterwarnings("ignore", message=r".*Chroma.*deprecated.*", category=DeprecationWarning)

def build_vector_store(persist_dir: str, data_dir: str = "data") -> Tuple[str, int]:
    """
    Build a brand-new Chroma store at persist_dir from text files under data_dir.
    Returns (persist_dir, num_chunks).
    """
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chunked: Dict[str, List[str]] = ingest_data(data_dir=data_dir)

    docs: List[Document] = []
    for source, chunks in chunked.items():
        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"source": source, "chunk_index": i}))

    if not docs:
        log_event(logging.getLogger(__name__), "vector_store.build.empty", persist_dir=persist_dir, data_dir=data_dir)
        return persist_dir, 0

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    db.persist()
    num_chunks = len(db.get(include=["documents"])['documents'])
    log_event(logging.getLogger(__name__), "vector_store.build.success", persist_dir=persist_dir, num_chunks=num_chunks)
    return persist_dir, num_chunks

def load_vector_store(persist_dir: str):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

if __name__ == "__main__":
    build_vector_store("./chroma_store_versions/default", data_dir="./data")
