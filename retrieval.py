from typing import List, Tuple
from langchain.docstore.document import Document
from vector_store import load_vector_store
import logging
from logging_utils import log_event


class RetrievalError(Exception):
    pass

def retrieve_relevant_chunks(query: str, k: int, persist_dir: str) -> List[Tuple[str, dict]]:
    logger = logging.getLogger(__name__)
    try:
        db = load_vector_store(persist_dir)
        results = db.similarity_search_with_score(query, k=k)

        out = []
        for doc, euclid in results:
            cos = 1 - (euclid ** 2) / 2
            meta = dict(doc.metadata)
            meta["score"] = round(cos, 4)
            out.append((doc.page_content, meta))
        return out
    except Exception as e:
        log_event(
            logger,
            "retrieval.error",
            error_type=type(e).__name__,
            message=str(e),
            persist_dir=persist_dir,
        )
        raise RetrievalError(str(e))

if __name__ == "__main__":
    q_1 = "What are transformers and how do they work?"
    q_2 = "What do you know about birds?"
    chunks = retrieve_relevant_chunks(q_1, k=2)
    for i, (text, meta) in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(text)
        print("Metadata:", meta)
