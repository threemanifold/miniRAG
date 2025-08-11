# rag_core.py
"""
Core RAG functionality without FastAPI dependencies.
This module contains the centralized RAG pipeline function.
"""

from typing import List, Any, Tuple, Dict
import logging
from logging_utils import log_event

from retrieval import retrieve_relevant_chunks
from summarizer import batch_summarize_chunks
from answer_generator import answer_with_citations, get_agent_prompt_text, get_answer_model_name
from vector_store import EMBEDDING_MODEL
from summarizer import get_summarizer_model_name
from query_generator import rewrite_query, get_query_model_name

def rag_pipeline(question: str, persist_dir: str, k: int = 3, history: List[Tuple[str, str]] = None):
    """
    Centralized RAG pipeline function that can be used by both API and UI.
    
    Args:
        question: The user's question
        persist_dir: Path to the vector store
        k: Number of chunks to retrieve
        history: Optional conversation history for context
    
    Returns:
        Tuple of (answer, retrieved_chunks, metadata, raw_chunks)
    """
    # Step 1: Rewrite user question into a retrieval-friendly query
    rewritten_query = rewrite_query(question, history=history)

    # Retrieve relevant chunks using rewritten query
    chunks = retrieve_relevant_chunks(rewritten_query, k=k, persist_dir=persist_dir)
    chunk_texts = [chunk[0] for chunk in chunks]  # Extract text from retrieved chunks
    chunk_metadata = [chunk[1] for chunk in chunks]  # Extract metadata

    # Generate summaries
    summaries = batch_summarize_chunks(question, chunk_texts)

    # Generate answer with citations
    answer = answer_with_citations(question, list(zip(chunk_texts, chunk_metadata)), history=history)

    # Prepare retrieved chunks for response
    retrieved_chunks = [
        {
            "summary": summaries[i],
            "metadata": chunk_metadata[i]
        }
        for i in range(len(summaries))
    ]

    # Prepare metadata
    prompt_text = get_agent_prompt_text()
    metadata = {
        "embeddings_model": EMBEDDING_MODEL,
        "llm_model_answer": get_answer_model_name(),
        "llm_model_summarizer": get_summarizer_model_name(),
        "llm_model_query": get_query_model_name(),
        "prompt_used_preview": (prompt_text[:300] + ("…" if len(prompt_text) > 300 else "")),
        "retrieval_strategy": f"Top‑k cosine similarity (k={k})",
        "persist_dir": persist_dir,
        "rewritten_query": rewritten_query,
    }

    # Minimal structured logs for challenge
    logger = logging.getLogger(__name__)
    try:
        # Log retrieval results (summaries, indices, scores) and strategy
        retrieval_log = {
            "retrieval_strategy": metadata["retrieval_strategy"],
            "k": k,
            "results": [
                {
                    "summary": summaries[i],
                    "source": chunk_metadata[i].get("source"),
                    "chunk_index": chunk_metadata[i].get("chunk_index"),
                    "score": chunk_metadata[i].get("score"),
                }
                for i in range(len(summaries))
            ],
        }
        log_event(logger, "retrieval.results", **retrieval_log)

        # Log LLM prompt/response details and models
        llm_log = {
            "embeddings_model": metadata["embeddings_model"],
            "answer_model": metadata.get("llm_model_answer"),
            "summarizer_model": metadata.get("llm_model_summarizer"),
            "query_model": metadata.get("llm_model_query"),
            "prompt_preview": metadata.get("prompt_used_preview"),
        }
        log_event(logger, "llm.metadata", **llm_log)
        # Log answer preview and length for observability
        try:
            preview = answer[:300] + ("…" if len(answer) > 300 else "")
            log_event(logger, "llm.answer", preview=preview, length=len(answer))
        except Exception:
            pass
    except Exception:
        # Logging should never break the pipeline
        pass
    
    # Also return raw chunks for UI display
    raw_chunks = list(zip(chunk_texts, chunk_metadata))
    
    return answer, retrieved_chunks, metadata, raw_chunks 