#!/usr/bin/env python3
"""
API endpoint functionality and error-handling tests using FastAPI TestClient.
These tests stub the RAG pipeline to avoid external API calls.
"""

import os
import shutil
import json
from typing import List, Dict, Tuple

from fastapi.testclient import TestClient

# Ensure project root in path
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app as app_module


client = TestClient(app_module.app)


def _make_store_dir(store_id: str) -> str:
    base = os.path.join(PROJECT_ROOT, "chroma_store_versions")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, store_id)
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_store_dir(store_id: str) -> None:
    path = os.path.join(PROJECT_ROOT, "chroma_store_versions", store_id)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_success_with_raw_chunks(monkeypatch):
    store_id = "test_api_store"
    _make_store_dir(store_id)

    def stub_rag_pipeline(question: str, persist_dir: str, k: int = 3, history=None):
        answer = "stubbed answer"
        retrieved_chunks = [
            {"summary": "s1", "metadata": {"source": "doc1.txt", "chunk_index": 0, "score": 0.9}},
            {"summary": "s2", "metadata": {"source": "doc2.txt", "chunk_index": 3, "score": 0.88}},
        ]
        metadata = {
            "embeddings_model": "text-embedding-3-small",
            "llm_model_answer": "gpt-4o",
            "llm_model_summarizer": "gpt-4o-mini",
            "prompt_used_preview": "...",
            "retrieval_strategy": f"Top-k cosine similarity (k={k})",
        }
        raw_chunks: List[Tuple[str, Dict]] = [
            ("raw text 1", {"source": "doc1.txt", "chunk_index": 0, "score": 0.9}),
            ("raw text 2", {"source": "doc2.txt", "chunk_index": 3, "score": 0.88}),
        ]
        return answer, retrieved_chunks, metadata, raw_chunks

    # Monkeypatch the app-level symbol imported in app.py
    monkeypatch.setattr(app_module, "rag_pipeline", stub_rag_pipeline)

    body = {
        "store_id": store_id,
        "question": "What is inside?",
        "k": 2,
        "history": [["user", "hi"], ["assistant", "hello"]],
        "include_raw_chunks": True,
    }
    r = client.post("/query", json=body)
    try:
        # Validate success response
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "stubbed answer"
        assert isinstance(data.get("retrieved_chunks"), list)
        assert len(data.get("retrieved_chunks")) == 2
        assert isinstance(data.get("metadata"), dict)
        # raw_chunks should be present since we requested them
        assert isinstance(data.get("raw_chunks"), list)
        assert len(data["raw_chunks"]) == 2

        # Check tracing headers
        assert "X-Request-ID" in r.headers
        assert "X-Response-Time-ms" in r.headers
        # Ensure response time header is a float-like string
        float(r.headers["X-Response-Time-ms"])  # will raise if not numeric
    finally:
        _cleanup_store_dir(store_id)


def test_query_empty_question_400():
    store_id = "test_api_store_empty"
    _make_store_dir(store_id)
    try:
        r = client.post("/query", json={
            "store_id": store_id,
            "question": "   ",
        })
        assert r.status_code == 400
        data = r.json()
        assert data.get("message") == "Question cannot be empty."
        assert "request_id" in data
        assert "X-Request-ID" in r.headers
    finally:
        _cleanup_store_dir(store_id)


def test_query_store_not_found_404():
    r = client.post("/query", json={
        "store_id": "does_not_exist",
        "question": "q",
    })
    assert r.status_code == 404
    data = r.json()
    assert data.get("message") == "Vector store not found. Build it first."
    assert "request_id" in data
    assert "X-Request-ID" in r.headers


def test_query_retrieval_failure_502(monkeypatch):
    store_id = "test_api_retrieval_fail"
    _make_store_dir(store_id)

    def raise_retrieval(*args, **kwargs):
        raise app_module.RetrievalError("boom")

    monkeypatch.setattr(app_module, "rag_pipeline", raise_retrieval)

    try:
        r = client.post("/query", json={
            "store_id": store_id,
            "question": "q",
        })
        assert r.status_code == 502
        data = r.json()
        assert data.get("message") == "Upstream retrieval failed. Please try again later."
        assert "request_id" in data
        assert "X-Request-ID" in r.headers
    finally:
        _cleanup_store_dir(store_id)


def test_query_pipeline_failure_502(monkeypatch):
    store_id = "test_api_pipeline_fail"
    _make_store_dir(store_id)

    def raise_runtime(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "rag_pipeline", raise_runtime)

    try:
        r = client.post("/query", json={
            "store_id": store_id,
            "question": "q",
        })
        assert r.status_code == 502
        data = r.json()
        assert data.get("message") == "Upstream LLM or pipeline failed. Please try again later."
        assert "request_id" in data
        assert "X-Request-ID" in r.headers
    finally:
        _cleanup_store_dir(store_id)


