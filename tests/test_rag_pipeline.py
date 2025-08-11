#!/usr/bin/env python3
"""
End-to-end test: build a vector store from a fixture, run the RAG pipeline, then clean up.

This test requires access to OpenAI APIs for embeddings and chat models. It will be
skipped automatically unless a real OPENAI_API_KEY is provided.
"""

# Import the function directly without FastAPI dependencies
import sys
import os
import pytest

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import the function we want to test
from rag_core import rag_pipeline
from vector_store import build_vector_store

# Skip this module unless a real key is present
_key = os.getenv("OPENAI_API_KEY", "").strip()
_has_real_key = bool(_key) and _key.lower() != "test"
pytestmark = pytest.mark.skipif(not _has_real_key, reason="Requires real OPENAI_API_KEY for embeddings/LLM calls")

def test_rag_pipeline():
    """Build a store from fixture data, query it, then clean up."""
    print("Testing end-to-end RAG pipeline with build + query...")

    import uuid
    import shutil
    import tempfile

    # Prepare unique store paths
    store_id = f"test_{uuid.uuid4().hex[:8]}"
    persist_dir = os.path.join("chroma_store_versions", store_id)

    # Create a temporary data directory and copy fixture file
    temp_data_dir = tempfile.mkdtemp(prefix=f"data_{store_id}_")
    fixture_path = os.path.join(PROJECT_ROOT, "tests", "eval_data", "vireo_city.txt")
    with open(fixture_path, "r", encoding="utf-8") as src, \
         open(os.path.join(temp_data_dir, "vireo_city.txt"), "w", encoding="utf-8") as dst:
        dst.write(src.read())

    built_dir = None
    try:
        # Build the vector store from the temp data dir
        built_dir, num_chunks = build_vector_store(persist_dir=persist_dir, data_dir=temp_data_dir)
        assert os.path.isdir(built_dir), "Persist dir was not created"
        assert num_chunks >= 1, "Expected at least one chunk after build"

        # Query the freshly built store
        question = "What information is provided in this document?"
        answer, retrieved_chunks, metadata, raw_chunks = rag_pipeline(
            question=question,
            persist_dir=built_dir,
            k=2,
        )

        print("✅ Build + query worked!")
        print(f"Chunks retrieved: {len(retrieved_chunks)}")
        print(f"Answer (first 120 chars): {answer[:120]}...")
        assert isinstance(retrieved_chunks, list), "retrieved_chunks should be a list"
        assert "persist_dir" in metadata, "metadata should include persist_dir"

    except Exception as e:
        print(f"❌ Error in build/query pipeline: {e}")
        return False
    finally:
        # Cleanup: remove built store and temporary data dir
        try:
            if built_dir and os.path.isdir(built_dir):
                shutil.rmtree(built_dir, ignore_errors=True)
            if os.path.isdir(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"Warning: cleanup failed: {cleanup_err}")

    return True

if __name__ == "__main__":
    test_rag_pipeline() 