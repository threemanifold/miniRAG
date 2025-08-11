#!/usr/bin/env python3
"""
Evaluation helpers colocated under tests/ per challenge scope.
Builds a temporary vector store from tests/eval_data/vireo_city.txt,
evaluates retrieval accuracy against ground_truth.py using an LLM judge,
and cleans up all temporary artifacts afterwards.
"""

from __future__ import annotations

import os
import shutil
import statistics
import time
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from rag_core import rag_pipeline
from vector_store import build_vector_store


class RetrievalAccuracyEvaluator:
    def __init__(self, persist_dir: str, k: int = 5):
        self.persist_dir = persist_dir
        self.k = k
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.judge_prompt = (
            "You are an accuracy judge for a RAG (Retrieval-Augmented Generation) system.\n\n"
            "Your task is to determine if the retrieved chunks contain the expected information.\n\n"
            "For each piece of expected information, you must decide if it is present in the retrieved chunks.\n"
            "- Answer TRUE if the information is clearly present in the chunks\n"
            "- Answer FALSE if the information is missing or only partially present\n\n"
            "Be strict but fair. The information should be explicitly stated or clearly implied in the chunks.\n\n"
            "Question: {question}\n\n"
            "Expected Information to Check ({num_expected} items):\n{expected_info}\n\n"
            "Retrieved Chunks:\n{retrieved_chunks}\n\n"
            "You must return exactly {num_expected} boolean values as a Python list.\n"
            "Return ONLY the list of booleans, nothing else.\n\n"
            "Example: [True, False, True]"
        )

    def _check_information_containment(
        self, question: str, expected_info: List[str], retrieved_chunks: List[Tuple[str, Dict]]
    ) -> List[bool]:
        if not retrieved_chunks:
            return [False] * len(expected_info)

        chunks_text = "\n\n".join(
            [f"Chunk {i+1} (source: {meta.get('source', 'unknown')}):\n{text}" for i, (text, meta) in enumerate(retrieved_chunks)]
        )
        expected_info_text = "\n".join([f"{i+1}. {info}" for i, info in enumerate(expected_info)])

        prompt = self.judge_prompt.format(
            question=question,
            expected_info=expected_info_text,
            retrieved_chunks=chunks_text,
            num_expected=len(expected_info),
        )

        messages = [
            SystemMessage(content="You are a precise accuracy judge for RAG systems. Return only Python lists of booleans."),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content.strip()

        try:
            import ast

            results = ast.literal_eval(response_text)
            if not isinstance(results, list):
                raise ValueError("Response is not a list")
            if len(results) != len(expected_info):
                results = [False] * len(expected_info)
            results = [bool(result) for result in results]
        except Exception:
            results = [False] * len(expected_info)

        return results

    def evaluate_all(self, questions_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        recalls: List[float] = []
        latencies_ms: List[float] = []

        for test_case in questions_dataset:
            question = test_case["question"]
            expected_info = test_case["relevant_information"]

            try:
                t0 = time.perf_counter()
                answer, retrieved_chunks, metadata, raw_chunks = rag_pipeline(
                    question=question, persist_dir=self.persist_dir, k=self.k
                )
                t_ms = (time.perf_counter() - t0) * 1000.0
                containment = self._check_information_containment(question, expected_info, raw_chunks)
                recall = sum(containment) / len(expected_info) if expected_info else 0.0
                results.append(
                    {
                        "question": question,
                        "recall": recall,
                        "containment_results": containment,
                        "num_expected": len(expected_info),
                        "num_found": sum(containment),
                        "retrieved_chunks_count": len(raw_chunks),
                        "latency_ms": t_ms,
                    }
                )
                recalls.append(recall)
                latencies_ms.append(t_ms)
            except Exception as e:
                results.append({"question": question, "recall": 0.0, "error": str(e)})

        if recalls:
            avg_recall = statistics.mean(recalls)
            std_recall = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        else:
            avg_recall = 0.0
            std_recall = 0.0

        return {
            "average_recall": avg_recall,
            "std_recall": std_recall,
            "total_questions": len(questions_dataset),
            "successful_evaluations": len(recalls),
            "individual_results": results,
            "avg_latency_ms": (statistics.mean(latencies_ms) if latencies_ms else 0.0),
            "std_latency_ms": (statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0),
        }


def run_retrieval_accuracy_fixture(k: int = 5, max_questions: Optional[int] = None) -> Dict[str, Any]:
    """Build a temp store from tests/eval_data/vireo_city.txt, evaluate, and cleanup."""
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    fixture_src = os.path.join(PROJECT_ROOT, "tests", "eval_data", "vireo_city.txt")

    built_dir = None
    temp_data_dir = None
    try:
        # Prepare temp persist dir
        store_id = f"eval_{uuid.uuid4().hex[:8]}"
        persist_dir = os.path.join(PROJECT_ROOT, "chroma_store_versions", store_id)
        os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

        # Prepare temp data dir with fixture
        temp_data_dir = tempfile.mkdtemp(prefix=f"data_{store_id}_", dir=PROJECT_ROOT)
        fixture_dst = os.path.join(temp_data_dir, "vireo_city.txt")
        with open(fixture_src, "r", encoding="utf-8") as src, open(fixture_dst, "w", encoding="utf-8") as dst:
            dst.write(src.read())

        # Build store
        built_dir, num_chunks = build_vector_store(persist_dir=persist_dir, data_dir=temp_data_dir)
        assert os.path.isdir(built_dir) and num_chunks >= 1, "Failed to build evaluation store"

        # Import questions dataset and evaluate
        from tests.eval_data.ground_truth import questions_dataset

        dataset = questions_dataset[:max_questions] if max_questions else questions_dataset
        evaluator = RetrievalAccuracyEvaluator(persist_dir=built_dir, k=k)
        results = evaluator.evaluate_all(dataset)
        return results
    finally:
        # Cleanup temp artifacts
        try:
            if built_dir and os.path.isdir(built_dir):
                shutil.rmtree(built_dir, ignore_errors=True)
            if temp_data_dir and os.path.isdir(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)
        except Exception:
            pass


