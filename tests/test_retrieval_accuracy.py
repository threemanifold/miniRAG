#!/usr/bin/env python3
"""
Retrieval Accuracy Evaluator for RAG System

This module evaluates the retrieval accuracy of the RAG system by:
1. Running the RAG pipeline with k=5 for each test question
2. Using an LLM judge to check if retrieved chunks contain expected information
3. Computing recall metrics (contained info / total expected info)
4. Returning average and standard deviation of recall scores
"""
import sys
import os
import statistics
import tempfile
import shutil
import uuid
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add parent directory to path to import our modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_core import rag_pipeline
from vector_store import build_vector_store
from tests.eval_data.ground_truth import questions_dataset

class RetrievalAccuracyEvaluator:
    """Evaluates retrieval accuracy using LLM judge"""
    
    def __init__(self, persist_dir: str, k: int = 5):
        """
        Initialize the evaluator
        
        Args:
            persist_dir: Path to the vector store
            k: Number of chunks to retrieve (default: 5)
        """
        self.persist_dir = persist_dir
        self.k = k
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # LLM judge prompt for checking information containment
        self.judge_prompt = """You are an accuracy judge for a RAG (Retrieval-Augmented Generation) system.

Your task is to determine if the retrieved chunks contain the expected information.

For each piece of expected information, you must decide if it is present in the retrieved chunks.
- Answer TRUE if the information is clearly present in the chunks
- Answer FALSE if the information is missing or only partially present

Be strict but fair. The information should be explicitly stated or clearly implied in the chunks.

Question: {question}

Expected Information to Check ({num_expected} items):
{expected_info}

Retrieved Chunks:
{retrieved_chunks}

You must return exactly {num_expected} boolean values as a Python list.
Return ONLY the list of booleans, nothing else.

Example: [True, False, True]"""

    def _check_information_containment(self, question: str, expected_info: List[str], 
                                     retrieved_chunks: List[Tuple[str, Dict]]) -> List[bool]:
        """
        Use LLM judge to check if each piece of expected information is contained in retrieved chunks
        
        Args:
            question: The original question
            expected_info: List of expected information pieces
            retrieved_chunks: List of (chunk_text, metadata) tuples
            
        Returns:
            List of boolean values indicating if each piece of info was found
        """
        # If no chunks are retrieved, all expected information is missing
        if not retrieved_chunks:
            print(f"WARNING: No chunks retrieved for question: {question[:50]}...")
            return [False] * len(expected_info)
        
        # Format retrieved chunks for the prompt
        chunks_text = "\n\n".join([
            f"Chunk {i+1} (source: {meta.get('source', 'unknown')}):\n{text}"
            for i, (text, meta) in enumerate(retrieved_chunks)
        ])
        
        # Format expected information
        expected_info_text = "\n".join([
            f"{i+1}. {info}" for i, info in enumerate(expected_info)
        ])
        
        # Create the prompt
        prompt = self.judge_prompt.format(
            question=question,
            expected_info=expected_info_text,
            retrieved_chunks=chunks_text,
            num_expected=len(expected_info)
        )
        
        # Get LLM judgment
        messages = [
            SystemMessage(content="You are a precise accuracy judge for RAG systems. Return only Python lists of booleans."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse the structured boolean list response
        try:
            # Try to evaluate the response as a Python list
            import ast
            results = ast.literal_eval(response_text)
            
            # Validate that it's a list of booleans with correct length
            if not isinstance(results, list):
                raise ValueError("Response is not a list")
            
            if len(results) != len(expected_info):
                print(f"Warning: Expected {len(expected_info)} results, got {len(results)}")
                results = [False] * len(expected_info)
            
            # Ensure all elements are booleans
            results = [bool(result) for result in results]
            
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse LLM response as boolean list: {e}")
            print(f"Response was: {response_text}")
            # If parsing failed, assume all information is missing
            results = [False] * len(expected_info)
        
        return results

    def evaluate_single_question(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate retrieval accuracy for a single question
        
        Args:
            test_case: Dictionary containing question and expected information
            
        Returns:
            Dictionary with evaluation results
        """
        question = test_case["question"]
        expected_info = test_case["relevant_information"]
        
        print(f"\nEvaluating question: {question[:100]}...")
        
        # Run RAG pipeline
        try:
            answer, retrieved_chunks, metadata, raw_chunks = rag_pipeline(
                question=question,
                persist_dir=self.persist_dir,
                k=self.k
            )
            
            # Check information containment
            containment_results = self._check_information_containment(
                question, expected_info, raw_chunks
            )
            
            # Calculate recall
            recall = sum(containment_results) / len(expected_info)
            
            # Prepare detailed results
            detailed_results = []
            for i, (info, found) in enumerate(zip(expected_info, containment_results)):
                detailed_results.append({
                    "expected_info": info,
                    "found": found
                })
            
            return {
                "question": question,
                "recall": recall,
                "containment_results": containment_results,
                "detailed_results": detailed_results,
                "num_expected": len(expected_info),
                "num_found": sum(containment_results),
                "retrieved_chunks_count": len(raw_chunks)
            }
            
        except Exception as e:
            print(f"Error evaluating question: {e}")
            return {
                "question": question,
                "recall": 0.0,
                "error": str(e)
            }

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate retrieval accuracy for all test questions
        
        Returns:
            Dictionary with overall evaluation results
        """
        print(f"Starting retrieval accuracy evaluation with k={self.k}")
        print(f"Vector store path: {self.persist_dir}")
        print(f"Number of test questions: {len(questions_dataset)}")
        
        results = []
        recalls = []
        
        for i, test_case in enumerate(questions_dataset):
            print(f"\n--- Question {i+1}/{len(questions_dataset)} ---")
            result = self.evaluate_single_question(test_case)
            results.append(result)
            
            if "recall" in result:
                recalls.append(result["recall"])
                print(f"Recall: {result['recall']:.3f} ({result['num_found']}/{result['num_expected']} found)")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Calculate overall statistics
        if recalls:
            avg_recall = statistics.mean(recalls)
            std_recall = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        else:
            avg_recall = 0.0
            std_recall = 0.0
        
        overall_results = {
            "average_recall": avg_recall,
            "std_recall": std_recall,
            "total_questions": len(questions_dataset),
            "successful_evaluations": len(recalls),
            "individual_results": results
        }
        
        print(f"\n{'='*50}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*50}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Standard Deviation: {std_recall:.3f}")
        print(f"Questions Evaluated: {len(recalls)}/{len(questions_dataset)}")
        
        return overall_results

def main():
    """Main function to run the evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval accuracy")
    parser.add_argument("--persist_dir", type=str, default="",
                        help="Path to an existing vector store. If omitted or missing, a temp store will be built from tests/eval_data/vireo_city.txt")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")

    args = parser.parse_args()

    # If persist_dir not provided or does not exist, build a temporary store from the fixture
    built_temp = False
    temp_data_dir = None
    persist_dir = args.persist_dir
    if not persist_dir or not os.path.isdir(persist_dir):
        print("No valid persist_dir provided; building a temporary store from fixture data…")
        built_temp = True
        store_id = f"eval_{uuid.uuid4().hex[:8]}"
        persist_dir = os.path.join(PROJECT_ROOT, "chroma_store_versions", store_id)
        os.makedirs(os.path.dirname(persist_dir), exist_ok=True)

        # Prepare temporary data directory with fixture file
        temp_data_dir = tempfile.mkdtemp(prefix=f"data_{store_id}_", dir=PROJECT_ROOT)
        fixture_src = os.path.join(PROJECT_ROOT, "tests", "eval_data", "vireo_city.txt")
        fixture_dst = os.path.join(temp_data_dir, "vireo_city.txt")
        with open(fixture_src, "r", encoding="utf-8") as src, open(fixture_dst, "w", encoding="utf-8") as dst:
            dst.write(src.read())

        # Build the store
        built_dir, num_chunks = build_vector_store(persist_dir=persist_dir, data_dir=temp_data_dir)
        assert os.path.isdir(built_dir) and num_chunks >= 1, "Failed to build temporary evaluation store"
        print(f"Built temp store at {built_dir} with {num_chunks} chunks")

    # Run evaluation
    evaluator = RetrievalAccuracyEvaluator(persist_dir, args.k)
    results = evaluator.evaluate_all()

    # Save results to file
    import json
    output_file = os.path.join(PROJECT_ROOT, f"retrieval_accuracy_results_k{args.k}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Cleanup temporary artifacts
    if built_temp:
        try:
            if os.path.isdir(persist_dir):
                shutil.rmtree(persist_dir, ignore_errors=True)
            if temp_data_dir and os.path.isdir(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"Warning: cleanup failed: {cleanup_err}")

if __name__ == "__main__":
    main() 