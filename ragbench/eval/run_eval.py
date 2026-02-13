"""End-to-end evaluation runner."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import uuid

from ragbench.retrieval.protocols import Retriever, Generator, RetrievedDocument
from ragbench.metrics.retrieval_metrics import compute_retrieval_metrics
from ragbench.metrics.faithfulness import FaithfulnessEvaluator


class EvalRunner:
    """Runs end-to-end evaluation of RAG systems."""
    
    def __init__(self, retriever: Retriever, generator: Generator, faithfulness_evaluator: FaithfulnessEvaluator):
        """
        Initialize evaluation runner.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
            faithfulness_evaluator: FaithfulnessEvaluator instance
        """
        self.retriever = retriever
        self.generator = generator
        self.faithfulness_evaluator = faithfulness_evaluator
    
    def load_gold_dataset(self, gold_file: Path) -> List[Dict[str, Any]]:
        """Load gold dataset from JSONL file."""
        items = []
        with open(gold_file, "r") as f:
            for line in f:
                items.append(json.loads(line))
        return items
    
    def run_evaluation(self, gold_file: Path, output_dir: Path, top_k: int = 10):
        """
        Run evaluation on gold dataset.
        
        Args:
            gold_file: Path to gold dataset JSONL
            output_dir: Directory to save results
            top_k: Number of documents to retrieve
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load gold dataset
        gold_items = self.load_gold_dataset(gold_file)
        
        # Run evaluation for each query
        traces = []
        retrieval_metrics_list = []
        faithfulness_results = []
        
        print(f"Evaluating {len(gold_items)} queries...")
        
        for idx, item in enumerate(gold_items):
            query = item["query"]
            gold_doc_ids = set(item["gold_doc_ids"])
            expected_answer = item.get("expected_answer", "")
            
            # Retrieve documents
            retrieved_docs = self.retriever.search(query, top_k=top_k)
            retrieved_doc_ids = [doc.doc_id for doc in retrieved_docs]
            
            # Generate answer
            answer = self.generator.answer(query, retrieved_docs)
            
            # Compute retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(retrieved_doc_ids, gold_doc_ids)
            retrieval_metrics_list.append({
                "query_id": item["id"],
                "query": query,
                **retrieval_metrics
            })
            
            # Compute faithfulness
            context_texts = [doc.text for doc in retrieved_docs]
            faithfulness_result = self.faithfulness_evaluator.evaluate(answer, context_texts)
            faithfulness_results.append({
                "query_id": item["id"],
                "query": query,
                "faithfulness_score": faithfulness_result.faithfulness_score,
                "hallucination_rate": faithfulness_result.hallucination_rate,
                "failure_type": faithfulness_result.failure_type,
                "num_unsupported_sentences": len(faithfulness_result.unsupported_sentences)
            })
            
            # Create trace
            trace = {
                "query_id": item["id"],
                "query": query,
                "retrieved_doc_ids": retrieved_doc_ids,
                "gold_doc_ids": list(gold_doc_ids),
                "answer": answer,
                "expected_answer": expected_answer,
                "retrieval_metrics": retrieval_metrics,
                "faithfulness_score": faithfulness_result.faithfulness_score,
                "hallucination_rate": faithfulness_result.hallucination_rate,
                "failure_type": faithfulness_result.failure_type,
                "unsupported_sentences": faithfulness_result.unsupported_sentences
            }
            traces.append(trace)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(gold_items)} queries...")
        
        # Save results
        self._save_results(output_dir, traces, retrieval_metrics_list, faithfulness_results)
        
        print(f"\nEvaluation complete! Results saved to: {output_dir}")
        return traces
    
    def _save_results(self, output_dir: Path, traces: List[Dict], retrieval_metrics: List[Dict], faithfulness_results: List[Dict]):
        """Save evaluation results to files."""
        # Save traces
        traces_file = output_dir / "traces.jsonl"
        with open(traces_file, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")
        
        # Save retrieval metrics CSV
        if retrieval_metrics:
            retrieval_file = output_dir / "retrieval_metrics.csv"
            fieldnames = list(retrieval_metrics[0].keys())
            with open(retrieval_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(retrieval_metrics)
        
        # Save faithfulness CSV
        if faithfulness_results:
            faithfulness_file = output_dir / "faithfulness.csv"
            fieldnames = list(faithfulness_results[0].keys())
            with open(faithfulness_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(faithfulness_results)
        
        # Save hallucination failures
        hallucination_failures = [
            trace for trace in traces
            if trace["failure_type"] in ["hallucination", "retrieval_miss", "context_noise"]
        ]
        if hallucination_failures:
            failures_file = output_dir / "hallucinations.jsonl"
            with open(failures_file, "w") as f:
                for failure in hallucination_failures:
                    f.write(json.dumps(failure) + "\n")


def main():
    """Main entry point for evaluation."""
    import argparse
    from pathlib import Path
    from ragbench.retrieval.tfidf_retriever import TFIDFRetriever
    from ragbench.generation.simple_generator import SimpleGenerator
    from ragbench.metrics.faithfulness import FaithfulnessEvaluator
    
    parser = argparse.ArgumentParser(description="Run RAGBench evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config (simplified - in full version would use YAML)
    config_path = Path(args.config)
    # For now, use defaults
    corpus_path = Path("data/raw/corpus.json")
    gold_file = Path("data/gold/gold.jsonl")
    
    # Initialize components
    retriever = TFIDFRetriever(corpus_path)
    generator = SimpleGenerator()
    faithfulness_evaluator = FaithfulnessEvaluator()
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    output_dir = Path("reports") / run_id
    
    # Run evaluation
    runner = EvalRunner(retriever, generator, faithfulness_evaluator)
    runner.run_evaluation(gold_file, output_dir, top_k=10)
    
    print(f"\nRun ID: {run_id}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

