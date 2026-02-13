"""Generate evaluation reports."""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import statistics


def load_traces(traces_file: Path) -> List[Dict]:
    """Load traces from JSONL file."""
    traces = []
    with open(traces_file, "r") as f:
        for line in f:
            traces.append(json.loads(line))
    return traces


def compute_aggregate_metrics(traces: List[Dict]) -> Dict[str, float]:
    """Compute aggregate metrics across all queries."""
    if not traces:
        return {}
    
    metrics = {}
    
    # Retrieval metrics
    precision_values = {k: [] for k in [1, 3, 5, 10]}
    recall_values = {k: [] for k in [1, 3, 5, 10]}
    ndcg_values = {k: [] for k in [1, 3, 5, 10]}
    mrr_values = []
    
    # Faithfulness metrics
    faithfulness_scores = []
    hallucination_rates = []
    
    for trace in traces:
        retrieval_metrics = trace.get("retrieval_metrics", {})
        
        # Collect retrieval metrics
        for k in [1, 3, 5, 10]:
            if f"precision@{k}" in retrieval_metrics:
                precision_values[k].append(retrieval_metrics[f"precision@{k}"])
            if f"recall@{k}" in retrieval_metrics:
                recall_values[k].append(retrieval_metrics[f"recall@{k}"])
            if f"ndcg@{k}" in retrieval_metrics:
                ndcg_values[k].append(retrieval_metrics[f"ndcg@{k}"])
        
        if "mrr" in retrieval_metrics:
            mrr_values.append(retrieval_metrics["mrr"])
        
        # Collect faithfulness metrics
        if "faithfulness_score" in trace:
            faithfulness_scores.append(trace["faithfulness_score"])
        if "hallucination_rate" in trace:
            hallucination_rates.append(trace["hallucination_rate"])
    
    # Compute averages
    for k in [1, 3, 5, 10]:
        if precision_values[k]:
            metrics[f"avg_precision@{k}"] = statistics.mean(precision_values[k])
        if recall_values[k]:
            metrics[f"avg_recall@{k}"] = statistics.mean(recall_values[k])
        if ndcg_values[k]:
            metrics[f"avg_ndcg@{k}"] = statistics.mean(ndcg_values[k])
    
    if mrr_values:
        metrics["avg_mrr"] = statistics.mean(mrr_values)
    
    if faithfulness_scores:
        metrics["avg_faithfulness"] = statistics.mean(faithfulness_scores)
    
    if hallucination_rates:
        metrics["avg_hallucination_rate"] = statistics.mean(hallucination_rates)
    
    return metrics


def categorize_failures(traces: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize failures by type."""
    categories = {
        "retrieval_miss": [],
        "context_noise": [],
        "hallucination": [],
        "none": []
    }
    
    for trace in traces:
        failure_type = trace.get("failure_type", "none")
        if failure_type in categories:
            categories[failure_type].append(trace)
    
    return categories


def generate_report(output_dir: Path) -> str:
    """
    Generate evaluation report.
    
    Args:
        output_dir: Directory containing evaluation results
        
    Returns:
        Report as markdown string
    """
    traces_file = output_dir / "traces.jsonl"
    if not traces_file.exists():
        return "# Evaluation Report\n\nNo traces found."
    
    traces = load_traces(traces_file)
    aggregate_metrics = compute_aggregate_metrics(traces)
    failure_categories = categorize_failures(traces)
    
    # Build report
    report_lines = []
    report_lines.append("# RAGBench Evaluation Report\n")
    report_lines.append(f"**Total Queries:** {len(traces)}\n")
    
    # Retrieval Metrics Table
    report_lines.append("## Retrieval Metrics\n")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    
    for k in [1, 3, 5, 10]:
        if f"avg_precision@{k}" in aggregate_metrics:
            report_lines.append(f"| Precision@{k} | {aggregate_metrics[f'avg_precision@{k}']:.3f} |")
        if f"avg_recall@{k}" in aggregate_metrics:
            report_lines.append(f"| Recall@{k} | {aggregate_metrics[f'avg_recall@{k}']:.3f} |")
        if f"avg_ndcg@{k}" in aggregate_metrics:
            report_lines.append(f"| nDCG@{k} | {aggregate_metrics[f'avg_ndcg@{k}']:.3f} |")
    
    if "avg_mrr" in aggregate_metrics:
        report_lines.append(f"| MRR | {aggregate_metrics['avg_mrr']:.3f} |")
    
    report_lines.append("")
    
    # Faithfulness Metrics
    report_lines.append("## Generation Metrics\n")
    if "avg_faithfulness" in aggregate_metrics:
        report_lines.append(f"**Average Faithfulness Score:** {aggregate_metrics['avg_faithfulness']:.3f}\n")
    if "avg_hallucination_rate" in aggregate_metrics:
        report_lines.append(f"**Average Hallucination Rate:** {aggregate_metrics['avg_hallucination_rate']:.3f}\n")
    report_lines.append("")
    
    # Failure Breakdown
    report_lines.append("## Failure Breakdown\n")
    report_lines.append("| Failure Type | Count | Percentage |")
    report_lines.append("|--------------|-------|------------|")
    
    total = len(traces)
    for failure_type, items in failure_categories.items():
        count = len(items)
        percentage = (count / total * 100) if total > 0 else 0
        report_lines.append(f"| {failure_type} | {count} | {percentage:.1f}% |")
    
    report_lines.append("")
    
    # Top Failure Examples
    report_lines.append("## Top Failure Examples\n")
    
    # Show examples from each failure category
    for failure_type in ["retrieval_miss", "hallucination", "context_noise"]:
        examples = failure_categories[failure_type][:3]  # Top 3 examples
        if examples:
            report_lines.append(f"### {failure_type.replace('_', ' ').title()}\n")
            for i, example in enumerate(examples, 1):
                report_lines.append(f"#### Example {i}\n")
                report_lines.append(f"**Query:** {example['query']}\n")
                report_lines.append(f"**Retrieved Docs:** {', '.join(example['retrieved_doc_ids'][:3])}\n")
                report_lines.append(f"**Gold Docs:** {', '.join(example['gold_doc_ids'])}\n")
                report_lines.append(f"**Answer:** {example['answer'][:200]}...\n")
                report_lines.append(f"**Faithfulness Score:** {example.get('faithfulness_score', 0):.3f}\n")
                if example.get('unsupported_sentences'):
                    report_lines.append(f"**Unsupported Sentences:** {len(example['unsupported_sentences'])}\n")
                report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    """Main entry point for report generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    output_dir = Path(args.results_dir)
    report_md = generate_report(output_dir)
    
    # Save report
    report_file = output_dir / "report.md"
    with open(report_file, "w") as f:
        f.write(report_md)
    
    print(f"Report generated: {report_file}")
    print("\n" + "="*80)
    print(report_md)
    print("="*80)

