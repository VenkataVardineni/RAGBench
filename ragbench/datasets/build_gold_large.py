"""Build gold evaluation dataset from large corpus using various strategies."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def build_gold_from_corpus(
    corpus: Dict[str, str],
    num_queries: int = 500,
    strategy: str = "random_sampling"
) -> List[Dict]:
    """
    Build gold dataset from corpus using different strategies.
    
    Args:
        corpus: Dictionary mapping doc_id to text
        num_queries: Number of queries to generate
        strategy: Strategy for generating queries
            - "random_sampling": Randomly sample documents and create queries
            - "diverse": Ensure diverse document coverage
            - "difficult": Focus on longer documents or multi-doc queries
            
    Returns:
        List of gold items with query, gold_doc_ids, and expected_answer
    """
    doc_ids = list(corpus.keys())
    
    if strategy == "random_sampling":
        return _random_sampling_strategy(corpus, doc_ids, num_queries)
    elif strategy == "diverse":
        return _diverse_strategy(corpus, doc_ids, num_queries)
    elif strategy == "difficult":
        return _difficult_strategy(corpus, doc_ids, num_queries)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _random_sampling_strategy(corpus: Dict[str, str], doc_ids: List[str], num_queries: int) -> List[Dict]:
    """Random sampling strategy - randomly select documents and create queries."""
    items = []
    
    for i in range(num_queries):
        # Randomly select 1-3 documents
        num_docs = random.choice([1, 1, 1, 2, 3])  # Bias toward single doc
        selected_docs = random.sample(doc_ids, min(num_docs, len(doc_ids)))
        
        # Create query from first document
        first_doc_text = corpus[selected_docs[0]]
        query = _generate_query_from_text(first_doc_text)
        
        # Extract expected answer (first sentence or first 200 chars)
        expected_answer = _extract_answer_from_text(first_doc_text)
        
        items.append({
            "id": f"query_{i+1:06d}",
            "query": query,
            "gold_doc_ids": selected_docs,
            "expected_answer": expected_answer
        })
    
    return items


def _diverse_strategy(corpus: Dict[str, str], doc_ids: List[str], num_queries: int) -> List[Dict]:
    """Diverse strategy - ensure good coverage across documents."""
    items = []
    used_docs = set()
    
    # Shuffle to randomize
    shuffled_docs = doc_ids.copy()
    random.shuffle(shuffled_docs)
    
    doc_idx = 0
    for i in range(num_queries):
        if doc_idx >= len(shuffled_docs):
            # Reset if we've used all docs
            doc_idx = 0
            random.shuffle(shuffled_docs)
        
        # Select document(s)
        num_docs = random.choice([1, 1, 2])
        selected_docs = []
        for _ in range(num_docs):
            if doc_idx < len(shuffled_docs):
                selected_docs.append(shuffled_docs[doc_idx])
                used_docs.add(shuffled_docs[doc_idx])
                doc_idx += 1
        
        if not selected_docs:
            continue
        
        # Create query
        first_doc_text = corpus[selected_docs[0]]
        query = _generate_query_from_text(first_doc_text)
        expected_answer = _extract_answer_from_text(first_doc_text)
        
        items.append({
            "id": f"query_{i+1:06d}",
            "query": query,
            "gold_doc_ids": selected_docs,
            "expected_answer": expected_answer
        })
    
    print(f"Diverse strategy: Used {len(used_docs)} unique documents out of {len(doc_ids)}")
    return items


def _difficult_strategy(corpus: Dict[str, str], doc_ids: List[str], num_queries: int) -> List[Dict]:
    """Difficult strategy - focus on longer documents and multi-doc queries."""
    # Sort by document length
    doc_lengths = [(doc_id, len(corpus[doc_id])) for doc_id in doc_ids]
    doc_lengths.sort(key=lambda x: x[1], reverse=True)
    
    # Take longer documents
    long_docs = [doc_id for doc_id, _ in doc_lengths[:len(doc_lengths)//2]]
    
    items = []
    for i in range(num_queries):
        # More likely to select multiple documents
        num_docs = random.choice([1, 2, 2, 3, 3])
        selected_docs = random.sample(long_docs, min(num_docs, len(long_docs)))
        
        first_doc_text = corpus[selected_docs[0]]
        query = _generate_query_from_text(first_doc_text)
        expected_answer = _extract_answer_from_text(first_doc_text)
        
        items.append({
            "id": f"query_{i+1:06d}",
            "query": query,
            "gold_doc_ids": selected_docs,
            "expected_answer": expected_answer
        })
    
    return items


def _generate_query_from_text(text: str) -> str:
    """Generate a query from document text."""
    # Simple strategy: take first sentence and convert to question
    sentences = text.split('.')
    first_sentence = sentences[0].strip() if sentences else text[:100]
    
    # Try to convert to question form
    if len(first_sentence) > 100:
        first_sentence = first_sentence[:100] + "..."
    
    # Simple question patterns
    question_patterns = [
        f"What is {first_sentence.split()[0] if first_sentence.split() else 'this'}?",
        f"Tell me about {first_sentence.split()[0] if first_sentence.split() else 'this'}.",
        f"Explain {first_sentence.split()[0] if first_sentence.split() else 'this'}.",
        first_sentence  # Keep as is
    ]
    
    return random.choice(question_patterns)


def _extract_answer_from_text(text: str, max_length: int = 300) -> str:
    """Extract expected answer from text."""
    # Take first sentence or first max_length characters
    sentences = text.split('.')
    if sentences and len(sentences[0]) > 50:
        return sentences[0].strip() + "."
    else:
        return text[:max_length].strip() + ("..." if len(text) > max_length else "")


def main():
    """Main function to build large gold dataset."""
    import argparse
    from ragbench.datasets.load_corpus import load_corpus_from_json
    
    parser = argparse.ArgumentParser(description="Build gold dataset from large corpus")
    parser.add_argument("--corpus", type=str, default="data/raw/corpus.json",
                       help="Path to corpus JSON file")
    parser.add_argument("--output", type=str, default="data/gold/gold.jsonl",
                       help="Output gold dataset file")
    parser.add_argument("--num-queries", type=int, default=500,
                       help="Number of queries to generate")
    parser.add_argument("--strategy", type=str, default="diverse",
                       choices=["random_sampling", "diverse", "difficult"],
                       help="Query generation strategy")
    
    args = parser.parse_args()
    
    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    corpus = load_corpus_from_json(Path(args.corpus))
    print(f"Loaded {len(corpus)} documents")
    
    # Build gold dataset
    print(f"Building gold dataset with {args.num_queries} queries using '{args.strategy}' strategy...")
    gold_items = build_gold_from_corpus(corpus, args.num_queries, args.strategy)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in gold_items:
            f.write(json.dumps(item) + "\n")
    
    print(f"\nCreated gold dataset with {len(gold_items)} queries")
    print(f"Saved to: {output_path}")
    
    # Statistics
    unique_docs = set()
    for item in gold_items:
        unique_docs.update(item["gold_doc_ids"])
    print(f"Coverage: {len(unique_docs)} unique documents referenced")


if __name__ == "__main__":
    main()

