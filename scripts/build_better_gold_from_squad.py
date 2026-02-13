#!/usr/bin/env python3
"""
Build a better gold dataset from SQuAD using actual questions and answers.

This creates a gold dataset where queries are actual questions from SQuAD,
and gold documents are the contexts that contain the answers.
This should give much better retrieval results.
"""

import json
from pathlib import Path
from typing import Dict, List, Set


def load_squad_dataset() -> List[Dict]:
    """Load SQuAD dataset with actual questions."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad", split="train")
    
    items = []
    context_to_doc_id = {}
    doc_id_counter = 0
    
    # First pass: map contexts to document IDs
    seen_contexts = {}
    for example in dataset:
        context = example["context"]
        context_hash = hash(context)
        
        if context_hash not in seen_contexts:
            seen_contexts[context_hash] = f"squad_{doc_id_counter:06d}"
            doc_id_counter += 1
    
    # Second pass: create gold items
    print("Creating gold dataset from SQuAD questions...")
    for i, example in enumerate(dataset):
        if i >= 500:  # Limit to 500 queries
            break
        
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"]
        
        # Get document ID for this context
        context_hash = hash(context)
        doc_id = seen_contexts[context_hash]
        
        # Use first answer as expected answer
        expected_answer = answers[0] if answers else ""
        
        items.append({
            "id": f"query_{i+1:06d}",
            "query": question,
            "gold_doc_ids": [doc_id],
            "expected_answer": expected_answer
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} queries...")
    
    return items, seen_contexts


def create_corpus_from_contexts(context_to_doc_id: Dict) -> Dict[str, str]:
    """Create corpus from SQuAD contexts."""
    try:
        from datasets import load_dataset
    except ImportError:
        from datasets import load_dataset
    
    dataset = load_dataset("squad", split="train")
    
    corpus = {}
    seen_contexts = set()
    
    for example in dataset:
        context = example["context"]
        context_hash = hash(context)
        
        if context_hash not in seen_contexts:
            doc_id = context_to_doc_id.get(context_hash)
            if doc_id:
                corpus[doc_id] = context
                seen_contexts.add(context_hash)
    
    return corpus


def main():
    """Main function."""
    print("=" * 60)
    print("Building Better Gold Dataset from SQuAD")
    print("=" * 60)
    print()
    print("This uses actual SQuAD questions (not generated ones)")
    print("which should give much better retrieval results.")
    print()
    
    # Load SQuAD and create gold items
    gold_items, context_to_doc_id = load_squad_dataset()
    
    # Create corpus
    print("\nCreating corpus from contexts...")
    corpus = create_corpus_from_contexts(context_to_doc_id)
    
    # Save corpus
    corpus_file = Path("data/raw/squad_corpus_better.json")
    corpus_file.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_file, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"✓ Saved corpus: {corpus_file} ({len(corpus)} documents)")
    
    # Save gold dataset
    gold_file = Path("data/gold/gold_squad_better.jsonl")
    gold_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_file, "w") as f:
        for item in gold_items:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved gold dataset: {gold_file} ({len(gold_items)} queries)")
    
    print(f"\n{'='*60}")
    print("Statistics:")
    print(f"{'='*60}")
    print(f"  Corpus documents: {len(corpus)}")
    print(f"  Gold queries: {len(gold_items)}")
    
    unique_docs = set()
    for item in gold_items:
        unique_docs.update(item["gold_doc_ids"])
    print(f"  Documents referenced: {len(unique_docs)}")
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Update configs/squad.yaml:")
    print("   data:")
    print("     corpus: \"data/raw/squad_corpus_better.json\"")
    print("     gold_dataset: \"data/gold/gold_squad_better.jsonl\"")
    print("\n2. Use embedding retriever for better results:")
    print("   retrieval:")
    print("     type: \"embedding\"")
    print("\n3. Run evaluation:")
    print("   python -m ragbench.eval --config configs/squad.yaml")


if __name__ == "__main__":
    main()

