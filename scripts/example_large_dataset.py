#!/usr/bin/env python3
"""
Example script showing how to set up RAGBench with a large dataset.

This demonstrates:
1. Creating a sample large corpus (you can replace this with your own data)
2. Loading it into RAGBench format
3. Building a gold evaluation dataset
4. Running evaluation
"""

import json
from pathlib import Path

def create_example_large_corpus(output_file: Path, num_docs: int = 1000):
    """
    Create an example large corpus for testing.
    In practice, you'd load this from your real data source.
    """
    print(f"Creating example corpus with {num_docs} documents...")
    
    # Example: Simulate a corpus with diverse topics
    topics = [
        "machine learning", "natural language processing", "computer vision",
        "data science", "software engineering", "web development",
        "databases", "cloud computing", "cybersecurity", "artificial intelligence"
    ]
    
    corpus = {}
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        doc_id = f"doc_{i:06d}"
        # Simulate document text
        text = f"""
        This is a document about {topic}. 
        Document {i} contains information related to {topic} and its applications.
        {topic} is an important field with many practical uses.
        Researchers and practitioners work on various aspects of {topic}.
        The field of {topic} continues to evolve with new techniques and methodologies.
        """
        corpus[doc_id] = text.strip()
    
    # Save corpus
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)
    
    print(f"✓ Created corpus: {output_file}")
    print(f"  Documents: {len(corpus)}")
    return corpus


def main():
    """Main example workflow."""
    print("=" * 60)
    print("RAGBench Large Dataset Example")
    print("=" * 60)
    print()
    
    # Step 1: Create or load corpus
    corpus_file = Path("data/raw/corpus_large.json")
    
    if not corpus_file.exists():
        print("Step 1: Creating example corpus...")
        create_example_large_corpus(corpus_file, num_docs=1000)
    else:
        print(f"Step 1: Using existing corpus: {corpus_file}")
    
    print()
    
    # Step 2: Build gold dataset
    print("Step 2: Building gold evaluation dataset...")
    print("Run this command:")
    print(f"  python -m ragbench.datasets.build_gold_large \\")
    print(f"    --corpus {corpus_file} \\")
    print(f"    --output data/gold/gold_large.jsonl \\")
    print(f"    --num-queries 500 \\")
    print(f"    --strategy diverse")
    print()
    
    # Step 3: Update config
    print("Step 3: Update configs/demo.yaml to point to your gold dataset:")
    print("  data:")
    print("    gold_dataset: \"data/gold/gold_large.jsonl\"")
    print()
    
    # Step 4: Run evaluation
    print("Step 4: Run evaluation:")
    print("  python -m ragbench.eval --config configs/demo.yaml")
    print()
    
    print("=" * 60)
    print("For real datasets, see: docs/LARGE_DATASET_GUIDE.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

