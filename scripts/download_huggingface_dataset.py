#!/usr/bin/env python3
"""
Download a real dataset from HuggingFace for RAGBench evaluation.

This script downloads publicly available datasets that work well for RAG evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List
import random


def download_squad_dataset(output_file: Path, max_docs: int = 500) -> Dict[str, str]:
    """
    Download SQuAD dataset (Question Answering dataset with context).
    This creates a corpus from the context passages.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    print("Downloading SQuAD dataset from HuggingFace...")
    print("This may take a few minutes on first run...\n")
    
    # Load SQuAD dataset
    dataset = load_dataset("squad", split="train")
    
    corpus = {}
    seen_contexts = set()
    doc_id = 0
    
    print(f"Processing {len(dataset)} examples...")
    for i, example in enumerate(dataset):
        if doc_id >= max_docs:
            break
        
        context = example["context"]
        # Use context hash to avoid duplicates
        context_hash = hash(context)
        
        if context_hash not in seen_contexts and len(context) > 100:
            seen_contexts.add(context_hash)
            corpus[f"squad_{doc_id:06d}"] = context
            doc_id += 1
            
            if doc_id % 50 == 0:
                print(f"  Processed {doc_id} documents...")
    
    # Save corpus
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Downloaded {len(corpus)} unique contexts")
    print(f"✓ Saved to: {output_file}")
    
    return corpus


def download_wikipedia_simple(output_file: Path, max_docs: int = 500) -> Dict[str, str]:
    """
    Download Wikipedia articles using the 'wikipedia' dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    print("Downloading Wikipedia dataset from HuggingFace...")
    print("This may take a few minutes on first run...\n")
    
    # Load Wikipedia dataset (simple English Wikipedia)
    dataset = load_dataset("wikipedia", "20220301.simple", split="train[:1000]")
    
    corpus = {}
    doc_id = 0
    
    print(f"Processing {len(dataset)} articles...")
    for i, article in enumerate(dataset):
        if doc_id >= max_docs:
            break
        
        text = article.get("text", "")
        title = article.get("title", f"article_{i}")
        
        if len(text) > 200:  # Filter very short articles
            corpus[f"wiki_{doc_id:06d}"] = text[:5000]  # Limit to 5000 chars
            doc_id += 1
            
            if doc_id % 50 == 0:
                print(f"  Processed {doc_id} articles...")
    
    # Save corpus
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Downloaded {len(corpus)} articles")
    print(f"✓ Saved to: {output_file}")
    
    return corpus


def download_news_dataset(output_file: Path, max_docs: int = 500) -> Dict[str, str]:
    """
    Download news articles dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    
    print("Downloading news dataset from HuggingFace...")
    print("This may take a few minutes on first run...\n")
    
    # Try to load a news dataset
    try:
        # Use CNN/DailyMail dataset (summarization dataset with articles)
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:2000]")
        
        corpus = {}
        doc_id = 0
        
        print(f"Processing {len(dataset)} articles...")
        for i, example in enumerate(dataset):
            if doc_id >= max_docs:
                break
            
            article = example.get("article", "")
            if len(article) > 200:
                corpus[f"news_{doc_id:06d}"] = article[:3000]  # Limit length
                doc_id += 1
                
                if doc_id % 50 == 0:
                    print(f"  Processed {doc_id} articles...")
        
        # Save corpus
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(corpus, f, indent=2)
        
        print(f"\n✓ Downloaded {len(corpus)} articles")
        print(f"✓ Saved to: {output_file}")
        
        return corpus
    except Exception as e:
        print(f"Error loading news dataset: {e}")
        print("Falling back to SQuAD dataset...")
        return download_squad_dataset(output_file, max_docs)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real dataset from HuggingFace")
    parser.add_argument("--dataset", type=str, default="squad",
                       choices=["squad", "wikipedia", "news"],
                       help="Dataset to download")
    parser.add_argument("--output", type=str, default="data/raw/real_corpus.json",
                       help="Output corpus file")
    parser.add_argument("--max-docs", type=int, default=500,
                       help="Maximum number of documents")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("=" * 60)
    print(f"Downloading {args.dataset} dataset")
    print("=" * 60)
    print()
    
    if args.dataset == "squad":
        corpus = download_squad_dataset(output_path, args.max_docs)
    elif args.dataset == "wikipedia":
        corpus = download_wikipedia_simple(output_path, args.max_docs)
    elif args.dataset == "news":
        corpus = download_news_dataset(output_path, args.max_docs)
    
    print(f"\n{'='*60}")
    print("Corpus Statistics:")
    print(f"{'='*60}")
    print(f"  Total documents: {len(corpus)}")
    total_chars = sum(len(text) for text in corpus.values())
    print(f"  Total characters: {total_chars:,}")
    if corpus:
        avg_length = total_chars // len(corpus)
        print(f"  Average length: {avg_length:,} characters")
        min_length = min(len(text) for text in corpus.values())
        max_length = max(len(text) for text in corpus.values())
        print(f"  Min length: {min_length:,} characters")
        print(f"  Max length: {max_length:,} characters")
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print(f"1. Build gold evaluation dataset:")
    print(f"   python -m ragbench.datasets.build_gold_large \\")
    print(f"     --corpus {output_path} \\")
    print(f"     --output data/gold/gold_real.jsonl \\")
    print(f"     --num-queries 500 \\")
    print(f"     --strategy diverse")
    print(f"\n2. Update configs/demo.yaml:")
    print(f"   data:")
    print(f"     corpus: \"{output_path}\"")
    print(f"     gold_dataset: \"data/gold/gold_real.jsonl\"")
    print(f"\n3. Run evaluation:")
    print(f"   python -m ragbench.eval --config configs/demo.yaml")


if __name__ == "__main__":
    main()

