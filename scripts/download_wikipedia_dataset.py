#!/usr/bin/env python3
"""
Download Wikipedia articles for RAGBench evaluation.

This script downloads Wikipedia articles on various topics to create
a real-world corpus for testing.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List
import time
import random


def fetch_wikipedia_article(title: str) -> Dict[str, str]:
    """
    Fetch a Wikipedia article by title.
    
    Args:
        title: Wikipedia article title
        
    Returns:
        Dictionary with 'id' and 'text' fields
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Get full article text
        full_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title.replace(' ', '_')}"
        full_response = requests.get(full_url, timeout=10)
        
        if full_response.status_code == 200:
            # Extract text from HTML (simplified)
            import re
            from html import unescape
            text = unescape(full_response.text)
            # Remove HTML tags (simple approach)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Use extract if available, otherwise use first 2000 chars
            extract = data.get("extract", "")
            if len(extract) < 500 and text:
                # Use full text (limited)
                full_text = text[:2000] if len(text) > 2000 else text
                article_text = extract + " " + full_text if extract else full_text
            else:
                article_text = extract
        else:
            article_text = data.get("extract", "")
        
        return {
            "id": title.replace(" ", "_").lower(),
            "title": title,
            "text": article_text.strip()
        }
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None


def download_wikipedia_corpus(
    topics: List[str],
    output_file: Path,
    max_articles: int = 100
) -> Dict[str, str]:
    """
    Download Wikipedia articles on given topics.
    
    Args:
        topics: List of Wikipedia article titles/topics
        output_file: Path to save corpus JSON
        max_articles: Maximum number of articles to download
        
    Returns:
        Dictionary mapping doc_id to text
    """
    corpus = {}
    downloaded = 0
    
    print(f"Downloading up to {max_articles} Wikipedia articles...")
    print("This may take a few minutes...\n")
    
    for i, topic in enumerate(topics[:max_articles]):
        print(f"[{i+1}/{min(len(topics), max_articles)}] Fetching: {topic}...", end=" ", flush=True)
        
        article = fetch_wikipedia_article(topic)
        
        if article and article["text"]:
            doc_id = f"wiki_{article['id']}"
            corpus[doc_id] = article["text"]
            downloaded += 1
            print(f"✓ ({len(article['text'])} chars)")
        else:
            print("✗ Failed")
        
        # Be respectful - rate limiting
        time.sleep(0.5)
    
    # Save corpus
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Downloaded {downloaded} articles")
    print(f"✓ Saved to: {output_file}")
    
    return corpus


def get_diverse_topics() -> List[str]:
    """Get a diverse list of Wikipedia topics for evaluation."""
    return [
        # Technology
        "Artificial Intelligence",
        "Machine Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Deep Learning",
        "Neural Network",
        "Transformer",
        "Large Language Model",
        # Science
        "Quantum Computing",
        "Quantum Mechanics",
        "General Relativity",
        "Evolution",
        "DNA",
        "Photosynthesis",
        "Climate Change",
        # History
        "World War II",
        "Renaissance",
        "Industrial Revolution",
        "Ancient Rome",
        "Ancient Greece",
        # Geography
        "Mount Everest",
        "Amazon River",
        "Pacific Ocean",
        "Sahara Desert",
        # Literature
        "Shakespeare",
        "Homer",
        "Dante Alighieri",
        # Philosophy
        "Philosophy",
        "Ethics",
        "Logic",
        # Mathematics
        "Mathematics",
        "Calculus",
        "Statistics",
        "Geometry",
        # Medicine
        "Medicine",
        "Anatomy",
        "Surgery",
        # Arts
        "Renaissance Art",
        "Classical Music",
        "Jazz",
        # Sports
        "Olympic Games",
        "Football",
        "Basketball",
        # Economics
        "Economics",
        "Capitalism",
        "Globalization",
        # Space
        "Solar System",
        "Mars",
        "Black Hole",
        "Big Bang",
        # Biology
        "Biology",
        "Ecology",
        "Genetics",
        # Chemistry
        "Chemistry",
        "Periodic Table",
        "Chemical Reaction",
        # Physics
        "Physics",
        "Electromagnetism",
        "Thermodynamics",
        # Engineering
        "Engineering",
        "Electrical Engineering",
        "Mechanical Engineering",
        # Computer Science
        "Computer Science",
        "Algorithm",
        "Data Structure",
        "Programming Language",
        # Psychology
        "Psychology",
        "Cognitive Science",
        "Neuroscience",
        # Social Sciences
        "Sociology",
        "Anthropology",
        "Political Science",
        # More diverse topics
        "Internet",
        "World Wide Web",
        "Cryptography",
        "Blockchain",
        "Renewable Energy",
        "Solar Power",
        "Wind Power",
        "Electric Vehicle",
        "Space Exploration",
        "International Space Station",
        "Telescope",
        "Microscope",
        "Telescope",
        "Microscope",
    ]


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wikipedia dataset for RAGBench")
    parser.add_argument("--output", type=str, default="data/raw/wikipedia_corpus.json",
                       help="Output corpus file")
    parser.add_argument("--max-articles", type=int, default=100,
                       help="Maximum number of articles to download")
    parser.add_argument("--topics-file", type=str, default=None,
                       help="JSON file with list of topics (optional)")
    
    args = parser.parse_args()
    
    # Get topics
    if args.topics_file:
        with open(args.topics_file, "r") as f:
            topics = json.load(f)
    else:
        topics = get_diverse_topics()
    
    # Download corpus
    output_path = Path(args.output)
    corpus = download_wikipedia_corpus(
        topics,
        output_path,
        max_articles=args.max_articles
    )
    
    print(f"\nCorpus Statistics:")
    print(f"  Total documents: {len(corpus)}")
    total_chars = sum(len(text) for text in corpus.values())
    print(f"  Total characters: {total_chars:,}")
    if corpus:
        avg_length = total_chars // len(corpus)
        print(f"  Average length: {avg_length:,} characters")
    
    print(f"\nNext steps:")
    print(f"1. Build gold dataset:")
    print(f"   python -m ragbench.datasets.build_gold_large \\")
    print(f"     --corpus {output_path} \\")
    print(f"     --output data/gold/gold_wikipedia.jsonl \\")
    print(f"     --num-queries 500 \\")
    print(f"     --strategy diverse")
    print(f"\n2. Update config to use this corpus")
    print(f"\n3. Run evaluation")


if __name__ == "__main__":
    main()

