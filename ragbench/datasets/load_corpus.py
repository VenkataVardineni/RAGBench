"""Load corpus from various sources for large-scale evaluation."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def load_corpus_from_json(corpus_file: Path) -> Dict[str, str]:
    """Load corpus from JSON file (doc_id -> text mapping)."""
    with open(corpus_file, "r") as f:
        return json.load(f)


def load_corpus_from_jsonl(jsonl_file: Path, id_field: str = "id", text_field: str = "text") -> Dict[str, str]:
    """
    Load corpus from JSONL file where each line is a document.
    
    Args:
        jsonl_file: Path to JSONL file
        id_field: Field name for document ID
        text_field: Field name for document text
        
    Returns:
        Dictionary mapping doc_id to text
    """
    corpus = {}
    with open(jsonl_file, "r") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get(id_field)
            text = doc.get(text_field)
            if doc_id and text:
                corpus[doc_id] = text
    return corpus


def load_corpus_from_directory(
    directory: Path,
    file_extensions: List[str] = [".txt", ".md"],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, str]:
    """
    Load corpus from directory of text files.
    
    Args:
        directory: Directory containing text files
        file_extensions: List of file extensions to include
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dictionary mapping doc_id to text chunk
    """
    corpus = {}
    chunk_id = 0
    
    for file_path in directory.rglob("*"):
        if file_path.suffix.lower() in file_extensions and file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split into chunks
                start = 0
                while start < len(content):
                    end = start + chunk_size
                    chunk = content[start:end]
                    
                    if chunk.strip():
                        doc_id = f"doc_{chunk_id:06d}"
                        corpus[doc_id] = chunk.strip()
                        chunk_id += 1
                    
                    start = end - chunk_overlap
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    return corpus


def load_corpus_from_wikipedia_dump(dump_file: Path) -> Dict[str, str]:
    """
    Load corpus from Wikipedia XML dump (requires wikiextractor preprocessing).
    For now, this is a placeholder - you'd need to preprocess Wikipedia dumps first.
    """
    # This would require parsing Wikipedia XML dumps
    # For now, suggest using preprocessed JSON/JSONL format
    raise NotImplementedError(
        "Wikipedia dump loading requires preprocessing. "
        "Use wikiextractor to convert to JSON/JSONL first, then use load_corpus_from_jsonl"
    )


def save_corpus(corpus: Dict[str, str], output_file: Path):
    """Save corpus to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved {len(corpus)} documents to {output_file}")


def main():
    """CLI for loading corpus from various sources."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load corpus for RAGBench")
    parser.add_argument("--source", type=str, required=True, 
                       choices=["json", "jsonl", "directory", "wikipedia"],
                       help="Source type")
    parser.add_argument("--input", type=str, required=True,
                       help="Input path (file or directory)")
    parser.add_argument("--output", type=str, default="data/raw/corpus.json",
                       help="Output corpus JSON file")
    parser.add_argument("--id-field", type=str, default="id",
                       help="ID field name (for JSONL)")
    parser.add_argument("--text-field", type=str, default="text",
                       help="Text field name (for JSONL)")
    parser.add_argument("--chunk-size", type=int, default=500,
                       help="Chunk size for directory loading")
    parser.add_argument("--chunk-overlap", type=int, default=50,
                       help="Chunk overlap for directory loading")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.source == "json":
        corpus = load_corpus_from_json(input_path)
    elif args.source == "jsonl":
        corpus = load_corpus_from_jsonl(input_path, args.id_field, args.text_field)
    elif args.source == "directory":
        corpus = load_corpus_from_directory(
            input_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    elif args.source == "wikipedia":
        corpus = load_corpus_from_wikipedia_dump(input_path)
    
    save_corpus(corpus, output_path)
    print(f"\nCorpus statistics:")
    print(f"  Total documents: {len(corpus)}")
    total_chars = sum(len(text) for text in corpus.values())
    print(f"  Total characters: {total_chars:,}")
    print(f"  Average document length: {total_chars // len(corpus) if corpus else 0:,} characters")


if __name__ == "__main__":
    main()

