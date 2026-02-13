# Large Dataset Setup Guide

This guide shows you how to set up RAGBench with large, real-world datasets.

## Quick Start

### Option 1: Use Your Own JSONL Dataset

If you have a JSONL file where each line is a document:

```bash
# Load corpus from JSONL
python -m ragbench.datasets.load_corpus \
    --source jsonl \
    --input your_data.jsonl \
    --output data/raw/corpus.json \
    --id-field id \
    --text-field text

# Build gold evaluation dataset
python -m ragbench.datasets.build_gold_large \
    --corpus data/raw/corpus.json \
    --output data/gold/gold.jsonl \
    --num-queries 500 \
    --strategy diverse

# Run evaluation
python -m ragbench.eval --config configs/demo.yaml
```

### Option 2: Load from Directory of Text Files

If you have a directory of text files (.txt, .md, etc.):

```bash
# Load and chunk documents
python -m ragbench.datasets.load_corpus \
    --source directory \
    --input /path/to/your/documents \
    --output data/raw/corpus.json \
    --chunk-size 500 \
    --chunk-overlap 50

# Build gold dataset
python -m ragbench.datasets.build_gold_large \
    --corpus data/raw/corpus.json \
    --output data/gold/gold.jsonl \
    --num-queries 500

# Run evaluation
python -m ragbench.eval --config configs/demo.yaml
```

### Option 3: Use Existing JSON Corpus

If you already have a corpus in JSON format (doc_id -> text mapping):

```bash
# Just build the gold dataset
python -m ragbench.datasets.build_gold_large \
    --corpus your_corpus.json \
    --output data/gold/gold.jsonl \
    --num-queries 1000 \
    --strategy diverse
```

## Dataset Sources

### Wikipedia

1. Download a Wikipedia dump or use preprocessed data
2. Convert to JSONL format (one article per line)
3. Load using the JSONL loader

### News Articles

Many news datasets are available in JSONL format. Example:

```python
# Example JSONL format:
# {"id": "article_001", "title": "...", "text": "...", "date": "..."}
python -m ragbench.datasets.load_corpus \
    --source jsonl \
    --input news_articles.jsonl \
    --id-field id \
    --text-field text
```

### Technical Documentation

For documentation sites:

```bash
# Scrape or export documentation to text files
python -m ragbench.datasets.load_corpus \
    --source directory \
    --input docs/ \
    --chunk-size 1000 \
    --chunk-overlap 100
```

### Custom Dataset

Create your own corpus JSON file:

```python
import json

corpus = {
    "doc_001": "Your document text here...",
    "doc_002": "Another document...",
    # ... more documents
}

with open("data/raw/corpus.json", "w") as f:
    json.dump(corpus, f, indent=2)
```

## Gold Dataset Strategies

The `build_gold_large` script supports three strategies:

- **random_sampling**: Randomly sample documents (fast, good for large datasets)
- **diverse**: Ensure good coverage across documents (recommended)
- **difficult**: Focus on longer documents and multi-doc queries (for challenging evaluation)

## Performance Tips

### For Very Large Datasets (>10K documents)

1. **Use embedding retriever with FAISS**:
   ```yaml
   # configs/demo.yaml
   retrieval:
     type: "embedding"
     use_faiss: true
   ```

2. **Reduce evaluation queries**:
   ```bash
   python -m ragbench.datasets.build_gold_large --num-queries 200
   ```

3. **Use smaller chunk sizes** for faster processing

### Memory Considerations

- TF-IDF retriever: Good for <10K documents
- Embedding retriever with FAISS: Scales to millions of documents
- Consider using GPU for embedding generation if available

## Example: Wikipedia Dataset

```bash
# 1. Download and preprocess Wikipedia (using wikiextractor)
# This creates JSON files

# 2. Convert to single JSONL
cat wiki_*/AA/wiki_* | jq -c '{id: .id, text: .text}' > wiki_corpus.jsonl

# 3. Load into RAGBench
python -m ragbench.datasets.load_corpus \
    --source jsonl \
    --input wiki_corpus.jsonl \
    --output data/raw/corpus.json

# 4. Build evaluation set
python -m ragbench.datasets.build_gold_large \
    --corpus data/raw/corpus.json \
    --num-queries 1000 \
    --strategy diverse

# 5. Run evaluation
python -m ragbench.eval --config configs/demo.yaml
```

## Troubleshooting

**Out of memory errors:**
- Use FAISS for large datasets
- Reduce chunk size
- Process in batches

**Slow evaluation:**
- Use TF-IDF for smaller datasets (<5K docs)
- Reduce number of evaluation queries
- Use smaller embedding models

**Low retrieval scores:**
- Check if your corpus format is correct
- Verify document IDs match between corpus and gold dataset
- Try different retrieval methods (TF-IDF vs embeddings)

