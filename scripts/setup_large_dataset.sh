#!/bin/bash
# Script to set up a large dataset for RAGBench evaluation

set -e

echo "=== RAGBench Large Dataset Setup ==="
echo ""

# Check if corpus file exists
if [ ! -f "data/raw/corpus.json" ]; then
    echo "Error: corpus.json not found. Please load a corpus first."
    echo ""
    echo "Options:"
    echo "1. Load from JSONL:"
    echo "   python -m ragbench.datasets.load_corpus --source jsonl --input your_data.jsonl --output data/raw/corpus.json"
    echo ""
    echo "2. Load from directory of text files:"
    echo "   python -m ragbench.datasets.load_corpus --source directory --input /path/to/docs --output data/raw/corpus.json"
    echo ""
    echo "3. Use your own corpus JSON file (doc_id -> text mapping)"
    exit 1
fi

# Get corpus size
CORPUS_SIZE=$(python3 -c "import json; print(len(json.load(open('data/raw/corpus.json'))))")
echo "Corpus size: $CORPUS_SIZE documents"

# Determine number of queries based on corpus size
if [ $CORPUS_SIZE -lt 100 ]; then
    NUM_QUERIES=50
elif [ $CORPUS_SIZE -lt 1000 ]; then
    NUM_QUERIES=200
elif [ $CORPUS_SIZE -lt 10000 ]; then
    NUM_QUERIES=500
else
    NUM_QUERIES=1000
fi

echo "Recommended number of queries: $NUM_QUERIES"
echo ""

# Build gold dataset
echo "Building gold evaluation dataset..."
python -m ragbench.datasets.build_gold_large \
    --corpus data/raw/corpus.json \
    --output data/gold/gold.jsonl \
    --num-queries $NUM_QUERIES \
    --strategy diverse

echo ""
echo "=== Setup Complete ==="
echo "You can now run evaluation with:"
echo "  python -m ragbench.eval --config configs/demo.yaml"

