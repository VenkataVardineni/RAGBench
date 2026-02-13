# RAGBench

**A complete evaluation harness for Retrieval-Augmented Generation systems**

RAGBench provides comprehensive evaluation tools to measure retrieval quality, generation faithfulness, and detect hallucinations in RAG pipelines. It helps developers identify where their RAG systems fail (bad retrieval vs hallucination) and includes an optional runtime guard that refuses to answer when it can't be grounded.

## Features

- **Retrieval Metrics**: MRR, nDCG@K, Precision@K, Recall@K
- **Generation Metrics**: Faithfulness scoring and hallucination rate estimation
- **Failure Analysis**: Categorizes failures into "retrieval miss", "context noise", and "hallucination"
- **Hallucination Gate**: Runtime guardrail that blocks ungrounded answers
- **Flexible Integration**: Plug in your own embeddings, vector store, and LLM

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VenkataVardineni/RAGBench.git
cd RAGBench

# Install using pip
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt

# Optional: Install FAISS for fast similarity search (recommended for large datasets)
pip install faiss-cpu  # or faiss on macOS

# Optional: Install spaCy for better entity extraction
pip install spacy
python -m spacy download en_core_web_sm

# Optional: Install sentence-transformers for embedding-based retrieval
pip install sentence-transformers
```

### Running Evaluation

#### With Sample Dataset

```bash
# Run evaluation with demo configuration
python -m ragbench.eval --config configs/demo.yaml
```

#### With Real Dataset (Recommended)

**Option 1: Download SQuAD Dataset (Best Results)**

```bash
# Download SQuAD dataset from HuggingFace
python scripts/download_huggingface_dataset.py \
    --dataset squad \
    --max-docs 500 \
    --output data/raw/squad_corpus.json

# Build gold dataset with actual SQuAD questions
python scripts/build_better_gold_from_squad.py

# Run evaluation with improved setup
python -m ragbench.eval --config configs/squad_better.yaml
```

**Option 2: Use Your Own Dataset**

```bash
# 1. Load your corpus (from JSONL, directory, or JSON)
python -m ragbench.datasets.load_corpus \
    --source jsonl \
    --input your_data.jsonl \
    --output data/raw/corpus.json

# 2. Build gold evaluation dataset
python -m ragbench.datasets.build_gold_large \
    --corpus data/raw/corpus.json \
    --output data/gold/gold.jsonl \
    --num-queries 500 \
    --strategy diverse

# 3. Run evaluation
python -m ragbench.eval --config configs/demo.yaml
```

See [docs/LARGE_DATASET_GUIDE.md](docs/LARGE_DATASET_GUIDE.md) for detailed instructions.

This will:
1. Load the gold evaluation dataset
2. Run retrieval and generation for each query
3. Compute retrieval and faithfulness metrics
4. Generate reports in `reports/<run_id>/`

### Viewing Results

```bash
# Generate and view report
python -m ragbench.eval.report --results-dir reports/<run_id>/
```

## Project Structure

```
RAGBench/
├── configs/              # Configuration files
├── data/                 # Datasets (raw, processed, gold)
├── ragbench/
│   ├── datasets/         # Dataset builders and loaders
│   ├── retrieval/       # Retrieval implementations (TF-IDF, embeddings)
│   ├── generation/       # Generation implementations
│   ├── metrics/          # Evaluation metrics
│   ├── eval/             # Evaluation runners and reporting
│   └── guardrails/       # Hallucination gate
└── reports/              # Evaluation results
```

## Plugging in Your RAG Pipeline

RAGBench uses protocol-based interfaces, making it easy to integrate your own components.

### Custom Retriever

Implement the `Retriever` protocol:

```python
from ragbench.retrieval.protocols import Retriever, RetrievedDocument

class MyRetriever(Retriever):
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        # Your retrieval logic
        return [RetrievedDocument(doc_id="...", text="...", score=0.9)]
```

### Custom Generator

Implement the `Generator` protocol:

```python
from ragbench.retrieval.protocols import Generator, RetrievedDocument

class MyGenerator(Generator):
    def answer(self, query: str, context_docs: List[RetrievedDocument]) -> str:
        # Your generation logic
        return "Generated answer..."
```

## Understanding Metrics

### Retrieval Metrics

- **Precision@K**: Fraction of top-K retrieved documents that are relevant
- **Recall@K**: Fraction of relevant documents found in top-K results
- **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant document
- **nDCG@K**: Normalized Discounted Cumulative Gain, considering ranking quality

### Generation Metrics

- **Faithfulness Score**: Measures how well the answer is supported by retrieved context (0-1)
- **Hallucination Rate**: Percentage of answer sentences not supported by context

### Failure Types

- **Retrieval Miss**: No relevant documents retrieved
- **Context Noise**: Too much irrelevant context retrieved
- **Hallucination**: Answer contains information not in retrieved context

## Hallucination Gate

The hallucination gate is a runtime guardrail that blocks answers when:

- Retrieval confidence is too low
- Faithfulness score is below threshold
- Insufficient documents retrieved

```python
from ragbench.guardrails.hallucination_gate import HallucinationGate
from ragbench.metrics.faithfulness import FaithfulnessEvaluator

gate = HallucinationGate(
    faithfulness_evaluator=FaithfulnessEvaluator(),
    min_retrieval_score=0.3,
    min_faithfulness_score=0.6
)

filtered_answer = gate.filter_answer(answer, retrieved_docs, query)
```

## Docker

Build and run with Docker:

```bash
# Build image
docker build -t ragbench .

# Run evaluation
docker run -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports ragbench
```

## Configuration

Edit `configs/demo.yaml` to customize:

- Retrieval method (TF-IDF or embeddings)
- Generation method
- Faithfulness thresholds
- Hallucination gate settings

## Example Results

With the improved setup (actual SQuAD questions + larger corpus), RAGBench achieves:

| Metric | Value | Description |
|--------|-------|-------------|
| **MRR** | **0.344** | Mean Reciprocal Rank - top result found in top 3 on average |
| **Precision@1** | **0.260** | 26% of top results are relevant |
| **Recall@10** | **0.532** | 53% of relevant docs found in top-10 |
| **nDCG@10** | **0.389** | Normalized Discounted Cumulative Gain |
| **Faithfulness** | **1.000** | 100% of answers grounded in context |
| **Hallucination Rate** | **0.000** | 0% unsupported content |

See [docs/IMPROVING_RESULTS.md](docs/IMPROVING_RESULTS.md) for tips on achieving even better results.

## Example Report

After running evaluation, you'll get:

- `retrieval_metrics.csv`: Detailed retrieval metrics per query
- `faithfulness.csv`: Faithfulness scores and hallucination rates
- `hallucinations.jsonl`: Examples of failures
- `traces.jsonl`: Complete per-query traces with all details
- `report.md`: Summary report with aggregate metrics and failure breakdown

## Dataset Download Options

RAGBench includes scripts to download real-world datasets:

```bash
# SQuAD (Question Answering) - Recommended for best results
python scripts/download_huggingface_dataset.py --dataset squad

# Wikipedia articles
python scripts/download_huggingface_dataset.py --dataset wikipedia

# News articles
python scripts/download_huggingface_dataset.py --dataset news
```

## Performance Tips

- **For large datasets (>10K docs)**: Use embedding retrieval with FAISS
- **For better results**: Use actual questions (like SQuAD) instead of generated queries
- **For faster evaluation**: Reduce number of queries or use smaller embedding models
- See [docs/IMPROVING_RESULTS.md](docs/IMPROVING_RESULTS.md) for detailed optimization guide

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT License

