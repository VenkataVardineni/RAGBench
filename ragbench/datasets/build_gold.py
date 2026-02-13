"""Build a gold evaluation dataset from a corpus."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import hashlib


def create_sample_corpus(output_dir: Path) -> Dict[str, str]:
    """
    Create a sample corpus from project documentation.
    Returns a dictionary mapping doc_id to text.
    """
    corpus = {}
    
    # Sample documents about RAG systems
    docs = [
        {
            "id": "doc_001",
            "text": """
            Retrieval-Augmented Generation (RAG) is a technique that enhances language model responses 
            by retrieving relevant information from a knowledge base before generating answers. 
            RAG combines the power of dense retrieval with generative language models to produce 
            more accurate and grounded responses.
            """
        },
        {
            "id": "doc_002",
            "text": """
            Vector embeddings are numerical representations of text that capture semantic meaning. 
            Popular embedding models include Sentence-BERT, OpenAI's text-embedding-ada-002, and 
            Cohere embeddings. These embeddings enable semantic search by computing cosine similarity 
            between query and document vectors.
            """
        },
        {
            "id": "doc_003",
            "text": """
            FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and 
            clustering of dense vectors. It supports both CPU and GPU implementations and can handle 
            billions of vectors. FAISS is commonly used in RAG systems for fast retrieval.
            """
        },
        {
            "id": "doc_004",
            "text": """
            Evaluation metrics for retrieval systems include Precision@K, Recall@K, Mean Reciprocal Rank (MRR), 
            and Normalized Discounted Cumulative Gain (nDCG). Precision@K measures the fraction of retrieved 
            documents that are relevant. MRR measures the reciprocal rank of the first relevant document.
            """
        },
        {
            "id": "doc_005",
            "text": """
            Hallucination in language models refers to generating information that is not supported by 
            the source context. Faithfulness metrics measure how well generated answers are grounded in 
            retrieved documents. Techniques include sentence-level similarity checks and named entity 
            consistency validation.
            """
        },
        {
            "id": "doc_006",
            "text": """
            TF-IDF (Term Frequency-Inverse Document Frequency) is a traditional information retrieval 
            method that scores documents based on term frequency and inverse document frequency. 
            While less sophisticated than embeddings, TF-IDF can serve as a baseline retriever 
            for RAG systems.
            """
        },
        {
            "id": "doc_007",
            "text": """
            Chunking strategies are crucial for RAG systems. Documents are typically split into 
            smaller chunks (e.g., 256-512 tokens) with overlap to preserve context. Common approaches 
            include fixed-size chunking, sentence-based chunking, and semantic chunking.
            """
        },
        {
            "id": "doc_008",
            "text": """
            RAGBench is an evaluation harness for RAG systems that measures retrieval quality, 
            generation faithfulness, and provides hallucination detection. It supports multiple 
            retrieval backends and generation models, enabling comprehensive evaluation of 
            RAG pipelines.
            """
        }
    ]
    
    for doc in docs:
        corpus[doc["id"]] = doc["text"].strip()
    
    # Save corpus
    corpus_file = output_dir / "corpus.json"
    with open(corpus_file, "w") as f:
        json.dump(corpus, f, indent=2)
    
    return corpus


def build_gold_dataset(corpus: Dict[str, str], output_file: Path, num_queries: int = 100):
    """
    Build a gold evaluation dataset with queries, gold document IDs, and expected answers.
    
    Args:
        corpus: Dictionary mapping doc_id to text
        output_file: Path to save the gold dataset (JSONL format)
        num_queries: Number of queries to generate
    """
    gold_items = []
    
    # Create query-answer pairs based on the corpus
    qa_pairs = [
        {
            "query": "What is Retrieval-Augmented Generation?",
            "gold_doc_ids": ["doc_001"],
            "expected_answer": "Retrieval-Augmented Generation (RAG) is a technique that enhances language model responses by retrieving relevant information from a knowledge base before generating answers."
        },
        {
            "query": "How does RAG work?",
            "gold_doc_ids": ["doc_001"],
            "expected_answer": "RAG combines the power of dense retrieval with generative language models to produce more accurate and grounded responses."
        },
        {
            "query": "What are vector embeddings?",
            "gold_doc_ids": ["doc_002"],
            "expected_answer": "Vector embeddings are numerical representations of text that capture semantic meaning."
        },
        {
            "query": "Name some popular embedding models",
            "gold_doc_ids": ["doc_002"],
            "expected_answer": "Popular embedding models include Sentence-BERT, OpenAI's text-embedding-ada-002, and Cohere embeddings."
        },
        {
            "query": "What is FAISS used for?",
            "gold_doc_ids": ["doc_003"],
            "expected_answer": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors."
        },
        {
            "query": "What evaluation metrics are used for retrieval?",
            "gold_doc_ids": ["doc_004"],
            "expected_answer": "Evaluation metrics for retrieval systems include Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG)."
        },
        {
            "query": "What is Precision@K?",
            "gold_doc_ids": ["doc_004"],
            "expected_answer": "Precision@K measures the fraction of retrieved documents that are relevant."
        },
        {
            "query": "What is MRR?",
            "gold_doc_ids": ["doc_004"],
            "expected_answer": "MRR measures the reciprocal rank of the first relevant document."
        },
        {
            "query": "What is hallucination in language models?",
            "gold_doc_ids": ["doc_005"],
            "expected_answer": "Hallucination in language models refers to generating information that is not supported by the source context."
        },
        {
            "query": "How do you measure faithfulness?",
            "gold_doc_ids": ["doc_005"],
            "expected_answer": "Faithfulness metrics measure how well generated answers are grounded in retrieved documents using techniques like sentence-level similarity checks and named entity consistency validation."
        },
        {
            "query": "What is TF-IDF?",
            "gold_doc_ids": ["doc_006"],
            "expected_answer": "TF-IDF (Term Frequency-Inverse Document Frequency) is a traditional information retrieval method that scores documents based on term frequency and inverse document frequency."
        },
        {
            "query": "How are documents chunked for RAG?",
            "gold_doc_ids": ["doc_007"],
            "expected_answer": "Documents are typically split into smaller chunks (e.g., 256-512 tokens) with overlap to preserve context using approaches like fixed-size chunking, sentence-based chunking, or semantic chunking."
        },
        {
            "query": "What is RAGBench?",
            "gold_doc_ids": ["doc_008"],
            "expected_answer": "RAGBench is an evaluation harness for RAG systems that measures retrieval quality, generation faithfulness, and provides hallucination detection."
        },
        {
            "query": "What does RAG combine?",
            "gold_doc_ids": ["doc_001"],
            "expected_answer": "RAG combines the power of dense retrieval with generative language models."
        },
        {
            "query": "What enables semantic search?",
            "gold_doc_ids": ["doc_002"],
            "expected_answer": "Embeddings enable semantic search by computing cosine similarity between query and document vectors."
        }
    ]
    
    # Expand to reach num_queries by creating variations
    expanded_pairs = qa_pairs.copy()
    while len(expanded_pairs) < num_queries:
        # Create variations by rephrasing
        for pair in qa_pairs:
            if len(expanded_pairs) >= num_queries:
                break
            # Add slight variations
            variations = [
                {
                    "query": f"Tell me about {pair['query'].lower()}",
                    "gold_doc_ids": pair["gold_doc_ids"],
                    "expected_answer": pair["expected_answer"]
                },
                {
                    "query": f"Explain {pair['query'].lower()}",
                    "gold_doc_ids": pair["gold_doc_ids"],
                    "expected_answer": pair["expected_answer"]
                }
            ]
            for var in variations:
                if len(expanded_pairs) >= num_queries:
                    break
                if var not in expanded_pairs:
                    expanded_pairs.append(var)
    
    # Create gold items
    for idx, pair in enumerate(expanded_pairs[:num_queries]):
        item = {
            "id": f"query_{idx+1:03d}",
            "query": pair["query"],
            "gold_doc_ids": pair["gold_doc_ids"],
            "expected_answer": pair["expected_answer"]
        }
        gold_items.append(item)
    
    # Save as JSONL
    with open(output_file, "w") as f:
        for item in gold_items:
            f.write(json.dumps(item) + "\n")
    
    print(f"Created gold dataset with {len(gold_items)} queries: {output_file}")
    return gold_items


def main():
    """Main function to build the gold dataset."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    gold_dir = data_dir / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create corpus
    corpus = create_sample_corpus(data_dir / "raw")
    
    # Build gold dataset
    gold_file = gold_dir / "gold.jsonl"
    build_gold_dataset(corpus, gold_file, num_queries=120)
    
    print(f"Gold dataset saved to: {gold_file}")


if __name__ == "__main__":
    main()

