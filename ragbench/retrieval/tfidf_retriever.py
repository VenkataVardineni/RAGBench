"""TF-IDF based retriever implementation."""

import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import math

from ragbench.retrieval.protocols import Retriever, RetrievedDocument


class TFIDFRetriever(Retriever):
    """TF-IDF based retriever."""
    
    def __init__(self, corpus_path: Path):
        """
        Initialize TF-IDF retriever.
        
        Args:
            corpus_path: Path to corpus JSON file (doc_id -> text mapping)
        """
        self.corpus_path = corpus_path
        self.corpus: Dict[str, str] = {}
        self.doc_freq: Dict[str, int] = {}  # term -> number of docs containing it
        self.doc_vectors: Dict[str, Dict[str, float]] = {}  # doc_id -> term -> tf-idf score
        self.vocab: set = set()
        
        self._load_corpus()
        self._build_index()
    
    def _load_corpus(self):
        """Load corpus from JSON file."""
        with open(self.corpus_path, "r") as f:
            self.corpus = json.load(f)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _build_index(self):
        """Build TF-IDF index."""
        # Count document frequency for each term
        doc_terms: Dict[str, set] = {}
        for doc_id, text in self.corpus.items():
            terms = set(self._tokenize(text))
            doc_terms[doc_id] = terms
            self.vocab.update(terms)
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
        
        num_docs = len(self.corpus)
        
        # Build TF-IDF vectors for each document
        for doc_id, text in self.corpus.items():
            terms = self._tokenize(text)
            term_counts = Counter(terms)
            doc_length = len(terms)
            
            tfidf_vector = {}
            for term, count in term_counts.items():
                tf = count / doc_length if doc_length > 0 else 0
                idf = math.log(num_docs / self.doc_freq[term]) if self.doc_freq[term] > 0 else 0
                tfidf_vector[term] = tf * idf
            
            self.doc_vectors[doc_id] = tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Get all terms
        terms = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """
        Search for relevant documents using TF-IDF.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievedDocument objects sorted by relevance
        """
        # Build query vector
        query_terms = self._tokenize(query)
        query_counts = Counter(query_terms)
        query_length = len(query_terms)
        
        query_vector = {}
        num_docs = len(self.corpus)
        for term, count in query_counts.items():
            if term in self.vocab:
                tf = count / query_length if query_length > 0 else 0
                idf = math.log(num_docs / self.doc_freq[term]) if term in self.doc_freq and self.doc_freq[term] > 0 else 0
                query_vector[term] = tf * idf
        
        # Compute similarity with each document
        scores = []
        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scores.append((doc_id, similarity))
        
        # Sort by score (descending) and take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            results.append(RetrievedDocument(
                doc_id=doc_id,
                text=self.corpus[doc_id],
                score=score
            ))
        
        return results

