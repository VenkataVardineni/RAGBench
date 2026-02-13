"""Embedding-based retriever using Sentence Transformers."""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except (ImportError, ValueError, Exception):
    # Handle various import errors including dependency conflicts
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from ragbench.retrieval.protocols import Retriever, RetrievedDocument


class EmbeddingRetriever(Retriever):
    """Embedding-based retriever using Sentence Transformers and FAISS."""
    
    def __init__(self, corpus_path: Path, model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = True):
        """
        Initialize embedding retriever.
        
        Args:
            corpus_path: Path to corpus JSON file
            model_name: Sentence Transformers model name
            use_faiss: Whether to use FAISS for fast search (falls back to numpy if unavailable)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.corpus_path = corpus_path
        self.model_name = model_name
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        self.corpus: Dict[str, str] = {}
        self.model = SentenceTransformer(model_name)
        self.doc_ids: List[str] = []
        self.doc_embeddings = None
        self.index = None
        
        self._load_corpus()
        self._build_index()
    
    def _load_corpus(self):
        """Load corpus from JSON file."""
        with open(self.corpus_path, "r") as f:
            self.corpus = json.load(f)
    
    def _build_index(self):
        """Build embedding index."""
        self.doc_ids = list(self.corpus.keys())
        texts = [self.corpus[doc_id] for doc_id in self.doc_ids]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if self.use_faiss:
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            self.doc_embeddings = None  # Don't store separately when using FAISS
        else:
            # Store embeddings for numpy-based search
            self.doc_embeddings = embeddings
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
            self.doc_embeddings = self.doc_embeddings / (norms + 1e-8)
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """
        Search for relevant documents using embeddings.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievedDocument objects sorted by relevance
        """
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        if self.use_faiss:
            # Normalize query
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.doc_ids):
                    # Convert L2 distance to similarity (1 - normalized distance)
                    similarity = 1.0 - min(dist / 2.0, 1.0)  # Approximate conversion
                    results.append(RetrievedDocument(
                        doc_id=self.doc_ids[idx],
                        text=self.corpus[self.doc_ids[idx]],
                        score=float(similarity)
                    ))
        else:
            # Numpy-based search
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            # Compute cosine similarities
            similarities = np.dot(self.doc_embeddings, query_embedding)
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append(RetrievedDocument(
                    doc_id=self.doc_ids[idx],
                    text=self.corpus[self.doc_ids[idx]],
                    score=float(similarities[idx])
                ))
        
        return results

