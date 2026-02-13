"""Protocols (interfaces) for retrieval and generation components."""

from typing import Protocol, List, Tuple
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata."""
    doc_id: str
    text: str
    score: float


class Retriever(Protocol):
    """Protocol for retrieval components."""
    
    def search(self, query: str, top_k: int = 10) -> List[RetrievedDocument]:
        """
        Search for relevant documents given a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievedDocument objects, sorted by relevance (highest score first)
        """
        ...


class Generator(Protocol):
    """Protocol for generation components."""
    
    def answer(self, query: str, context_docs: List[RetrievedDocument]) -> str:
        """
        Generate an answer given a query and retrieved context documents.
        
        Args:
            query: The user query
            context_docs: List of retrieved documents to use as context
            
        Returns:
            The generated answer text
        """
        ...

