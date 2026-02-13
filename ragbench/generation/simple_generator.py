"""Simple extractive generator for baseline evaluation."""

from typing import List
from ragbench.retrieval.protocols import RetrievedDocument, Generator


class SimpleGenerator(Generator):
    """Simple extractive generator that returns the most relevant document text."""
    
    def answer(self, query: str, context_docs: List[RetrievedDocument]) -> str:
        """
        Generate answer by extracting from context.
        
        Args:
            query: The user query
            context_docs: List of retrieved documents
            
        Returns:
            The answer text (currently just the top document)
        """
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        # For now, return the top document's text
        # In a full implementation, this would use an LLM
        return context_docs[0].text.strip()

