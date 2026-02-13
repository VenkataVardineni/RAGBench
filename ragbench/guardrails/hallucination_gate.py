"""Hallucination gate - runtime guardrail for RAG systems."""

from typing import List, Optional
from ragbench.retrieval.protocols import RetrievedDocument
from ragbench.metrics.faithfulness import FaithfulnessEvaluator


class HallucinationGate:
    """Runtime guardrail that blocks answers not supported by retrieved context."""
    
    def __init__(
        self,
        faithfulness_evaluator: FaithfulnessEvaluator,
        min_retrieval_score: float = 0.3,
        min_faithfulness_score: float = 0.6,
        min_retrieved_docs: int = 1
    ):
        """
        Initialize hallucination gate.
        
        Args:
            faithfulness_evaluator: FaithfulnessEvaluator instance
            min_retrieval_score: Minimum average retrieval score threshold
            min_faithfulness_score: Minimum faithfulness score threshold
            min_retrieved_docs: Minimum number of retrieved documents required
        """
        self.faithfulness_evaluator = faithfulness_evaluator
        self.min_retrieval_score = min_retrieval_score
        self.min_faithfulness_score = min_faithfulness_score
        self.min_retrieved_docs = min_retrieved_docs
    
    def should_block(
        self,
        answer: str,
        retrieved_docs: List[RetrievedDocument],
        query: Optional[str] = None
    ) -> tuple:
        """
        Determine if answer should be blocked.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            query: Original query (optional, for logging)
            
        Returns:
            Tuple of (should_block: bool, reason: str)
        """
        # Check minimum documents
        if len(retrieved_docs) < self.min_retrieved_docs:
            return True, f"Insufficient retrieved documents ({len(retrieved_docs)} < {self.min_retrieved_docs})"
        
        # Check retrieval confidence (average score)
        if retrieved_docs:
            avg_score = sum(doc.score for doc in retrieved_docs) / len(retrieved_docs)
            if avg_score < self.min_retrieval_score:
                return True, f"Low retrieval confidence ({avg_score:.3f} < {self.min_retrieval_score})"
        
        # Check faithfulness
        context_texts = [doc.text for doc in retrieved_docs]
        faithfulness_result = self.faithfulness_evaluator.evaluate(answer, context_texts)
        
        if faithfulness_result.faithfulness_score < self.min_faithfulness_score:
            return True, (
                f"Low faithfulness score ({faithfulness_result.faithfulness_score:.3f} < {self.min_faithfulness_score}). "
                f"Failure type: {faithfulness_result.failure_type}"
            )
        
        return False, "OK"
    
    def filter_answer(
        self,
        answer: str,
        retrieved_docs: List[RetrievedDocument],
        query: Optional[str] = None
    ) -> str:
        """
        Filter answer through hallucination gate.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            query: Original query
            
        Returns:
            Filtered answer (or rejection message if blocked)
        """
        should_block, reason = self.should_block(answer, retrieved_docs, query)
        
        if should_block:
            return (
                "I don't have enough grounded evidence in the retrieved documents to answer. "
                f"Reason: {reason}"
            )
        
        return answer

