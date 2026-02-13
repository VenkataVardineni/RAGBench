"""Retrieval evaluation metrics."""

from typing import List, Set
import numpy as np


def precision_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    """
    Compute Precision@K.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ordered by relevance)
        gold_doc_ids: Set of gold (relevant) document IDs
        k: Cutoff value
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in gold_doc_ids)
    return relevant_count / k


def recall_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    """
    Compute Recall@K.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        gold_doc_ids: Set of gold document IDs
        k: Cutoff value
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(gold_doc_ids) == 0:
        return 0.0
    
    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in gold_doc_ids)
    return relevant_count / len(gold_doc_ids)


def mean_reciprocal_rank(retrieved_doc_ids: List[str], gold_doc_ids: Set[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        gold_doc_ids: Set of gold document IDs
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (nDCG@K).
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        gold_doc_ids: Set of gold document IDs
        k: Cutoff value
        
    Returns:
        nDCG@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved_doc_ids[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        if doc_id in gold_doc_ids:
            # Relevance is 1 if relevant, 0 otherwise
            relevance = 1.0
            dcg += relevance / np.log2(i + 1)
    
    # Compute IDCG (ideal DCG)
    num_relevant = min(len(gold_doc_ids), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_retrieval_metrics(retrieved_doc_ids: List[str], gold_doc_ids: Set[str], k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """
    Compute all retrieval metrics.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs
        gold_doc_ids: Set of gold document IDs
        k_values: List of K values to compute metrics for
        
    Returns:
        Dictionary of metric scores
    """
    metrics = {}
    
    # Precision@K
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_doc_ids, gold_doc_ids, k)
    
    # Recall@K
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, gold_doc_ids, k)
    
    # MRR
    metrics["mrr"] = mean_reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
    
    # nDCG@K
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, gold_doc_ids, k)
    
    return metrics

