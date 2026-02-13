"""Faithfulness and hallucination detection metrics."""

import re
from typing import List, Dict, Set
from dataclasses import dataclass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    faithfulness_score: float  # 0.0 to 1.0
    hallucination_rate: float  # Percentage of unsupported sentences
    unsupported_sentences: List[str]
    failure_type: str  # "retrieval_miss", "context_noise", "hallucination", "none"


class FaithfulnessEvaluator:
    """Evaluates faithfulness of generated answers to retrieved context."""
    
    def __init__(self, similarity_threshold: float = 0.5, use_spacy: bool = True):
        """
        Initialize faithfulness evaluator.
        
        Args:
            similarity_threshold: Minimum similarity score for a sentence to be considered supported
            use_spacy: Whether to use spaCy for entity extraction (falls back to regex if unavailable)
        """
        self.similarity_threshold = similarity_threshold
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.similarity_model = None
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, fall back to regex
                self.use_spacy = False
                self.nlp = None
        else:
            self.nlp = None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        entities = set()
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.add(ent.text.lower())
        else:
            # Fallback: extract capitalized words/phrases (simple heuristic)
            # Match capitalized words and phrases
            patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases
                r'\b[A-Z]{2,}\b',  # Acronyms
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities.update(m.lower() for m in matches)
        
        return entities
    
    def _sentence_similarity(self, sentence1: str, sentence2: str) -> float:
        """Compute similarity between two sentences."""
        if self.similarity_model:
            embeddings = self.similarity_model.encode([sentence1, sentence2])
            # Cosine similarity
            dot_product = embeddings[0] @ embeddings[1]
            norm1 = (embeddings[0] @ embeddings[0]) ** 0.5
            norm2 = (embeddings[1] @ embeddings[1]) ** 0.5
            return dot_product / (norm1 * norm2 + 1e-8)
        else:
            # Fallback: simple word overlap
            words1 = set(sentence1.lower().split())
            words2 = set(sentence2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0
    
    def _check_sentence_support(self, sentence: str, context_sentences: List[str]) -> bool:
        """Check if a sentence is supported by context."""
        if not context_sentences:
            return False
        
        # Check similarity to any context sentence
        max_similarity = max(
            self._sentence_similarity(sentence, ctx_sent)
            for ctx_sent in context_sentences
        )
        
        return max_similarity >= self.similarity_threshold
    
    def _check_entity_consistency(self, answer_entities: Set[str], context_entities: Set[str]) -> bool:
        """Check if answer entities appear in context."""
        if not answer_entities:
            return True  # No entities to check
        
        # Check if significant entities are in context
        # Allow some entities to be missing (common words, etc.)
        significant_entities = {e for e in answer_entities if len(e) > 3}  # Filter short words
        if not significant_entities:
            return True
        
        overlap = significant_entities & context_entities
        overlap_ratio = len(overlap) / len(significant_entities) if significant_entities else 1.0
        
        # Require at least 50% of significant entities to be in context
        return overlap_ratio >= 0.5
    
    def evaluate(self, answer: str, context_docs: List[str]) -> FaithfulnessResult:
        """
        Evaluate faithfulness of answer to context.
        
        Args:
            answer: Generated answer text
            context_docs: List of context document texts
            
        Returns:
            FaithfulnessResult with scores and failure analysis
        """
        if not answer:
            return FaithfulnessResult(
                faithfulness_score=0.0,
                hallucination_rate=1.0,
                unsupported_sentences=[],
                failure_type="none"
            )
        
        # Split into sentences
        answer_sentences = self._split_sentences(answer)
        if not answer_sentences:
            return FaithfulnessResult(
                faithfulness_score=0.0,
                hallucination_rate=1.0,
                unsupported_sentences=[],
                failure_type="none"
            )
        
        # Combine context into sentences
        context_sentences = []
        for doc in context_docs:
            context_sentences.extend(self._split_sentences(doc))
        
        # Check each answer sentence
        unsupported = []
        for sentence in answer_sentences:
            if not self._check_sentence_support(sentence, context_sentences):
                unsupported.append(sentence)
        
        # Check entity consistency
        answer_entities = self._extract_entities(answer)
        context_entities = set()
        for doc in context_docs:
            context_entities.update(self._extract_entities(doc))
        
        entity_consistent = self._check_entity_consistency(answer_entities, context_entities)
        
        # Calculate scores
        total_sentences = len(answer_sentences)
        supported_sentences = total_sentences - len(unsupported)
        faithfulness_score = supported_sentences / total_sentences if total_sentences > 0 else 0.0
        
        # Adjust score based on entity consistency
        if not entity_consistent:
            faithfulness_score *= 0.7  # Penalize entity inconsistencies
        
        hallucination_rate = len(unsupported) / total_sentences if total_sentences > 0 else 1.0
        
        # Determine failure type
        if not context_docs:
            failure_type = "retrieval_miss"
        elif len(unsupported) / total_sentences > 0.5:
            failure_type = "hallucination"
        elif len(context_sentences) > 100:  # Heuristic: too much context
            failure_type = "context_noise"
        else:
            failure_type = "none"
        
        return FaithfulnessResult(
            faithfulness_score=faithfulness_score,
            hallucination_rate=hallucination_rate,
            unsupported_sentences=unsupported,
            failure_type=failure_type
        )

