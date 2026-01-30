"""
Data quality scoring for preference pairs.

Quality dimensions:
- Length ratio: Chosen should be meaningfully longer
- Distinctiveness: Chosen/rejected should be different enough
- Coherence: Both should be coherent responses
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class QualityScores:
    """Quality scores for a preference pair."""
    length_score: float  # 0-1, higher is better
    distinction_score: float  # 0-1, higher is better
    coherence_score: float  # 0-1, higher is better
    overall_score: float  # Weighted average
    
    @property
    def passes_threshold(self, threshold: float = 0.6) -> bool:
        """Check if overall score passes threshold."""
        return self.overall_score >= threshold


class DataQualityScorer:
    """
    Score preference pairs for quality.
    
    Filters out low-quality pairs that may hurt training.
    """
    
    def __init__(self, embedding_model: Optional[Any] = None):
        """
        Initialize quality scorer.
        
        Args:
            embedding_model: Optional embedding model for semantic similarity
        """
        self.embedding_model = embedding_model
    
    def score_pair(self, pair: Dict[str, str]) -> QualityScores:
        """
        Score a single preference pair.
        
        Args:
            pair: Dictionary with 'chosen' and 'rejected' keys
        
        Returns:
            QualityScores object
        """
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        # Length score: chosen should be at least 1.5x longer
        len_ratio = len(chosen) / max(len(rejected), 1)
        length_score = min(len_ratio / 1.5, 1.0)
        
        # Distinction score: should be meaningfully different
        if self.embedding_model:
            try:
                emb_c = self.embedding_model.encode(chosen)
                emb_r = self.embedding_model.encode(rejected)
                similarity = np.dot(emb_c, emb_r) / (
                    np.linalg.norm(emb_c) * np.linalg.norm(emb_r)
                )
                distinction_score = 1 - similarity
            except Exception:
                distinction_score = self._word_overlap_score(chosen, rejected)
        else:
            distinction_score = self._word_overlap_score(chosen, rejected)
        
        # Coherence: simple heuristics
        coherence_score = self._coherence_heuristics(chosen, rejected)
        
        # Weighted average
        overall = (
            0.3 * length_score +
            0.4 * distinction_score +
            0.3 * coherence_score
        )
        
        return QualityScores(
            length_score=length_score,
            distinction_score=distinction_score,
            coherence_score=coherence_score,
            overall_score=overall,
        )
    
    def filter_dataset(
        self,
        pairs: List[Dict[str, str]],
        min_score: float = 0.6
    ) -> List[Dict[str, str]]:
        """
        Filter dataset to high-quality pairs only.
        
        Args:
            pairs: List of preference pairs
            min_score: Minimum overall score threshold
        
        Returns:
            Filtered list of pairs
        """
        return [
            p for p in pairs
            if self.score_pair(p).overall_score >= min_score
        ]
    
    def _word_overlap_score(self, chosen: str, rejected: str) -> float:
        """Compute distinction score using word overlap."""
        words_c = set(chosen.lower().split())
        words_r = set(rejected.lower().split())
        if not words_c or not words_r:
            return 0.5  # Neutral if empty
        overlap = len(words_c & words_r) / len(words_c | words_r)
        return 1 - overlap
    
    def _coherence_heuristics(self, chosen: str, rejected: str) -> float:
        """Simple coherence heuristics."""
        # Check for minimum length
        if len(chosen) < 20 or len(rejected) < 20:
            return 0.5
        
        # Check for sentence structure (has periods)
        has_sentences_c = '.' in chosen or '!' in chosen or '?' in chosen
        has_sentences_r = '.' in rejected or '!' in rejected or '?' in rejected
        
        if has_sentences_c and has_sentences_r:
            return 1.0
        elif has_sentences_c or has_sentences_r:
            return 0.75
        else:
            return 0.5
