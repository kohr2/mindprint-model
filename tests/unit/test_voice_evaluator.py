"""
Tests for VoiceFidelityEvaluator - evaluating voice fidelity of model responses.

Tests cover:
- Semantic similarity computation
- Voice marker analysis
- Critical distinction checking
- Negative pattern detection
- Overall score calculation
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.evaluation.voice_evaluator import (
    VoiceFidelityEvaluator,
    VoiceEvaluationResult,
    EvaluationWeights,
)
from src.evaluation.voice_markers import VoiceMarkers


@pytest.fixture
def mock_embedder():
    """Create a mock sentence transformer."""
    with patch("src.evaluation.voice_evaluator.SentenceTransformer") as mock:
        embedder = MagicMock()
        # Return simple embeddings that give predictable similarities
        embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock.return_value = embedder
        yield embedder


@pytest.fixture
def evaluator(mock_embedder) -> VoiceFidelityEvaluator:
    """Create a VoiceFidelityEvaluator with mocked embedder."""
    return VoiceFidelityEvaluator()


class TestVoiceEvaluationResult:
    """Test VoiceEvaluationResult dataclass."""

    def test_passed_property_when_above_threshold(self) -> None:
        """Result passes when score >= 0.75 and no violations."""
        result = VoiceEvaluationResult(
            overall_score=0.80,
            semantic_similarity=0.85,
            voice_marker_score=0.75,
            confidence_score=0.70,
            psychology_score=0.80,
            terminology_score=0.75,
            critical_distinctions_passed=True,
            negative_patterns_avoided=True,
        )

        assert result.passed is True

    def test_failed_when_below_threshold(self) -> None:
        """Result fails when score < 0.75."""
        result = VoiceEvaluationResult(
            overall_score=0.70,
            semantic_similarity=0.65,
            voice_marker_score=0.60,
            confidence_score=0.50,
            psychology_score=0.60,
            terminology_score=0.55,
            critical_distinctions_passed=True,
            negative_patterns_avoided=True,
        )

        assert result.passed is False

    def test_failed_when_critical_distinction_fails(self) -> None:
        """Result fails when critical distinctions fail."""
        result = VoiceEvaluationResult(
            overall_score=0.85,
            semantic_similarity=0.90,
            voice_marker_score=0.80,
            confidence_score=0.80,
            psychology_score=0.80,
            terminology_score=0.80,
            critical_distinctions_passed=False,  # FAIL
            negative_patterns_avoided=True,
        )

        assert result.passed is False

    def test_failed_when_negative_patterns_found(self) -> None:
        """Result fails when negative patterns are found."""
        result = VoiceEvaluationResult(
            overall_score=0.85,
            semantic_similarity=0.90,
            voice_marker_score=0.80,
            confidence_score=0.80,
            psychology_score=0.80,
            terminology_score=0.80,
            critical_distinctions_passed=True,
            negative_patterns_avoided=False,  # FAIL
        )

        assert result.passed is False

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict includes all relevant fields."""
        result = VoiceEvaluationResult(
            overall_score=0.80,
            semantic_similarity=0.85,
            voice_marker_score=0.75,
            confidence_score=0.70,
            psychology_score=0.80,
            terminology_score=0.75,
            critical_distinctions_passed=True,
            negative_patterns_avoided=True,
            violations=["test violation"],
        )

        result_dict = result.to_dict()

        assert "overall_score" in result_dict
        assert "semantic_similarity" in result_dict
        assert "voice_marker_score" in result_dict
        assert "passed" in result_dict
        assert result_dict["violations"] == ["test violation"]


class TestEvaluationWeights:
    """Test EvaluationWeights configuration."""

    def test_default_weights(self) -> None:
        """Default weights are sensible."""
        weights = EvaluationWeights()

        assert weights.semantic_similarity == 0.50
        assert weights.voice_markers == 0.30
        assert weights.critical_distinctions == 0.10
        assert weights.negative_patterns == 0.10

    def test_weights_sum_to_one(self) -> None:
        """Main weights sum to 1.0."""
        weights = EvaluationWeights()

        total = (
            weights.semantic_similarity
            + weights.voice_markers
            + weights.critical_distinctions
            + weights.negative_patterns
        )

        assert abs(total - 1.0) < 0.001

    def test_voice_subweights_sum_to_one(self) -> None:
        """Voice marker sub-weights sum to 1.0."""
        weights = EvaluationWeights()

        total = (
            weights.confidence_weight
            + weights.psychology_weight
            + weights.terminology_weight
        )

        assert abs(total - 1.0) < 0.001


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def test_identical_texts_high_similarity(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Identical texts have high similarity."""
        # Configure mock to return same embeddings for identical texts
        evaluator.embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        result = evaluator._compute_semantic_similarity(
            ["The market moves in cycles."],
            ["The market moves in cycles."],
        )

        assert result == 1.0

    def test_empty_lists_return_zero(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Empty input lists return zero similarity."""
        result = evaluator.evaluate([], [])

        assert result.semantic_similarity == 0.0


class TestCriticalDistinctions:
    """Test critical distinction checking (halving vs cycle)."""

    def test_detects_halving_causes_violation(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Detects 'halving causes' as a violation."""
        answers = ["The halving causes the bull market to start."]

        passed = evaluator._check_critical_distinctions(answers)

        assert passed is False

    def test_detects_halving_drives_violation(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Detects 'halving drives' as a violation."""
        answers = ["The halving drives the 4-year cycle."]

        passed = evaluator._check_critical_distinctions(answers)

        assert passed is False

    def test_correct_statement_passes(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Correct statement about halving passes."""
        answers = [
            "The 4-year cycle is NOT caused by the halving. "
            "It's driven by market psychology."
        ]

        passed = evaluator._check_critical_distinctions(answers)

        assert passed is True

    def test_halving_mention_without_causation_passes(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Mentioning halving without causation claim passes."""
        answers = ["The halving coincides with the cycle but doesn't cause it."]

        passed = evaluator._check_critical_distinctions(answers)

        assert passed is True


class TestNegativePatterns:
    """Test negative pattern detection."""

    def test_detects_guaranteed_claim(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Detects 'guaranteed' as a negative pattern."""
        answers = ["This strategy is guaranteed to work."]

        violations = evaluator._check_negative_patterns(answers)

        assert len(violations) > 0

    def test_clean_text_no_violations(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Clean Bob-style text has no violations."""
        answers = [
            "I've tracked this pattern for years. "
            "The cycle is driven by market psychology."
        ]

        violations = evaluator._check_negative_patterns(answers)

        assert len(violations) == 0


class TestVoiceMarkerAnalysis:
    """Test voice marker analysis."""

    def test_analyzes_confidence_markers(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Analyzes confidence markers in text."""
        answers = ["I've tracked this pattern. I've seen it play out."]

        results = evaluator._analyze_voice_markers(answers)

        assert "confidence" in results
        assert results["confidence"] > 0

    def test_analyzes_psychology_markers(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Analyzes psychology markers in text."""
        answers = ["Market psychology drives fear and greed cycles."]

        results = evaluator._analyze_voice_markers(answers)

        assert "psychology" in results
        assert results["psychology"] > 0

    def test_found_markers_returned(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Specific found markers are returned."""
        answers = ["I've tracked the 4-year cycle."]

        results = evaluator._analyze_voice_markers(answers)

        assert "found" in results
        assert "confidence" in results["found"]


class TestFullEvaluation:
    """Test full evaluation flow."""

    def test_evaluate_returns_result(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Evaluate returns a VoiceEvaluationResult."""
        result = evaluator.evaluate(
            ["I've tracked market psychology."],
            ["I've tracked market psychology."],
        )

        assert isinstance(result, VoiceEvaluationResult)

    def test_evaluate_single_wrapper(self, evaluator: VoiceFidelityEvaluator) -> None:
        """evaluate_single is a wrapper for single items."""
        result = evaluator.evaluate_single(
            "Test answer.",
            "Test reference.",
        )

        assert isinstance(result, VoiceEvaluationResult)

    def test_high_quality_answer_scores_well(self, evaluator: VoiceFidelityEvaluator) -> None:
        """High-quality Bob-style answer scores well on markers."""
        # Configure mock for high similarity
        evaluator.embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        generated = [
            "Look, I've tracked this pattern for years. "
            "The 4-year cycle is driven by market psychology - "
            "fear, greed, capitulation, and euphoria. "
            "This is NOT caused by the halving."
        ]
        reference = [
            "The 4-year cycle is driven by market psychology."
        ]

        result = evaluator.evaluate(generated, reference)

        # Should have decent voice marker scores
        assert result.voice_marker_score > 0.3
        assert result.critical_distinctions_passed is True

    def test_generic_answer_scores_poorly(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Generic answer scores poorly on voice markers."""
        evaluator.embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        generated = ["Bitcoin goes up and down in price over time."]
        reference = ["The 4-year cycle is driven by market psychology."]

        result = evaluator.evaluate(generated, reference)

        # Should have low voice marker scores
        assert result.voice_marker_score < 0.5


class TestOverallScoreCalculation:
    """Test overall score calculation."""

    def test_score_combines_components(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Overall score combines all components according to weights."""
        evaluator.embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        result = evaluator.evaluate(
            ["I've tracked the 4-year cycle driven by psychology."],
            ["I've tracked the 4-year cycle driven by psychology."],
        )

        # Score should be positive and bounded
        assert 0.0 <= result.overall_score <= 1.0

    def test_critical_failure_reduces_score(self, evaluator: VoiceFidelityEvaluator) -> None:
        """Critical distinction failure reduces overall score."""
        evaluator.embedder.encode.return_value = np.array([[1.0, 0.0, 0.0]])

        # Answer with halving causation claim
        result = evaluator.evaluate(
            ["The halving causes the cycle."],
            ["The cycle is not caused by halving."],
        )

        # Critical failure should reduce score
        assert result.critical_distinctions_passed is False
        # Score should be lower than if it passed
        assert result.overall_score < 1.0
