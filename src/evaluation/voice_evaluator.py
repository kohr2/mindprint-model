"""
VoiceFidelityEvaluator - Evaluate voice fidelity of model responses.

Combines semantic similarity with voice marker analysis to measure
how well a response matches Bob Loukas's distinctive style.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .voice_markers import VoiceMarkers, DEFAULT_VOICE_MARKERS

logger = logging.getLogger(__name__)


@dataclass
class VoiceEvaluationResult:
    """Result of voice fidelity evaluation."""

    # Overall composite score (0.0 - 1.0)
    overall_score: float

    # Component scores
    semantic_similarity: float
    voice_marker_score: float
    confidence_score: float
    psychology_score: float
    terminology_score: float

    # Critical checks (boolean)
    critical_distinctions_passed: bool
    negative_patterns_avoided: bool

    # Details
    violations: List[str] = field(default_factory=list)
    markers_found: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if the evaluation passed all criteria."""
        return (
            self.overall_score >= 0.75
            and self.critical_distinctions_passed
            and self.negative_patterns_avoided
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "semantic_similarity": self.semantic_similarity,
            "voice_marker_score": self.voice_marker_score,
            "confidence_score": self.confidence_score,
            "psychology_score": self.psychology_score,
            "terminology_score": self.terminology_score,
            "critical_distinctions_passed": self.critical_distinctions_passed,
            "negative_patterns_avoided": self.negative_patterns_avoided,
            "violations": self.violations,
            "passed": self.passed,
        }


@dataclass
class EvaluationWeights:
    """Weights for combining evaluation components."""

    semantic_similarity: float = 0.50  # Core factual alignment
    voice_markers: float = 0.30  # Bob's communication style
    critical_distinctions: float = 0.10  # Must not conflate halving/cycle
    negative_patterns: float = 0.10  # Must not contain bad patterns

    # Voice marker sub-weights (must sum to 1.0)
    confidence_weight: float = 0.30
    psychology_weight: float = 0.35
    terminology_weight: float = 0.35


class VoiceFidelityEvaluator:
    """Evaluates voice fidelity of model responses against Bob's style."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        voice_markers: Optional[VoiceMarkers] = None,
        weights: Optional[EvaluationWeights] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            embedding_model: Sentence transformer model for semantic similarity
            voice_markers: VoiceMarkers instance (uses default if None)
            weights: Evaluation weights (uses defaults if None)
        """
        self.embedder = SentenceTransformer(embedding_model)
        self.voice_markers = voice_markers or DEFAULT_VOICE_MARKERS
        self.weights = weights or EvaluationWeights()

    def evaluate(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
    ) -> VoiceEvaluationResult:
        """
        Evaluate voice fidelity of generated answers.

        Args:
            generated_answers: List of model-generated answers
            reference_answers: List of reference answers (Bob's voice)

        Returns:
            VoiceEvaluationResult with scores and details
        """
        if not generated_answers or not reference_answers:
            return VoiceEvaluationResult(
                overall_score=0.0,
                semantic_similarity=0.0,
                voice_marker_score=0.0,
                confidence_score=0.0,
                psychology_score=0.0,
                terminology_score=0.0,
                critical_distinctions_passed=True,
                negative_patterns_avoided=True,
            )

        # 1. Compute semantic similarity
        semantic_sim = self._compute_semantic_similarity(
            generated_answers, reference_answers
        )

        # 2. Analyze voice markers
        marker_results = self._analyze_voice_markers(generated_answers)

        # 3. Check critical distinctions (halving vs cycle)
        critical_passed = self._check_critical_distinctions(generated_answers)

        # 4. Check for negative patterns
        violations = self._check_negative_patterns(generated_answers)
        negative_avoided = len(violations) == 0

        # 5. Compute voice marker composite score
        voice_marker_score = (
            marker_results["confidence"] * self.weights.confidence_weight
            + marker_results["psychology"] * self.weights.psychology_weight
            + marker_results["terminology"] * self.weights.terminology_weight
        )

        # 6. Compute overall score
        overall_score = (
            semantic_sim * self.weights.semantic_similarity
            + voice_marker_score * self.weights.voice_markers
            + (1.0 if critical_passed else 0.0) * self.weights.critical_distinctions
            + (1.0 if negative_avoided else 0.0) * self.weights.negative_patterns
        )

        return VoiceEvaluationResult(
            overall_score=overall_score,
            semantic_similarity=semantic_sim,
            voice_marker_score=voice_marker_score,
            confidence_score=marker_results["confidence"],
            psychology_score=marker_results["psychology"],
            terminology_score=marker_results["terminology"],
            critical_distinctions_passed=critical_passed,
            negative_patterns_avoided=negative_avoided,
            violations=violations,
            markers_found=marker_results.get("found", {}),
        )

    def evaluate_single(
        self, generated: str, reference: str
    ) -> VoiceEvaluationResult:
        """
        Evaluate a single generated answer against its reference.

        Args:
            generated: Single model-generated answer
            reference: Single reference answer

        Returns:
            VoiceEvaluationResult for this pair
        """
        return self.evaluate([generated], [reference])

    def _compute_semantic_similarity(
        self, generated: List[str], reference: List[str]
    ) -> float:
        """Compute semantic similarity using embeddings."""
        try:
            gen_embeddings = self.embedder.encode(generated)
            ref_embeddings = self.embedder.encode(reference)

            # Compute pairwise similarities along diagonal
            similarities = cosine_similarity(gen_embeddings, ref_embeddings)

            # If same length, compare element-wise
            if len(generated) == len(reference):
                return float(np.diag(similarities).mean())
            else:
                # Otherwise, use mean of all similarities
                return float(similarities.mean())

        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0

    def _analyze_voice_markers(self, answers: List[str]) -> Dict:
        """Analyze presence of voice markers across all answers."""
        combined_text = " ".join(answers)

        # Compute marker scores
        scores = self.voice_markers.compute_marker_scores(combined_text)

        # Also find specific markers for debugging
        found = {
            "confidence": self.voice_markers.find_markers(
                combined_text, self.voice_markers.confidence_markers
            ),
            "psychology": self.voice_markers.find_markers(
                combined_text, self.voice_markers.psychology_markers
            ),
            "terminology": self.voice_markers.find_markers(
                combined_text, self.voice_markers.cycle_terminology
            ),
        }

        return {
            "confidence": scores["confidence"],
            "psychology": scores["psychology"],
            "terminology": scores["terminology"],
            "found": found,
        }

    def _check_critical_distinctions(self, answers: List[str]) -> bool:
        """
        Check that halving/cycle distinction is correct.

        Returns False if any answer incorrectly claims halving causes cycle.
        """
        combined_text = " ".join(answers)

        # Patterns indicating incorrect causation
        incorrect_patterns = [
            r"halving causes",
            r"halving drives",
            r"halving is responsible for",
            r"because of.*halving",
            r"halving leads to.*cycle",
            r"halving creates.*cycle",
        ]

        for pattern in incorrect_patterns:
            import re

            if re.search(pattern, combined_text, re.IGNORECASE):
                logger.warning(f"Critical distinction violation: {pattern}")
                return False

        return True

    def _check_negative_patterns(self, answers: List[str]) -> List[str]:
        """Find any negative patterns that should not appear."""
        combined_text = " ".join(answers)
        return self.voice_markers.check_negative_patterns(combined_text)


class QuizEvaluator:
    """Combined quiz + voice evaluator for model assessment."""

    def __init__(
        self,
        model,
        tokenizer,
        voice_evaluator: Optional[VoiceFidelityEvaluator] = None,
    ):
        """
        Initialize the quiz evaluator.

        Args:
            model: The language model to evaluate
            tokenizer: Tokenizer for the model
            voice_evaluator: VoiceFidelityEvaluator instance (creates default if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.voice_evaluator = voice_evaluator or VoiceFidelityEvaluator()

    def evaluate(
        self, questions: List[Dict], threshold: float = 0.75
    ) -> Dict:
        """
        Evaluate a quiz with both accuracy and voice fidelity.

        Args:
            questions: List of question dicts with 'question' and 'reference_answer'
            threshold: Pass/fail threshold for combined score

        Returns:
            Dict with accuracy, voice scores, and pass/fail status
        """
        prompts = [q["question"] for q in questions]
        references = [q["reference_answer"] for q in questions]

        # Generate answers
        generated = self._generate_answers(prompts)

        # Voice fidelity evaluation
        voice_result = self.voice_evaluator.evaluate(generated, references)

        return {
            "accuracy": voice_result.semantic_similarity,
            "voice_score": voice_result.voice_marker_score,
            "combined_score": voice_result.overall_score,
            "passed": voice_result.overall_score >= threshold,
            "voice_details": voice_result.to_dict(),
            "generated_answers": generated,
        }

    def _generate_answers(self, questions: List[str]) -> List[str]:
        """Generate model answers for questions."""
        answers = []

        # Check if model is a ModelInterface (backend mode)
        is_backend_mode = hasattr(self.model, 'get_underlying_model')

        for question in questions:
            prompt = self._format_prompt(question)

            if is_backend_mode:
                # Backend mode - use ModelInterface generate
                # Tokenize to get input IDs
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )
                input_ids = inputs["input_ids"]

                # Call ModelInterface generate (returns text for MLX, tokens for PyTorch)
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=512,
                    temperature=0.1,
                )

                # Check if output is string (MLX) or tensor (PyTorch)
                if isinstance(output, str):
                    # MLX returns string directly - mlx_lm.generate returns full text including prompt
                    # Extract only the generated part after the prompt
                    if prompt.strip() in output:
                        # Find where prompt ends and extract from there
                        prompt_end_idx = output.find(prompt.strip()) + len(prompt.strip())
                        answer = output[prompt_end_idx:].strip()
                    else:
                        # If prompt not found, assume entire output is the answer
                        answer = output.strip()
                    
                    # Log if answer is suspiciously short or empty
                    if len(answer) < 10:
                        logger.warning(f"Very short generated answer ({len(answer)} chars): '{answer[:100]}'")
                else:
                    # PyTorch returns tokens
                    answer = self.tokenizer.decode(
                        output[0][input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )
            else:
                # Legacy PyTorch mode
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                ).to(self.model.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Decode only the generated part
                answer = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

            answers.append(answer.strip())

        return answers

    def _format_prompt(self, question: str) -> str:
        """Format question as model prompt using tokenizer's chat template."""
        # Try to use tokenizer's chat template if available (Qwen, Llama, etc.)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                messages = [{"role": "user", "content": question}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, falling back to default format")
        
        # Fallback to Gemma-3 format for models without chat template
        return f"""<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
