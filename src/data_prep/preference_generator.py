"""
PreferencePairGenerator - Generate preference pairs for DPO/RLHF training.

Creates rejected responses by stripping Bob's voice markers and truncating answers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import re
import logging

from .textbook_parser import Question

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A preference pair for DPO training."""

    prompt: str
    chosen: str
    rejected: str
    source: str = ""  # Where this pair came from (topic/chapter/etc.)


@dataclass
class VoiceStrippingConfig:
    """Configuration for voice marker stripping."""

    # Confidence markers to remove
    confidence_markers: List[str] = field(
        default_factory=lambda: [
            r"I've tracked",
            r"I've seen",
            r"I've observed",
            r"In my experience",
            r"Here's what I've",
            r"What I've found",
            r"I believe",
            r"I've learned",
            r"I've noticed",
            r"Over the years",
            r"In my years of",
            r"After years of",
            r"The data shows",
        ]
    )

    # Engagement markers to remove
    engagement_markers: List[str] = field(
        default_factory=lambda: [
            r"Look,",
            r"Okay, so",
            r"Here's the thing",
            r"Here's what you need to understand",
            r"Why does this matter\?",
            r"Now,? you might be thinking",
            r"Let me explain",
            r"Think about it this way",
            r"Here's the key",
            r"The key point is",
            r"What's important here is",
        ]
    )

    # Psychology emphasis markers
    psychology_markers: List[str] = field(
        default_factory=lambda: [
            r"market psychology",
            r"crowd behavior",
            r"herd mentality",
            r"emotional discipline",
        ]
    )

    # Truncation settings
    truncate_to_ratio: float = 0.4  # Keep this fraction of sentences
    min_sentences: int = 2  # Minimum sentences to keep


class PreferencePairGenerator:
    """Generate preference pairs from questions with reference answers."""

    def __init__(self, config: VoiceStrippingConfig = None):
        """
        Initialize the generator.

        Args:
            config: Configuration for voice stripping
        """
        self.config = config or VoiceStrippingConfig()

    def generate_pair(self, question: Question, source: str = "") -> PreferencePair:
        """
        Generate a preference pair from a question.

        Args:
            question: Question with reference answer (becomes 'chosen')
            source: Source identifier for the pair

        Returns:
            PreferencePair with chosen (original) and rejected (stripped) responses
        """
        chosen = question.reference_answer
        # Get source from question if available, otherwise use provided source
        topic_source = question.source if question.source else source
        
        # Ensure topic_source is not empty or "unknown"
        if not topic_source or topic_source == "unknown":
            logger.warning(
                f"Generating pair without proper topic source. "
                f"Question: {question.question[:50]}..."
            )
            # Try to extract from question context if possible
            if not topic_source:
                topic_source = source if source else "unknown"
        
        rejected = self._create_rejected_response(question.question, chosen, topic_source)

        return PreferencePair(
            prompt=question.question,
            chosen=chosen,
            rejected=rejected,
            source=topic_source,  # Properly set to topic ID
        )

    def generate_all(
        self, questions: List[Tuple[Question, str]]
    ) -> List[PreferencePair]:
        """
        Generate preference pairs from a list of questions.

        Args:
            questions: List of (Question, source) tuples

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        for question, source in questions:
            try:
                pair = self.generate_pair(question, source)
                pairs.append(pair)
            except Exception as e:
                logger.error(f"Failed to generate pair for {source}: {e}")

        return pairs

    def _create_rejected_response(self, question: str, reference: str, source: str = "") -> str:
        """
        Create a rejected response by stripping voice markers and truncating.

        The rejected response should be:
        1. Factually similar but missing Bob's voice
        2. Shorter and less complete
        3. More generic in tone

        Args:
            question: The question being answered
            reference: The reference (chosen) answer
            source: Source identifier (e.g., "episode-2026-01-24" for transcripts)
        """
        rejected = reference

        # Check if this is from a transcript (episode source)
        is_transcript = source.startswith("episode-")

        # Step 1: Remove confidence markers
        for marker in self.config.confidence_markers:
            rejected = re.sub(marker, "", rejected, flags=re.IGNORECASE)

        # Step 2: Remove engagement markers
        for marker in self.config.engagement_markers:
            rejected = re.sub(marker, "", rejected, flags=re.IGNORECASE)

        # Step 3: Remove bold formatting
        rejected = re.sub(r"\*\*([^*]+)\*\*", r"\1", rejected)

        # Step 4: Remove emphasis markers (italics)
        rejected = re.sub(r"\*([^*]+)\*", r"\1", rejected)

        # Step 5: Transcript-specific rejection strategies
        if is_transcript:
            rejected = self._apply_transcript_rejection_strategies(rejected)

        # Step 6: Clean up extra whitespace
        rejected = re.sub(r"\n\s*\n\s*\n", "\n\n", rejected)
        rejected = re.sub(r"  +", " ", rejected)
        rejected = re.sub(r"^\s+", "", rejected, flags=re.MULTILINE)

        # Step 7: Truncate to make it incomplete
        rejected = self._truncate_response(rejected)

        # Step 8: Make it more generic by removing specific insights
        rejected = self._genericize_response(rejected)

        # Clean up any remaining issues
        rejected = rejected.strip()

        # Ensure the rejected response is actually different
        if self._similarity_ratio(reference, rejected) > 0.95:
            # If too similar, force more changes
            rejected = self._force_generic(rejected)

        return rejected

    def _apply_transcript_rejection_strategies(self, text: str) -> str:
        """
        Apply transcript-specific rejection strategies.

        For transcripts, we want to reject responses that:
        1. Are overly formal (Bob speaks casually)
        2. Missing terminology (DCL, right-translated, Bressert bands)
        3. Generic advice instead of specific insights
        4. Missing time context (episode date references)
        """
        # Remove Bob-specific terminology
        terminology_patterns = [
            (r"\bDCL\b", "cycle low"),
            (r"\bright-translated\b", "bullish"),
            (r"\bleft-translated\b", "bearish"),
            (r"\bBressert bands?\b", "technical indicators"),
            (r"\b40-week low\b", "support level"),
            (r"\b4-year cycle\b", "market cycle"),
        ]

        for pattern, replacement in terminology_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove date references (time context)
        text = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", "", text)
        text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", text)

        # Make it more formal/academic (opposite of Bob's casual tone)
        text = re.sub(r"\bLook,\s*", "It should be noted that ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bOkay,?\s+so\b", "Therefore, ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bHere's the thing\b", "The key point is", text, flags=re.IGNORECASE)

        # Replace specific insights with generic platitudes
        generic_replacements = [
            (r"I've tracked this for (over )?\d+ years", "Historical analysis suggests"),
            (r"In my experience", "Generally speaking"),
            (r"What I've found", "Research indicates"),
        ]

        for pattern, replacement in generic_replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _truncate_response(self, text: str) -> str:
        """Truncate the response to a fraction of the original."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        if len(sentences) <= self.config.min_sentences:
            return text

        # Calculate how many sentences to keep
        keep_count = max(
            self.config.min_sentences,
            int(len(sentences) * self.config.truncate_to_ratio),
        )

        truncated = " ".join(sentences[:keep_count])

        # Ensure it ends properly
        if not truncated.endswith((".", "!", "?")):
            truncated += "."

        return truncated

    def _genericize_response(self, text: str) -> str:
        """Make the response more generic by removing specific insights."""
        # Replace first-person with third-person where possible
        text = re.sub(r"\bI\b(?!')", "One", text)
        text = re.sub(r"\bmy\b", "the", text, flags=re.IGNORECASE)
        text = re.sub(r"\bme\b", "one", text, flags=re.IGNORECASE)

        # Remove parenthetical asides (often contain personality)
        text = re.sub(r"\([^)]*\)", "", text)

        # Remove em-dashes with surrounding content (often asides)
        text = re.sub(r"—[^—]*—", "", text)

        return text.strip()

    def _force_generic(self, text: str) -> str:
        """Force the text to be more generic when other methods fail."""
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Keep only the most factual-looking sentences
        generic_sentences = []
        for sentence in sentences:
            # Skip sentences with strong voice markers
            if any(
                marker.lower() in sentence.lower()
                for marker in ["I've", "I believe", "I think", "Here's"]
            ):
                continue
            generic_sentences.append(sentence)

        if generic_sentences:
            return " ".join(generic_sentences[: self.config.min_sentences + 1])

        # Fallback: just take first two sentences
        return " ".join(sentences[: self.config.min_sentences])

    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Calculate a simple similarity ratio between two texts."""
        # Simple word overlap ratio
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def to_jsonl_format(self, pairs: List[PreferencePair]) -> List[Dict]:
        """
        Convert preference pairs to JSONL format for training.

        Returns format compatible with TRL DPOTrainer:
        {"prompt": ..., "chosen": ..., "rejected": ...}
        """
        return [
            {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            }
            for pair in pairs
        ]

    def to_sft_format(self, pairs: List[PreferencePair]) -> List[Dict]:
        """
        Convert pairs to SFT format (instruction/input/output).

        Uses the 'chosen' response as the target output.
        """
        return [
            {
                "instruction": pair.prompt,
                "input": "",
                "output": pair.chosen,
            }
            for pair in pairs
        ]
