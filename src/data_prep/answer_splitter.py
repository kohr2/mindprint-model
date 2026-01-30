"""
AnswerSplitter - Split overly long answers into focused segments.

Creates multiple question-answer pairs from a single long answer,
ensuring each segment is 600-1200 characters and preserves Bob's voice.
"""

from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)


class AnswerSplitter:
    """Split long answers into focused segments with sub-questions."""

    def __init__(self, target_min: int = 600, target_max: int = 1200):
        """
        Initialize the splitter.

        Args:
            target_min: Minimum characters per segment (default: 600)
            target_max: Maximum characters per segment (default: 1200)
        """
        self.target_min = target_min
        self.target_max = target_max

    def split_answer(
        self, question: str, answer: str, topic: str = ""
    ) -> List[Dict]:
        """
        Split a long answer into focused segments with sub-questions.

        If answer <= target_max, returns single item.
        If answer > target_max, splits into multiple segments, each with
        a focused sub-question.

        Args:
            question: Original question
            answer: Answer to split (may be very long)
            topic: Topic identifier for context

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        if len(answer) <= self.target_max:
            return [{"question": question, "answer": answer}]

        # Split into logical sections
        segments = self._split_into_segments(answer)

        # Create sub-questions for each segment
        result = []
        for i, segment in enumerate(segments, 1):
            sub_question = self._create_sub_question(question, segment, i, len(segments))
            result.append({"question": sub_question, "answer": segment})

        return result

    def _split_into_segments(self, answer: str) -> List[str]:
        """
        Split answer into logical segments.

        Tries to split by paragraphs first, then by sentences if needed.

        Args:
            answer: Answer text to split

        Returns:
            List of answer segments, each 600-1200 chars
        """
        # First, try splitting by double newlines (paragraphs)
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]

        if not paragraphs:
            # Fallback: split by single newlines
            paragraphs = [p.strip() for p in answer.split("\n") if p.strip()]

        # Check if we have a single very long paragraph that needs splitting
        if len(paragraphs) == 1 and len(paragraphs[0]) > self.target_max:
            # Single long paragraph, try splitting by sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0])
            
            # If no sentence boundaries found, split by character count
            if len(sentences) == 1:
                # No natural boundaries, split by character count
                paragraphs = []
                start = 0
                while start < len(answer):
                    end = min(start + self.target_max, len(answer))
                    # Try to break at word boundary
                    if end < len(answer):
                        # Look for last space before end
                        last_space = answer.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space
                    paragraphs.append(answer[start:end].strip())
                    start = end
            else:
                # Group sentences into paragraphs
                paragraphs = []
                current = []
                current_len = 0

                for sentence in sentences:
                    sent_len = len(sentence)
                    if current_len + sent_len > self.target_max and current:
                        paragraphs.append(" ".join(current))
                        current = [sentence]
                        current_len = sent_len
                    else:
                        current.append(sentence)
                        current_len += sent_len + 1

                if current:
                    paragraphs.append(" ".join(current))
        elif not paragraphs:
            # Empty answer, return empty list
            return []

        # Group paragraphs into segments of appropriate length
        segments = []
        current_segment = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # If adding this para would exceed max, finalize current segment
            if current_len + para_len > self.target_max and current_segment:
                segment_text = "\n\n".join(current_segment)
                if len(segment_text) >= self.target_min:
                    segments.append(segment_text)
                    current_segment = [para]
                    current_len = para_len
                else:
                    # Current segment too short, add para anyway
                    current_segment.append(para)
                    current_len += para_len + 2
            else:
                current_segment.append(para)
                current_len += para_len + 2  # +2 for \n\n

        # Add final segment
        if current_segment:
            segment_text = "\n\n".join(current_segment)
            segments.append(segment_text)

        # Ensure segments meet minimum length
        final_segments = []
        for segment in segments:
            if len(segment) >= self.target_min:
                final_segments.append(segment)
            elif final_segments:
                # Merge with previous segment if too short
                final_segments[-1] += "\n\n" + segment
            else:
                # First segment too short, keep it anyway
                final_segments.append(segment)

        return final_segments

    def _create_sub_question(
        self, original_question: str, segment: str, segment_num: int, total_segments: int
    ) -> str:
        """
        Create a focused sub-question for a segment.

        Args:
            original_question: Original question
            segment: Answer segment
            segment_num: Current segment number (1-indexed)
            total_segments: Total number of segments

        Returns:
            Focused sub-question
        """
        # Extract key topic from segment (first sentence or key phrase)
        first_sentence = segment.split(".")[0] if "." in segment else segment[:100]

        # Create focused question
        if total_segments > 1:
            # Multi-part answer
            focus_phrases = [
                f"Regarding {original_question.lower().rstrip('?')}, can you explain",
                f"Specifically, {original_question.lower().rstrip('?')}",
                f"In more detail, {original_question.lower().rstrip('?')}",
            ]

            # Try to extract what this segment focuses on
            if any(word in first_sentence.lower() for word in ["psychology", "sentiment", "emotion"]):
                focus = "the psychological aspects"
            elif any(word in first_sentence.lower() for word in ["cycle", "phase", "period"]):
                focus = "the cycle dynamics"
            elif any(word in first_sentence.lower() for word in ["pattern", "indicator", "signal"]):
                focus = "the specific patterns and indicators"
            else:
                focus = "this aspect"

            return f"{focus_phrases[0]} {focus}?"
        else:
            # Single segment, return original question
            return original_question
