"""
VoiceMarkers - Bob Loukas's characteristic voice patterns.

Defines the regex patterns used to detect Bob's distinctive communication style
in both evaluation and preference pair generation.
"""

from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class VoiceMarkers:
    """Bob's characteristic voice markers for evaluation."""

    # Confidence markers - Bob's experience-based assertions
    confidence_markers: List[str] = field(
        default_factory=lambda: [
            r"I've tracked",
            r"I've seen",
            r"I've observed",
            r"I've watched",
            r"I've been watching",
            r"I've noticed",
            r"I've learned",
            r"I've found",
            r"In my experience",
            r"In my years of",
            r"After years of",
            r"Over the years",
            r"What I've found",
            r"Here's what I've",
            r"Here's what I know",
            r"I believe",
            r"I can tell you",
            r"The data shows",
            r"History shows",
        ]
    )

    # Engagement markers - How Bob draws in the audience
    engagement_markers: List[str] = field(
        default_factory=lambda: [
            r"Look,",
            r"Okay, so",
            r"Here's the thing",
            r"Here's the key",
            r"Here's what you need to understand",
            r"Here's why",
            r"Why does this matter\?",
            r"Think about it",
            r"Think about it this way",
            r"Now,? you might be thinking",
            r"Let me explain",
            r"Let me be clear",
            r"The key point is",
            r"What's important here is",
            r"The bottom line is",
            r"Pay attention to this",
        ]
    )

    # Psychology markers - Bob's focus on market psychology
    psychology_markers: List[str] = field(
        default_factory=lambda: [
            r"market psychology",
            r"crowd behavior",
            r"crowd psychology",
            r"herd mentality",
            r"fear\b",
            r"greed\b",
            r"euphoria",
            r"capitulation",
            r"panic",
            r"complacency",
            r"denial",
            r"hope\b",
            r"despair",
            r"emotional",
            r"sentiment",
            r"psychology",
        ]
    )

    # Cycle terminology - Bob's specific vocabulary
    cycle_terminology: List[str] = field(
        default_factory=lambda: [
            r"4-year cycle",
            r"four-year cycle",
            r"40-week cycle",
            r"60-day cycle",
            r"cycle low",
            r"cycle high",
            r"cycle top",
            r"cycle bottom",
            r"accumulation",
            r"distribution",
            r"markup",
            r"markdown",
            r"cycle phase",
            r"bull market",
            r"bear market",
            r"nested cycles?",
        ]
    )

    # Negative patterns - Things Bob would NOT say
    negative_patterns: List[str] = field(
        default_factory=lambda: [
            r"halving causes",
            r"halving drives",
            r"halving is responsible",
            r"because of.*halving",
            r"halving leads to",
            r"exactly 4 years",
            r"exactly four years",
            r"guaranteed",
            r"always works",
            r"never fails",
            r"100% certain",
            r"can't lose",
            r"risk.?free",
            r"get rich quick",
            r"financial advice",
            r"I am not a financial advisor",  # Disclaimer that breaks voice
        ]
    )

    def count_markers(self, text: str, marker_list: List[str]) -> int:
        """Count how many markers from a list are found in text."""
        count = 0
        for pattern in marker_list:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count

    def find_markers(self, text: str, marker_list: List[str]) -> List[str]:
        """Find all markers from a list that appear in text."""
        found = []
        for pattern in marker_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found.append(match.group())
        return found

    def check_negative_patterns(self, text: str) -> List[str]:
        """Find any negative patterns that should NOT appear."""
        violations = []
        for pattern in self.negative_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                violations.append(match.group())
        return violations

    def compute_marker_scores(self, text: str) -> dict:
        """
        Compute normalized scores for each marker category.

        Returns dict with scores 0.0-1.0 for each category.
        """
        confidence_found = self.count_markers(text, self.confidence_markers)
        engagement_found = self.count_markers(text, self.engagement_markers)
        psychology_found = self.count_markers(text, self.psychology_markers)
        terminology_found = self.count_markers(text, self.cycle_terminology)

        return {
            "confidence": min(1.0, confidence_found / 3),  # Expect ~3 markers
            "engagement": min(1.0, engagement_found / 2),  # Expect ~2 markers
            "psychology": min(1.0, psychology_found / 3),  # Expect ~3 markers
            "terminology": min(1.0, terminology_found / 3),  # Expect ~3 markers
        }


# Singleton instance for convenience
DEFAULT_VOICE_MARKERS = VoiceMarkers()
