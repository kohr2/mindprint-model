"""
TranscriptProcessor - Process Bob Loukas video transcripts into training data.

Converts raw transcript files into structured Question objects for training.
Can optionally use episode summaries from mindprint-agent for context.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
import re
from datetime import datetime

from .textbook_parser import Question

logger = logging.getLogger(__name__)


@dataclass
class EpisodeSummary:
    """Episode summary structure (from mindprint-agent)."""

    episode_date: str
    thesis: str
    risks: str
    opportunities: str
    topics: List[str] = field(default_factory=list)
    cycle_phase: Optional[str] = None
    key_indicators: Optional[str] = None


class TranscriptProcessor:
    """Process transcripts into training examples."""

    def __init__(self, transcripts_dir: str, summaries_dir: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            transcripts_dir: Directory containing raw transcript .txt files
            summaries_dir: Optional directory containing summaries.json file (from mindprint-agent)
                          Path should point to directory containing summaries.json, e.g.:
                          data/bob-loukas/brain-areas/bob-loukas-bitcoinlive-episodes/references/
        """
        self.transcripts_dir = Path(transcripts_dir)
        self.summaries_dir = Path(summaries_dir) if summaries_dir else None
        self._summaries_cache: Optional[Dict[str, Dict]] = None

        if not self.transcripts_dir.exists():
            raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    def parse_transcript(self, file_path: Path) -> str:
        """
        Parse a transcript file and return raw text.

        Args:
            file_path: Path to transcript .txt file

        Returns:
            Raw transcript text
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            # Basic cleanup
            content = content.strip()
            return content
        except Exception as e:
            logger.error(f"Failed to parse transcript {file_path}: {e}")
            return ""

    def _load_summaries_cache(self) -> Dict[str, Dict]:
        """
        Load and cache summaries.json file.

        Returns:
            Dictionary mapping date strings to summary data
        """
        if self._summaries_cache is not None:
            return self._summaries_cache

        if not self.summaries_dir:
            self._summaries_cache = {}
            return {}

        summaries_path = self.summaries_dir / "summaries.json"
        if not summaries_path.exists():
            logger.debug(f"Summaries file not found: {summaries_path}")
            self._summaries_cache = {}
            return {}

        try:
            with open(summaries_path, "r", encoding="utf-8") as f:
                self._summaries_cache = json.load(f)
            logger.info(f"Loaded {len(self._summaries_cache)} episode summaries from {summaries_path}")
            return self._summaries_cache
        except Exception as e:
            logger.warning(f"Failed to load summaries: {e}")
            self._summaries_cache = {}
            return {}

    def load_episode_summary(self, date: str) -> Optional[EpisodeSummary]:
        """
        Load episode summary for a given date.

        Args:
            date: Episode date in YYYY-MM-DD format

        Returns:
            EpisodeSummary if found, None otherwise
        """
        summaries = self._load_summaries_cache()
        data = summaries.get(date)

        if not data:
            logger.debug(f"Summary not found for {date}")
            return None

        try:
            # Handle both new nested format and legacy format
            return EpisodeSummary(
                episode_date=date,
                thesis=data.get("thesis", ""),
                risks=data.get("risks", ""),
                opportunities=data.get("opportunities", ""),
                topics=data.get("topics", []),
                cycle_phase=data.get("cyclePhase"),
                key_indicators=data.get("keyIndicators"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse summary for {date}: {e}")
            return None

    def generate_qa_pairs(
        self,
        transcript: str,
        date: str,
        summary: Optional[EpisodeSummary] = None,
    ) -> List[Question]:
        """
        Generate question-answer pairs from a transcript.

        This is a basic implementation that creates questions from the summary.
        More sophisticated generation is handled by TranscriptQuestionGenerator.

        Args:
            transcript: Raw transcript text
            date: Episode date in YYYY-MM-DD format
            summary: Optional episode summary for context

        Returns:
            List of Question objects
        """
        questions = []

        if summary:
            # Generate summary-based questions
            if summary.thesis:
                questions.append(Question(
                    question=f"What was Bob's market thesis on {date}?",
                    reference_answer=self._extract_answer_from_transcript(
                        transcript, summary.thesis
                    ),
                    question_type="open",
                    key_concepts=["thesis", "market_outlook"],
                ))

            if summary.risks:
                questions.append(Question(
                    question=f"What risks did Bob identify on {date}?",
                    reference_answer=self._extract_answer_from_transcript(
                        transcript, summary.risks
                    ),
                    question_type="open",
                    key_concepts=["risks", "risk_management"],
                ))

            if summary.opportunities:
                questions.append(Question(
                    question=f"What opportunities was Bob watching on {date}?",
                    reference_answer=self._extract_answer_from_transcript(
                        transcript, summary.opportunities
                    ),
                    question_type="open",
                    key_concepts=["opportunities", "market_setup"],
                ))

            # Topic-based questions
            for topic in summary.topics[:5]:  # Limit to 5 topics
                questions.append(Question(
                    question=f"Explain Bob's view on {topic} from {date}",
                    reference_answer=self._extract_topic_answer(
                        transcript, topic
                    ),
                    question_type="open",
                    key_concepts=[topic.lower().replace(" ", "_")],
                ))

        # If no summary, create basic questions from transcript
        if not questions:
            # Extract key segments and create questions
            segments = self._extract_key_segments(transcript)
            for i, segment in enumerate(segments[:3]):  # Limit to 3 segments
                questions.append(Question(
                    question=f"What did Bob discuss about market cycles on {date}?",
                    reference_answer=segment,
                    question_type="open",
                    key_concepts=["cycles", "market_analysis"],
                ))

        return questions

    def _extract_answer_from_transcript(
        self, transcript: str, summary_text: str
    ) -> str:
        """
        Extract relevant answer from transcript based on summary.

        This is a simple implementation - in practice, you might use
        semantic search or LLM to find the best matching segment.

        Args:
            transcript: Full transcript text
            summary_text: Summary text to match against

        Returns:
            Extracted answer text
        """
        # Simple keyword matching - find sentences containing summary keywords
        summary_keywords = set(
            word.lower()
            for word in re.findall(r"\b\w+\b", summary_text)
            if len(word) > 4  # Skip short words
        )

        sentences = re.split(r"[.!?]+", transcript)
        matching_sentences = []

        for sentence in sentences:
            sentence_words = set(
                word.lower() for word in re.findall(r"\b\w+\b", sentence)
            )
            overlap = len(summary_keywords & sentence_words)
            if overlap > 0:
                matching_sentences.append((overlap, sentence.strip()))

        # Sort by overlap and take top sentences
        matching_sentences.sort(reverse=True, key=lambda x: x[0])
        answer = " ".join(s[1] for s in matching_sentences[:3])

        # Fallback to summary if no good match
        if not answer or len(answer) < 50:
            answer = summary_text

        return answer.strip()

    def _extract_topic_answer(self, transcript: str, topic: str) -> str:
        """Extract answer about a specific topic from transcript."""
        # Find sentences mentioning the topic
        topic_lower = topic.lower()
        sentences = re.split(r"[.!?]+", transcript)

        matching = [
            s.strip()
            for s in sentences
            if topic_lower in s.lower() and len(s.strip()) > 20
        ]

        if matching:
            return " ".join(matching[:2])  # Take first 2 matching sentences

        return f"Bob discussed {topic} in this episode, emphasizing practical application and market psychology."

    def _extract_key_segments(self, transcript: str, max_segments: int = 3) -> List[str]:
        """
        Extract key segments from transcript when no summary is available.

        Args:
            transcript: Full transcript text
            max_segments: Maximum number of segments to extract

        Returns:
            List of key segment texts
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in transcript.split("\n\n") if p.strip()]

        # Filter for substantial paragraphs (at least 100 chars)
        substantial = [p for p in paragraphs if len(p) > 100]

        # Return first few substantial paragraphs
        return substantial[:max_segments]

    def process_all_transcripts(self) -> List[Question]:
        """
        Process all transcripts in the directory.

        Returns:
            List of all Question objects from all transcripts
        """
        raw_dir = self.transcripts_dir / "raw"
        if not raw_dir.exists():
            logger.warning(f"Raw transcripts directory not found: {raw_dir}")
            return []

        all_questions = []

        # Find all .txt files
        transcript_files = sorted(raw_dir.glob("*.txt"))

        logger.info(f"Found {len(transcript_files)} transcript files")

        for transcript_file in transcript_files:
            # Extract date from filename - handle multiple formats
            date_match = re.match(r"(\d{4}-\d{2}-\d{2})\.txt", transcript_file.name)
            if not date_match:
                # Try timestamp format: YYYY-MM-DDTHH-MM-SSZ.txt or YYYY-MM-DDTHH-MM-SSZ_F.txt
                timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2})T", transcript_file.name)
                if timestamp_match:
                    date = timestamp_match.group(1)
                else:
                    logger.warning(f"Could not parse date from filename: {transcript_file.name}")
                    continue
            else:
                date = date_match.group(1)

            try:
                # Parse transcript
                transcript = self.parse_transcript(transcript_file)
                if not transcript:
                    continue

                # Load summary if available
                summary = self.load_episode_summary(date)

                # Generate QA pairs
                questions = self.generate_qa_pairs(transcript, date, summary)

                # Add source identifier
                for q in questions:
                    q.source = f"episode-{date}"

                all_questions.extend(questions)
                logger.info(
                    f"Processed {transcript_file.name}: {len(questions)} questions"
                )

            except Exception as e:
                logger.error(f"Failed to process {transcript_file.name}: {e}")

        logger.info(f"Total questions generated: {len(all_questions)}")
        return all_questions
