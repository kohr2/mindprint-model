"""
TranscriptQuestionGenerator - Generate high-quality questions from transcripts.

Uses Claude API to generate questions from transcript content, leveraging
episode summaries for context. Generates 10-20 questions per episode.
"""

from dataclasses import dataclass
from typing import List, Optional
import json
import logging
import os

from anthropic import Anthropic

from .textbook_parser import Question
from .transcript_processor import TranscriptProcessor, EpisodeSummary
from .question_generator import GenerationConfig

logger = logging.getLogger(__name__)


class TranscriptQuestionGenerator:
    """Generate questions from transcripts using Claude API."""

    def __init__(
        self,
        processor: TranscriptProcessor,
        config: Optional[GenerationConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the generator.

        Args:
            processor: TranscriptProcessor instance
            config: Generation configuration
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.processor = processor
        self.config = config or GenerationConfig()

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = Anthropic(api_key=api_key)

    def generate_for_episode(
        self,
        transcript: str,
        date: str,
        summary: Optional[EpisodeSummary] = None,
        target_count: int = 15,
    ) -> List[Question]:
        """
        Generate questions for a single episode.

        Args:
            transcript: Raw transcript text
            date: Episode date in YYYY-MM-DD format
            summary: Optional episode summary for context
            target_count: Target number of questions to generate

        Returns:
            List of Question objects
        """
        logger.info(f"Generating {target_count} questions for episode {date}")

        # Mix of summary-based (3-5) and topic-based (7-15)
        summary_count = min(5, max(3, target_count // 4))
        topic_count = target_count - summary_count

        questions = []

        # Generate summary-based questions
        if summary:
            summary_questions = self._generate_summary_questions(
                transcript, date, summary, count=summary_count
            )
            questions.extend(summary_questions)

        # Generate topic-based questions
        topic_questions = self._generate_topic_questions(
            transcript, date, summary, count=topic_count
        )
        questions.extend(topic_questions)

        # Add source identifier
        for q in questions:
            q.source = f"episode-{date}"

        logger.info(f"Generated {len(questions)} questions for episode {date}")
        return questions[:target_count]  # Ensure we don't exceed target

    def _generate_summary_questions(
        self,
        transcript: str,
        date: str,
        summary: EpisodeSummary,
        count: int,
    ) -> List[Question]:
        """Generate high-level summary-based questions."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_summary_prompt(transcript, date, summary, count)

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            content = response.content[0].text
            questions = self._parse_generated_questions(content)
            return questions[:count]

        except Exception as e:
            logger.error(f"Failed to generate summary questions: {e}")
            return []

    def _generate_topic_questions(
        self,
        transcript: str,
        date: str,
        summary: Optional[EpisodeSummary],
        count: int,
    ) -> List[Question]:
        """Generate topic-based questions."""
        system_prompt = self._build_system_prompt()

        # Extract topics from summary or transcript
        topics = summary.topics if summary else self._extract_topics_from_transcript(transcript)

        if not topics:
            # Fallback: generate general questions
            return self._generate_general_questions(transcript, date, count)

        # Generate questions for each topic
        questions_per_topic = max(1, count // max(len(topics), 1))
        all_questions = []

        for topic in topics[:10]:  # Limit to 10 topics
            user_prompt = self._build_topic_prompt(transcript, date, topic, questions_per_topic)

            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                content = response.content[0].text
                topic_questions = self._parse_generated_questions(content)
                all_questions.extend(topic_questions[:questions_per_topic])

                if len(all_questions) >= count:
                    break

            except Exception as e:
                logger.error(f"Failed to generate questions for topic {topic}: {e}")

        return all_questions[:count]

    def _generate_general_questions(
        self, transcript: str, date: str, count: int
    ) -> List[Question]:
        """Generate general questions when no topics available."""
        system_prompt = self._build_system_prompt()
        user_prompt = f"""Generate {count} questions about Bob's market analysis from the transcript dated {date}.

Transcript:
{transcript[:3000]}  # Truncate for context

Generate questions that:
1. Test understanding of Bob's market thesis
2. Explore cycle analysis concepts
3. Cover risk management insights
4. Ask about specific indicators or patterns

Format as JSON array with "question" and "reference_answer" fields.
"""

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            content = response.content[0].text
            return self._parse_generated_questions(content)[:count]

        except Exception as e:
            logger.error(f"Failed to generate general questions: {e}")
            return []

    def _build_system_prompt(self) -> str:
        """Build the system prompt with Bob's style guide."""
        return """You are Bob Loukas, a Bitcoin cycle trading expert with over 25 years of market experience. You are creating quiz questions and reference answers from your video transcripts.

Your communication style:
- Confident but not arrogant
- Educational and engaging
- Focus on patterns and market psychology
- Use phrases like "I've tracked", "I've seen", "In my experience"
- Emphasize practical application over theory
- Reference your experience observing market cycles

CRITICAL: The 4-year market cycle is NOT caused by the halving. The halving coincides with the cycle but does not cause it. The cycle is driven by market psychology and capital flows. Never conflate correlation with causation.

When generating reference answers:
1. Write in first person as Bob
2. Use Bob's exact words from the transcript when possible
3. Include confidence markers ("I've observed", "Here's what I've seen")
4. Reference market psychology concepts
5. Use cycle terminology correctly (DCL, right-translated, Bressert bands)
6. Be educational but direct
7. Give practical, actionable insights
8. Preserve Bob's casual, conversational tone"""

    def _build_summary_prompt(
        self,
        transcript: str,
        date: str,
        summary: EpisodeSummary,
        count: int,
    ) -> str:
        """Build prompt for summary-based questions."""
        return f"""Generate {count} high-level questions about Bob's market analysis from {date}.

Episode Summary:
- Thesis: {summary.thesis}
- Risks: {summary.risks}
- Opportunities: {summary.opportunities}
- Topics: {', '.join(summary.topics[:10])}

Transcript excerpt (for reference answers):
{transcript[:2000]}

Generate questions that:
1. Test understanding of Bob's overall thesis for this episode
2. Explore the risks and opportunities he identified
3. Cover his market outlook and expectations
4. Ask about cycle positioning and indicators

Format as JSON array:
```json
[
  {{
    "question": "What was Bob's market thesis on {date}?",
    "reference_answer": "[Use Bob's exact words from transcript when possible]",
    "key_concepts": ["thesis", "market_outlook"]
  }},
  ...
]
```

Generate exactly {count} questions."""

    def _build_topic_prompt(
        self, transcript: str, date: str, topic: str, count: int
    ) -> str:
        """Build prompt for topic-based questions."""
        return f"""Generate {count} questions about "{topic}" from Bob's episode on {date}.

Transcript excerpt:
{transcript[:2000]}

Generate questions that:
1. Test understanding of Bob's view on {topic}
2. Explore how {topic} relates to cycle analysis
3. Cover practical implications
4. Use Bob's terminology and phrasing

Format as JSON array with "question", "reference_answer", and "key_concepts" fields.
Generate exactly {count} questions."""

    def _extract_topics_from_transcript(self, transcript: str) -> List[str]:
        """Extract topics from transcript using simple keyword matching."""
        # Common Bob Loukas topics
        common_topics = [
            "4-year cycle",
            "Bressert bands",
            "DCL",
            "right-translated",
            "left-translated",
            "accumulation",
            "distribution",
            "market psychology",
            "cycle low",
            "halving",
        ]

        found_topics = []
        transcript_lower = transcript.lower()

        for topic in common_topics:
            if topic.lower() in transcript_lower:
                found_topics.append(topic)

        return found_topics[:10]  # Limit to 10 topics

    def _parse_generated_questions(self, content: str) -> List[Question]:
        """Parse generated questions from Claude's response."""
        questions = []

        # Try to extract JSON from the response
        import re

        # Look for JSON in code blocks
        json_match = None
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, content)
            if match:
                json_match = match.group(1)
                break

        if not json_match:
            # Try to find JSON array directly
            array_match = re.search(r"\[[\s\S]*\]", content)
            if array_match:
                json_match = array_match.group(0)

        if not json_match:
            logger.error("Could not find JSON in response")
            return []

        try:
            data = json.loads(json_match)
            if not isinstance(data, list):
                data = [data]

            for item in data:
                if isinstance(item, dict) and "question" in item and "reference_answer" in item:
                    question = Question(
                        question=item["question"],
                        reference_answer=item["reference_answer"],
                        question_type="open",
                        key_concepts=item.get("key_concepts", []),
                    )
                    questions.append(question)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Content was: {json_match[:500]}")

        return questions
