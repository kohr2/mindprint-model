"""
QuestionGenerator - Use Claude API to generate additional questions for topics.

Generates questions in Bob Loukas's voice to augment topics with fewer than 10 questions.
"""

from dataclasses import dataclass
from typing import List, Optional
import json
import logging
import os

from anthropic import Anthropic

from .textbook_parser import Question, TopicQuiz, TextbookParser

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for question generation."""

    target_questions: int = 10
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7


class QuestionGenerator:
    """Generate additional questions using Claude API."""

    def __init__(
        self,
        parser: TextbookParser,
        config: Optional[GenerationConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the question generator.

        Args:
            parser: TextbookParser instance for accessing content
            config: Generation configuration
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.parser = parser
        self.config = config or GenerationConfig()
        self.style_guide = parser.get_style_guide() or ""

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )
        self.client = Anthropic(api_key=api_key)

    def augment_topic(self, topic_quiz: TopicQuiz) -> TopicQuiz:
        """
        Augment a topic quiz to reach target number of questions.

        Args:
            topic_quiz: The topic quiz to augment

        Returns:
            Updated TopicQuiz with additional questions
        """
        current_count = len(topic_quiz.questions)
        target = self.config.target_questions

        if current_count >= target:
            logger.info(f"{topic_quiz.identifier}: Already has {current_count} questions")
            return topic_quiz

        questions_needed = target - current_count
        logger.info(
            f"{topic_quiz.identifier}: Generating {questions_needed} additional questions"
        )

        # Get topic content for context
        topic_content = self.parser.get_topic_content(
            topic_quiz.unit, topic_quiz.chapter, topic_quiz.topic
        )

        # Generate new questions
        new_questions = self._generate_questions(
            topic_quiz=topic_quiz,
            topic_content=topic_content,
            count=questions_needed,
        )

        # Add new questions to the quiz
        topic_quiz.questions.extend(new_questions)
        logger.info(
            f"{topic_quiz.identifier}: Now has {len(topic_quiz.questions)} questions"
        )

        return topic_quiz

    def augment_all(self, topic_quizzes: List[TopicQuiz]) -> List[TopicQuiz]:
        """
        Augment all topic quizzes to reach target number of questions.

        Args:
            topic_quizzes: List of topic quizzes to augment

        Returns:
            List of augmented TopicQuiz objects
        """
        augmented = []
        for quiz in topic_quizzes:
            try:
                augmented_quiz = self.augment_topic(quiz)
                augmented.append(augmented_quiz)
            except Exception as e:
                logger.error(f"Failed to augment {quiz.identifier}: {e}")
                augmented.append(quiz)  # Keep original on failure

        return augmented

    def _generate_questions(
        self,
        topic_quiz: TopicQuiz,
        topic_content: Optional[str],
        count: int,
    ) -> List[Question]:
        """Generate new questions using Claude API."""
        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(topic_quiz, topic_content, count)

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse the response
            content = response.content[0].text
            questions = self._parse_generated_questions(content)

            return questions[:count]  # Ensure we don't exceed requested count

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return []

    def _build_system_prompt(self) -> str:
        """Build the system prompt with Bob's style guide."""
        return f"""You are Bob Loukas, a Bitcoin cycle trading expert with over 25 years of market experience. You are creating quiz questions and reference answers for a training dataset.

Your communication style:
- Confident but not arrogant
- Educational and engaging
- Focus on patterns and market psychology
- Use phrases like "I've tracked", "I've seen", "In my experience"
- Emphasize practical application over theory
- Reference your experience observing market cycles

{self.style_guide}

CRITICAL: The 4-year market cycle is NOT caused by the halving. The halving coincides with the cycle but does not cause it. The cycle is driven by market psychology and capital flows. Never conflate correlation with causation.

When generating reference answers:
1. Write in first person as Bob
2. Include confidence markers ("I've observed", "Here's what I've seen")
3. Reference market psychology concepts
4. Use cycle terminology correctly
5. Be educational but direct
6. Give practical, actionable insights"""

    def _build_user_prompt(
        self,
        topic_quiz: TopicQuiz,
        topic_content: Optional[str],
        count: int,
    ) -> str:
        """Build the user prompt for question generation."""
        # Format existing questions as examples
        existing_examples = ""
        for i, q in enumerate(topic_quiz.questions[:3], 1):  # Use up to 3 examples
            existing_examples += f"""
Example Question {i}:
{q.question}

Example Reference Answer {i}:
{q.reference_answer}
---
"""

        # Include topic content if available
        content_section = ""
        if topic_content:
            # Truncate if too long
            max_content_length = 4000
            if len(topic_content) > max_content_length:
                topic_content = topic_content[:max_content_length] + "\n\n[Content truncated]"
            content_section = f"""
## Topic Content

{topic_content}
"""

        return f"""Generate {count} NEW quiz questions for the topic: "{topic_quiz.title}"

Topic: {topic_quiz.topic}
Chapter: {topic_quiz.chapter}
Unit: {topic_quiz.unit}
{content_section}
## Existing Questions (for reference, DO NOT duplicate these)

{existing_examples}

## Instructions

Generate {count} new questions that:
1. Test different aspects of the topic than the existing questions
2. Range from conceptual understanding to practical application
3. Include questions that test critical thinking about cycles
4. Have reference answers written in Bob's voice (confident, educational, pattern-focused)

Format your response as JSON array:
```json
[
  {{
    "question": "The question text here?",
    "reference_answer": "Bob's answer in first person, with confidence markers and educational tone...",
    "key_concepts": ["concept1", "concept2"]
  }},
  ...
]
```

Generate exactly {count} questions. Ensure each question is unique and tests different knowledge than existing questions."""

    def _parse_generated_questions(self, content: str) -> List[Question]:
        """Parse the generated questions from Claude's response."""
        questions = []

        # Try to extract JSON from the response
        json_match = None

        # Look for JSON in code blocks
        code_block_match = None
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = __import__("re").search(pattern, content)
            if match:
                code_block_match = match.group(1)
                break

        if code_block_match:
            json_str = code_block_match
        else:
            # Try to find JSON array directly
            array_match = __import__("re").search(r"\[[\s\S]*\]", content)
            if array_match:
                json_str = array_match.group(0)
            else:
                logger.error("Could not find JSON in response")
                return []

        try:
            data = json.loads(json_str)
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
            logger.debug(f"Content was: {json_str[:500]}")

        return questions
