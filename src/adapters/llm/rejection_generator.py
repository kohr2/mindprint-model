"""
LLM-generated rejections for preference learning.

Why: Rule-based rejection generation (voice stripping) creates
predictable, easily distinguishable rejections. LLM-generated
rejections are more realistic and challenging.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import hashlib
import json
from pathlib import Path


@dataclass
class RejectionConfig:
    """Rejection generation configuration."""
    model: str = "claude-3-haiku-20240307"
    temperature: float = 0.7
    max_tokens: int = 1024
    cache_dir: Optional[Path] = None
    max_concurrent: int = 10


class LLMRejectionGenerator:
    """
    Generate rejections using Claude/GPT.
    
    Features:
    - Async batch processing with rate limiting
    - Disk caching to avoid regenerating
    - Multiple rejection strategies
    """
    
    REJECTION_PROMPT = """You are helping create training data for preference learning.

Given a question and a high-quality answer, generate an alternative answer that is:
- Factually correct but less engaging
- Missing the personality and voice of the original
- More generic and textbook-like
- Shorter and less detailed

Question: {question}

High-quality answer: {chosen}

Generate a mediocre alternative (just the answer, no explanation):"""

    def __init__(self, client: Any, config: RejectionConfig):
        """
        Initialize LLM rejection generator.
        
        Args:
            client: Anthropic or OpenAI client
            config: Rejection generation configuration
        """
        self.client = client
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._cache: Dict[str, str] = {}
        
        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    async def generate_rejection(
        self,
        question: str,
        chosen: str
    ) -> str:
        """
        Generate a single rejection.
        
        Args:
            question: The prompt/question
            chosen: The high-quality chosen response
        
        Returns:
            Generated rejection response
        """
        cache_key = self._cache_key(question, chosen)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        async with self._semaphore:
            try:
                # Try Anthropic format first
                if hasattr(self.client, 'messages'):
                    response = await self.client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{
                            "role": "user",
                            "content": self.REJECTION_PROMPT.format(
                                question=question,
                                chosen=chosen
                            )
                        }]
                    )
                    rejection = response.content[0].text
                else:
                    # OpenAI format
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{
                            "role": "user",
                            "content": self.REJECTION_PROMPT.format(
                                question=question,
                                chosen=chosen
                            )
                        }]
                    )
                    rejection = response.choices[0].message.content
                
                self._cache[cache_key] = rejection
                self._save_cache_entry(cache_key, rejection)
                
                return rejection
            except Exception as e:
                # Fallback to rule-based if LLM fails
                from src.data_prep.preference_generator import PreferencePairGenerator
                generator = PreferencePairGenerator()
                # Create a mock question object
                from src.data_prep.textbook_parser import Question
                mock_q = Question(
                    question=question,
                    reference_answer=chosen,
                    source="llm_fallback"
                )
                pair = generator.generate_pair(mock_q)
                return pair.rejected
    
    async def batch_generate(
        self,
        pairs: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Generate rejections for a batch of pairs.
        
        Args:
            pairs: List of dicts with 'prompt' and 'chosen' keys
        
        Returns:
            List of pairs with 'rejected' key added
        """
        tasks = [
            self.generate_rejection(p["prompt"], p["chosen"])
            for p in pairs
        ]
        rejections = await asyncio.gather(*tasks)
        
        return [
            {**p, "rejected": r}
            for p, r in zip(pairs, rejections)
        ]
    
    def _cache_key(self, question: str, chosen: str) -> str:
        """Generate cache key."""
        content = f"{question}|||{chosen}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.config.cache_dir:
            return
        
        cache_file = self.config.cache_dir / "rejections_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
    
    def _save_cache_entry(self, key: str, value: str) -> None:
        """Save cache entry to disk."""
        if not self.config.cache_dir:
            return
        
        cache_file = self.config.cache_dir / "rejections_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception:
            pass  # Don't fail if cache write fails
