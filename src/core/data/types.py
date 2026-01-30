"""
Core data types for training.

Defines the fundamental data structures used throughout the codebase.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreferencePair:
    """
    A preference pair for preference learning.
    
    Represents a prompt with a chosen (preferred) response and
    a rejected (less preferred) response.
    """
    prompt: str
    chosen: str
    rejected: str
    source: str = ""  # Where this pair came from (topic/chapter/etc.)
    
    def __post_init__(self):
        """Validate preference pair."""
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if not self.chosen:
            raise ValueError("chosen cannot be empty")
        if not self.rejected:
            raise ValueError("rejected cannot be empty")


@dataclass
class TrainingSample:
    """
    A training sample for supervised fine-tuning.
    
    Represents a prompt-response pair for SFT training.
    """
    instruction: str
    output: str
    input: Optional[str] = None  # Optional input context
    source: str = ""  # Where this sample came from
    
    def __post_init__(self):
        """Validate training sample."""
        if not self.instruction:
            raise ValueError("instruction cannot be empty")
        if not self.output:
            raise ValueError("output cannot be empty")
