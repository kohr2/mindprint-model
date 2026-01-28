"""
Utility functions for adapter path management.

Provides centralized path management for LoRA adapters across training phases.
"""

from pathlib import Path
from typing import Tuple


def get_adapter_paths(
    output_dir: str,
    unit_id: str,
    chapter_id: str,
    topic_id: str,
    phase: str = "all"
) -> Tuple[Path, Path, Path]:
    """
    Generate standardized adapter paths for a topic.

    Args:
        output_dir: Base output directory
        unit_id: Unit identifier (e.g., "unit-01")
        chapter_id: Chapter identifier (e.g., "chapter-01")
        topic_id: Topic identifier (e.g., "topic-01" or full path)
        phase: Phase to return path for ("sft", "reward", "ppo", or "all")

    Returns:
        Tuple of (sft_path, reward_path, ppo_path)

    Example:
        >>> sft, reward, ppo = get_adapter_paths(
        ...     "./output", "unit-01", "chapter-01", "topic-01"
        ... )
        >>> print(sft)
        output/adapters/unit-01/chapter-01/topic-01/sft_adapter
    """
    base = Path(output_dir) / "adapters" / unit_id / chapter_id / topic_id

    sft_path = base / "sft_adapter"
    reward_path = base / "reward_adapter"
    ppo_path = base / "ppo_adapter"

    return (sft_path, reward_path, ppo_path)


def get_merged_adapter_path(output_dir: str, unit_id: str) -> Path:
    """
    Get path for merged unit adapter.

    Args:
        output_dir: Base output directory
        unit_id: Unit identifier (e.g., "unit-01")

    Returns:
        Path to merged adapter directory

    Example:
        >>> path = get_merged_adapter_path("./output", "unit-01")
        >>> print(path)
        output/merged_adapters/unit-01_merged
    """
    return Path(output_dir) / "merged_adapters" / f"{unit_id}_merged"


def parse_topic_id(topic_id: str) -> Tuple[str, str, str]:
    """
    Parse topic_id into unit, chapter, and topic components.

    Args:
        topic_id: Topic identifier (e.g., "unit-01/chapter-01/topic-01")

    Returns:
        Tuple of (unit_id, chapter_id, topic_name)

    Example:
        >>> unit, chapter, topic = parse_topic_id("unit-01/chapter-01/topic-01")
        >>> print(unit, chapter, topic)
        unit-01 chapter-01 topic-01
    """
    parts = topic_id.split("/")

    if len(parts) >= 3:
        # Full path: unit-01/chapter-01/topic-01
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        # Chapter level: unit-01/chapter-01
        return parts[0], parts[1], "default"
    elif len(parts) == 1:
        # Unit level: unit-01
        return parts[0], "default", "default"
    else:
        # Fallback
        return "unknown", "unknown", topic_id
