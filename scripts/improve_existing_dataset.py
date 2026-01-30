#!/usr/bin/env python3
"""
Improve existing dataset by applying voice enhancement and length normalization.

Post-processes existing sft_data.jsonl and preference_data.jsonl to:
- Enhance voice markers in answers
- Normalize answer lengths (600-1200 chars)
- Ensure proper topic mapping
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_prep.voice_aware_processor import VoiceAwareProcessor
from src.data_prep.answer_splitter import AnswerSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: Path):
    """Save JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def improve_sft_data(
    sft_data: List[Dict],
    enhance_voice: bool = True,
    normalize_lengths: bool = True,
    min_length: int = 600,
    max_length: int = 1200,
    min_voice_density: float = 20.0,
) -> List[Dict]:
    """
    Improve SFT data with voice enhancement and length normalization.

    Args:
        sft_data: List of SFT examples
        enhance_voice: Whether to enhance voice markers
        normalize_lengths: Whether to normalize lengths
        min_length: Minimum answer length
        max_length: Maximum answer length
        min_voice_density: Minimum voice marker density percentage

    Returns:
        Improved SFT data (may have more examples if answers were split)
    """
    voice_processor = VoiceAwareProcessor(
        min_voice_density=min_voice_density,
        min_length=min_length,
        max_length=max_length,
    )
    answer_splitter = AnswerSplitter(
        target_min=min_length,
        target_max=max_length,
    )

    improved_data = []
    stats = {
        "original_count": len(sft_data),
        "enhanced_count": 0,
        "split_count": 0,
        "total_after": 0,
    }

    for item in sft_data:
        question = item.get("instruction", item.get("question", ""))
        answer = item.get("output", item.get("reference_answer", ""))
        source = item.get("source", "unknown")

        # Enhance voice if enabled
        if enhance_voice:
            answer = voice_processor.enhance_answer(question, answer, source)
            stats["enhanced_count"] += 1

        # Normalize length if enabled
        if normalize_lengths:
            # Check if answer needs splitting
            if len(answer) > max_length:
                segments = answer_splitter.split_answer(question, answer, source)
                
                if len(segments) > 1:
                    stats["split_count"] += 1
                    # Create multiple examples from split answer
                    for segment in segments:
                        improved_data.append({
                            "instruction": segment["question"],
                            "input": "",
                            "output": segment["answer"],
                            "source": source,
                        })
                else:
                    # Splitter returned single segment (shouldn't happen if > max_length)
                    # Use normalize_length from voice processor as fallback
                    normalized_segments = voice_processor.normalize_length(answer)
                    if len(normalized_segments) > 1:
                        stats["split_count"] += 1
                        for i, segment_answer in enumerate(normalized_segments, 1):
                            sub_question = f"{question} (Part {i})" if len(normalized_segments) > 1 else question
                            improved_data.append({
                                "instruction": sub_question,
                                "input": "",
                                "output": segment_answer,
                                "source": source,
                            })
                    else:
                        improved_data.append({
                            "instruction": question,
                            "input": "",
                            "output": normalized_segments[0] if normalized_segments else answer,
                            "source": source,
                        })
            else:
                # Answer is within length limits, just use enhanced version
                improved_data.append({
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "source": source,
                })
        else:
            # Just update answer if enhanced
            improved_data.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "source": source,
            })

    stats["total_after"] = len(improved_data)
    return improved_data, stats


def improve_preference_data(
    pref_data: List[Dict],
    enhance_voice: bool = True,
    normalize_lengths: bool = True,
    min_length: int = 600,
    max_length: int = 1200,
) -> List[Dict]:
    """
    Improve preference pair data.

    Args:
        pref_data: List of preference pairs
        enhance_voice: Whether to enhance voice markers
        normalize_lengths: Whether to normalize lengths
        min_length: Minimum answer length
        max_length: Maximum answer length

    Returns:
        Improved preference data
    """
    voice_processor = VoiceAwareProcessor(
        min_length=min_length,
        max_length=max_length,
    )

    improved_data = []
    stats = {
        "original_count": len(pref_data),
        "enhanced_count": 0,
    }

    for item in pref_data:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        source = item.get("source", "unknown")

        # Enhance chosen answer voice
        if enhance_voice:
            chosen = voice_processor.enhance_answer(prompt, chosen, source)
            stats["enhanced_count"] += 1

        # Normalize chosen answer length (rejected stays as-is for contrast)
        if normalize_lengths and len(chosen) > max_length:
            segments = voice_processor.normalize_length(chosen)
            # Use first segment (most important part)
            chosen = segments[0] if segments else chosen

        improved_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": source,  # Ensure source is included
        })

    stats["total_after"] = len(improved_data)
    return improved_data, stats


def main():
    parser = argparse.ArgumentParser(description="Improve existing dataset quality")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing sft_data.jsonl and preference_data.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for improved dataset"
    )
    parser.add_argument(
        "--enhance-voice",
        action="store_true",
        default=True,
        help="Enhance voice markers (default: True)"
    )
    parser.add_argument(
        "--no-enhance-voice",
        action="store_false",
        dest="enhance_voice",
        help="Disable voice enhancement"
    )
    parser.add_argument(
        "--normalize-lengths",
        action="store_true",
        default=True,
        help="Normalize answer lengths (default: True)"
    )
    parser.add_argument(
        "--no-normalize-lengths",
        action="store_false",
        dest="normalize_lengths",
        help="Disable length normalization"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=600,
        help="Minimum answer length (default: 600)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1200,
        help="Maximum answer length (default: 1200)"
    )
    parser.add_argument(
        "--min-voice-density",
        type=float,
        default=20.0,
        help="Minimum voice marker density percentage (default: 20.0)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_file = input_dir / "sft_data.jsonl"
    pref_file = input_dir / "preference_data.jsonl"

    if not sft_file.exists():
        logger.error(f"SFT data file not found: {sft_file}")
        return 1

    if not pref_file.exists():
        logger.error(f"Preference data file not found: {pref_file}")
        return 1

    logger.info("Loading SFT data...")
    sft_data = load_jsonl(sft_file)
    logger.info(f"Loaded {len(sft_data)} SFT examples")

    logger.info("Loading preference data...")
    pref_data = load_jsonl(pref_file)
    logger.info(f"Loaded {len(pref_data)} preference pairs")

    # Improve SFT data
    logger.info("Improving SFT data...")
    improved_sft, sft_stats = improve_sft_data(
        sft_data,
        enhance_voice=args.enhance_voice,
        normalize_lengths=args.normalize_lengths,
        min_length=args.min_length,
        max_length=args.max_length,
        min_voice_density=args.min_voice_density,
    )

    logger.info(f"SFT improvement stats: {sft_stats}")

    # Improve preference data
    logger.info("Improving preference data...")
    improved_pref, pref_stats = improve_preference_data(
        pref_data,
        enhance_voice=args.enhance_voice,
        normalize_lengths=args.normalize_lengths,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    logger.info(f"Preference improvement stats: {pref_stats}")

    # Save improved data
    logger.info("Saving improved dataset...")
    save_jsonl(improved_sft, output_dir / "sft_data.jsonl")
    save_jsonl(improved_pref, output_dir / "preference_data.jsonl")

    # Copy other files
    import shutil
    for file_name in ["quiz_data.json", "chapter_tests.json", "unit_exams.json", "final_assessment.json", "critical_distinctions.jsonl"]:
        src_file = input_dir / file_name
        if src_file.exists():
            shutil.copy(src_file, output_dir / file_name)

    print("\n" + "=" * 80)
    print("DATASET IMPROVEMENT COMPLETE")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nSFT Data:")
    print(f"  Original: {sft_stats['original_count']} examples")
    print(f"  Enhanced: {sft_stats['enhanced_count']} examples")
    print(f"  Split: {sft_stats['split_count']} answers split")
    print(f"  Final: {sft_stats['total_after']} examples")
    print(f"\nPreference Data:")
    print(f"  Original: {pref_stats['original_count']} pairs")
    print(f"  Enhanced: {pref_stats['enhanced_count']} pairs")
    print(f"  Final: {pref_stats['total_after']} pairs")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
