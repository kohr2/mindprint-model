#!/usr/bin/env python3
"""
Regenerate preference pairs using LLM-generated rejections.

Usage:
    python scripts/regenerate_preferences.py \
        --input data/bob_loukas/transcripts/preference_data.jsonl \
        --output data/bob_loukas/transcripts/preference_data_llm.jsonl \
        --method llm
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.llm import LLMRejectionGenerator, RejectionConfig


async def regenerate_preferences(
    input_path: Path,
    output_path: Path,
    method: str = "llm",
) -> None:
    """
    Regenerate preference pairs.
    
    Args:
        input_path: Path to input preference data
        output_path: Path to save regenerated data
        method: "llm" or "rule_based"
    """
    # Load existing pairs
    pairs = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    print(f"Loaded {len(pairs)} preference pairs")
    
    if method == "llm":
        # Initialize LLM client
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        except ImportError:
            print("Error: anthropic package not installed")
            print("Install with: pip install anthropic")
            sys.exit(1)
        
        # Generate rejections
        config = RejectionConfig(
            cache_dir=Path(".cache/rejections"),
            max_concurrent=10,
        )
        generator = LLMRejectionGenerator(client, config)
        
        print("Generating LLM rejections...")
        regenerated = await generator.batch_generate(pairs)
        
    else:
        # Use rule-based (existing logic)
        from src.data_prep.preference_generator import PreferencePairGenerator
        generator = PreferencePairGenerator()
        
        regenerated = []
        for p in pairs:
            from src.data_prep.textbook_parser import Question
            q = Question(
                question=p["prompt"],
                reference_answer=p["chosen"],
                source=p.get("source", ""),
            )
            pair = generator.generate_pair(q)
            regenerated.append({
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "source": pair.source,
            })
    
    # Save regenerated pairs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p in regenerated:
            f.write(json.dumps(p) + '\n')
    
    print(f"Saved {len(regenerated)} regenerated pairs to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate preference pairs with LLM rejections"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input preference data JSONL file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for regenerated data"
    )
    parser.add_argument(
        "--method",
        choices=["llm", "rule_based"],
        default="llm",
        help="Rejection generation method"
    )
    
    args = parser.parse_args()
    
    asyncio.run(regenerate_preferences(
        args.input,
        args.output,
        args.method,
    ))


if __name__ == "__main__":
    main()
