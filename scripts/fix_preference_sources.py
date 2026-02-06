#!/usr/bin/env python3
"""
Patch preference_data.jsonl with correct source attribution by matching
prompts to SFT instructions. Run from project root.

Usage:
    python scripts/fix_preference_sources.py [--data-dir data/bob_loukas/textbook]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch preference_data.jsonl with sources from SFT data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/bob_loukas/textbook"),
        help="Directory containing sft_data.jsonl and preference_data.jsonl",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    sft_path = data_dir / "sft_data.jsonl"
    pref_path = data_dir / "preference_data.jsonl"

    if not sft_path.exists():
        print(f"Error: {sft_path} not found", file=sys.stderr)
        return 1
    if not pref_path.exists():
        print(f"Error: {pref_path} not found", file=sys.stderr)
        return 1

    # Build prompt -> source from SFT
    prompt_to_source = {}
    with open(sft_path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            inst = obj.get("instruction", "")
            src = obj.get("source", obj.get("topic_id", ""))
            if inst and src and src != "unknown":
                prompt_to_source[inst] = src

    # Load preference data and patch
    pairs = []
    with open(pref_path) as f:
        for line in f:
            if not line.strip():
                continue
            pairs.append(json.loads(line))

    # Fallback: use first available unit if no exact match (so no pair stays "unknown")
    fallback_source = next(iter(prompt_to_source.values()), "unit-01") if prompt_to_source else "unit-01"
    if fallback_source and "/" in fallback_source:
        fallback_source = fallback_source.split("/")[0]  # e.g. unit-01

    patched = 0
    for p in pairs:
        prompt = p.get("prompt", "")
        src = prompt_to_source.get(prompt)
        if src is not None:
            p["source"] = src
            patched += 1
        else:
            # No exact match: assign fallback so pipeline can still group (e.g. unit-01)
            p["source"] = fallback_source

    # Write back
    with open(pref_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Patched {patched}/{len(pairs)} preference pairs with source from SFT data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
