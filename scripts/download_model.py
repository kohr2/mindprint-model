#!/usr/bin/env python3
"""
Model Download Script for Bob Loukas Mindprint Training.

Downloads Gemma-3-12B or Qwen2.5-7B from HuggingFace Hub.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.download import main

if __name__ == "__main__":
    sys.exit(main())
