#!/usr/bin/env python3
"""
Evaluation Script for Bob Loukas Mindprint Models.

Runs the hierarchical evaluation pipeline:
Topic → Chapter → Unit → Final Assessment

Measures both accuracy and voice fidelity.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.cli import main

if __name__ == "__main__":
    sys.exit(main())
