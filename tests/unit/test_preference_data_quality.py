"""
Tests for curriculum preference data quality.

Ensures preference_data.jsonl has proper source attribution so that
ORPO training gets topic-specific preference pairs.
"""

import json
import re
import pytest
from pathlib import Path
from collections import defaultdict


# Path to curriculum data (relative to project root)
TEXTBOOK_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bob_loukas" / "textbook"
TRANSCRIPTS_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bob_loukas" / "transcripts"
TEXTBOOK_PREFERENCE_FILE = TEXTBOOK_DATA_DIR / "preference_data.jsonl"
TRANSCRIPTS_PREFERENCE_FILE = TRANSCRIPTS_DATA_DIR / "preference_data.jsonl"


def _load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    if not path.exists():
        return []
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _extract_topic_id(item: dict) -> str:
    """Extract topic ID from preference pair (mirrors pipeline logic)."""
    source = item.get("source") or item.get("topic_id")
    if source:
        return source
    prompt = item.get("prompt", "")
    match = re.search(r"(\d{4}-\d{2}-\d{2})", prompt)
    if match:
        return f"episode-{match.group(1)}"
    return "general"


class TestPreferenceDataSourceAttribution:
    """Test that preference data has valid topic sources."""

    @pytest.fixture(params=[
        pytest.param(TEXTBOOK_PREFERENCE_FILE, id="textbook"),
        pytest.param(TRANSCRIPTS_PREFERENCE_FILE, id="transcripts"),
    ])
    def preference_data(self, request):
        path = request.param
        if not path.exists():
            pytest.skip(f"{path.name} not found")
        return _load_jsonl(path)

    def test_no_unknown_topics(self, preference_data):
        """Every preference pair must resolve to a known topic ID."""
        unknown = [p for p in preference_data if _extract_topic_id(p) in ("unknown", "")]
        assert len(unknown) == 0, (
            f"Found {len(unknown)} preference pairs that cannot be assigned a topic. "
            "Ensure each pair has a 'source' field or a date in the prompt."
        )

    def test_preference_pairs_distributed_across_topics(self, preference_data):
        """Preference pairs should be distributed across topics (not all in one)."""
        if len(preference_data) < 2:
            pytest.skip("Not enough preference pairs")
        by_topic = defaultdict(int)
        for p in preference_data:
            by_topic[_extract_topic_id(p)] += 1
        num_topics = len([t for t, c in by_topic.items() if t != "unknown" and c > 0])
        assert num_topics >= 2, (
            f"Preference pairs are not distributed across topics (found {num_topics} topics). "
            "Check source attribution."
        )
