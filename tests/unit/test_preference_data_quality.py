"""
Tests for curriculum preference data quality.

Ensures preference_data.jsonl has proper source attribution so that
ORPO training gets topic-specific preference pairs.
"""

import json
import pytest
from pathlib import Path
from collections import defaultdict


# Path to curriculum data (relative to project root)
TEXTBOOK_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bob_loukas" / "textbook"
PREFERENCE_FILE = TEXTBOOK_DATA_DIR / "preference_data.jsonl"
SFT_FILE = TEXTBOOK_DATA_DIR / "sft_data.jsonl"


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


@pytest.fixture
def preference_data():
    """Load preference data if file exists."""
    return _load_jsonl(PREFERENCE_FILE)


@pytest.fixture
def sft_data():
    """Load SFT data if file exists."""
    return _load_jsonl(SFT_FILE)


class TestPreferenceDataSourceAttribution:
    """Test that preference data has valid topic sources."""

    @pytest.mark.skipif(not PREFERENCE_FILE.exists(), reason="preference_data.jsonl not found")
    def test_no_unknown_sources(self, preference_data):
        """Preference pairs must not have source='unknown' or empty."""
        unknown = [p for p in preference_data if p.get("source") in (None, "", "unknown")]
        assert len(unknown) == 0, (
            f"Found {len(unknown)} preference pairs with missing or 'unknown' source. "
            "Regenerate preference data with proper source attribution."
        )

    @pytest.mark.skipif(not PREFERENCE_FILE.exists(), reason="preference_data.jsonl not found")
    @pytest.mark.skipif(not SFT_FILE.exists(), reason="sft_data.jsonl not found")
    def test_every_source_matches_sft_topic(self, preference_data, sft_data):
        """Every preference pair's source must match a valid SFT topic."""
        valid_sources = {item.get("source", item.get("topic_id")) for item in sft_data}
        valid_sources.discard("unknown")
        invalid = [p for p in preference_data if p.get("source") not in valid_sources]
        assert len(invalid) == 0, (
            f"Found {len(invalid)} preference pairs with source not in SFT topics. "
            f"Valid sources: {sorted(valid_sources)[:10]}..."
        )

    @pytest.mark.skipif(not PREFERENCE_FILE.exists(), reason="preference_data.jsonl not found")
    @pytest.mark.skipif(not SFT_FILE.exists(), reason="sft_data.jsonl not found")
    def test_preference_pairs_distributed_across_topics(self, preference_data, sft_data):
        """Preference pairs should be distributed across topics (not all in one)."""
        if len(preference_data) < 2:
            pytest.skip("Not enough preference pairs")
        by_source = defaultdict(int)
        for p in preference_data:
            src = p.get("source") or "unknown"
            by_source[src] += 1
        # We expect more than one topic to have pairs (after fixing unknown)
        num_topics_with_pairs = len([s for s, c in by_source.items() if s != "unknown" and c > 0])
        assert num_topics_with_pairs >= 2, (
            f"Preference pairs are not distributed across topics (found {num_topics_with_pairs} topics). "
            "Check source attribution."
        )
