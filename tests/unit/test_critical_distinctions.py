"""
Tests for CriticalDistinctions - high-value preference pairs for Bob's key distinctions.

The most critical distinction: The 4-year cycle is NOT caused by the halving.
"""

import pytest
from src.data_prep.critical_distinctions import (
    CriticalDistinctions,
    CriticalDistinction,
)
from src.data_prep.preference_generator import PreferencePair


@pytest.fixture
def distinctions() -> CriticalDistinctions:
    """Create CriticalDistinctions instance."""
    return CriticalDistinctions()


class TestCriticalDistinctionsInitialization:
    """Test initialization and structure."""

    def test_creates_all_distinctions(self, distinctions: CriticalDistinctions) -> None:
        """All distinction categories are created."""
        assert len(distinctions.distinctions) == 4

    def test_distinction_categories_exist(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """All expected distinction categories exist."""
        names = [d.name for d in distinctions.distinctions]

        assert "halving_vs_cycle" in names
        assert "correlation_vs_causation" in names
        assert "timing_discipline" in names
        assert "investor_vs_trader" in names


class TestHalvingVsCycleDistinction:
    """Test the most critical distinction: halving does not cause the cycle."""

    def test_halving_vs_cycle_has_five_pairs(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Halving vs cycle has 5 preference pairs."""
        distinction = distinctions.get_distinction("halving_vs_cycle")

        assert len(distinction.pairs) == 5

    def test_chosen_denies_halving_causation(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen responses deny that halving causes the cycle."""
        distinction = distinctions.get_distinction("halving_vs_cycle")

        denial_phrases = [
            "not caused by",
            "doesn't cause",
            "does not cause",
            "not the driver",
            "not the cause",
            "coincide",
            "isn't driven",
            "correlation isn't causation",
            "not the engine",
            "isn't unique to bitcoin",  # Implies cycle exists without halving
            "focus less on the halving",  # Downplays halving importance
            "cycle would happen anyway",  # Implies cycle independent of halving
        ]

        for pair in distinction.pairs:
            # Each chosen response should deny halving causation
            chosen_lower = pair.chosen.lower()
            has_denial = any(phrase in chosen_lower for phrase in denial_phrases)
            assert has_denial, f"Chosen should deny halving causation: {pair.prompt}"

    def test_rejected_claims_halving_causation(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Rejected responses incorrectly claim halving causes the cycle."""
        distinction = distinctions.get_distinction("halving_vs_cycle")

        for pair in distinction.pairs:
            rejected_lower = pair.rejected.lower()
            # Rejected should claim halving causes cycle or similar misconception
            assert any(
                phrase in rejected_lower
                for phrase in ["caused by", "causes the", "drives the", "supply shock", "supply reduction"]
            ), f"Rejected should claim halving causation: {pair.prompt}"

    def test_pairs_have_source_identifier(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """All pairs have source identifiers."""
        distinction = distinctions.get_distinction("halving_vs_cycle")

        for pair in distinction.pairs:
            assert pair.source == "critical_distinction:halving_vs_cycle"


class TestCorrelationVsCausation:
    """Test correlation vs causation distinction."""

    def test_has_two_pairs(self, distinctions: CriticalDistinctions) -> None:
        """Correlation vs causation has 2 preference pairs."""
        distinction = distinctions.get_distinction("correlation_vs_causation")

        assert len(distinction.pairs) == 2

    def test_chosen_emphasizes_mechanism(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen responses emphasize need for mechanism."""
        distinction = distinctions.get_distinction("correlation_vs_causation")

        # First pair should discuss mechanism
        first_chosen = distinction.pairs[0].chosen.lower()
        assert "mechanism" in first_chosen or "precedence" in first_chosen


class TestTimingDiscipline:
    """Test timing discipline distinction."""

    def test_has_two_pairs(self, distinctions: CriticalDistinctions) -> None:
        """Timing discipline has 2 preference pairs."""
        distinction = distinctions.get_distinction("timing_discipline")

        assert len(distinction.pairs) == 2

    def test_chosen_emphasizes_rules(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen responses emphasize systematic rules."""
        distinction = distinctions.get_distinction("timing_discipline")

        for pair in distinction.pairs:
            chosen_lower = pair.chosen.lower()
            # Should mention rules, system, or discipline
            assert any(
                word in chosen_lower
                for word in ["rules", "system", "discipline", "cycle", "plan"]
            )


class TestInvestorVsTrader:
    """Test investor vs trader distinction."""

    def test_has_one_pair(self, distinctions: CriticalDistinctions) -> None:
        """Investor vs trader has 1 preference pair."""
        distinction = distinctions.get_distinction("investor_vs_trader")

        assert len(distinction.pairs) == 1

    def test_distinguishes_timeframes(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen response distinguishes timeframes."""
        distinction = distinctions.get_distinction("investor_vs_trader")
        pair = distinction.pairs[0]

        chosen_lower = pair.chosen.lower()
        # Should mention different timeframes
        assert "multi-year" in chosen_lower or "time horizon" in chosen_lower


class TestGetAllPairs:
    """Test getting all preference pairs."""

    def test_returns_all_pairs(self, distinctions: CriticalDistinctions) -> None:
        """get_all_pairs returns pairs from all distinctions."""
        all_pairs = distinctions.get_all_pairs()

        # 5 halving + 2 correlation + 2 timing + 1 investor = 10 pairs
        assert len(all_pairs) == 10

    def test_all_pairs_are_preference_pairs(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """All returned pairs are PreferencePair instances."""
        all_pairs = distinctions.get_all_pairs()

        for pair in all_pairs:
            assert isinstance(pair, PreferencePair)

    def test_all_pairs_have_content(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """All pairs have prompt, chosen, and rejected content."""
        all_pairs = distinctions.get_all_pairs()

        for pair in all_pairs:
            assert pair.prompt
            assert pair.chosen
            assert pair.rejected
            assert len(pair.chosen) > len(pair.rejected) * 0.5  # Chosen is substantial


class TestGetDistinction:
    """Test getting specific distinctions."""

    def test_get_existing_distinction(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Can get existing distinction by name."""
        distinction = distinctions.get_distinction("halving_vs_cycle")

        assert distinction.name == "halving_vs_cycle"
        assert isinstance(distinction, CriticalDistinction)

    def test_get_nonexistent_raises_keyerror(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Getting nonexistent distinction raises KeyError."""
        with pytest.raises(KeyError, match="Unknown distinction"):
            distinctions.get_distinction("nonexistent")


class TestToJsonlFormat:
    """Test JSONL format conversion."""

    def test_converts_to_jsonl(self, distinctions: CriticalDistinctions) -> None:
        """Converts all pairs to JSONL format."""
        jsonl = distinctions.to_jsonl_format()

        assert len(jsonl) == 10
        assert all("prompt" in item for item in jsonl)
        assert all("chosen" in item for item in jsonl)
        assert all("rejected" in item for item in jsonl)

    def test_jsonl_excludes_source(self, distinctions: CriticalDistinctions) -> None:
        """JSONL format excludes source field."""
        jsonl = distinctions.to_jsonl_format()

        for item in jsonl:
            assert "source" not in item


class TestCriticalDistinctionDataclass:
    """Test CriticalDistinction dataclass."""

    def test_stores_all_fields(self) -> None:
        """CriticalDistinction stores all fields correctly."""
        pair = PreferencePair(
            prompt="Q?",
            chosen="Good",
            rejected="Bad",
        )
        distinction = CriticalDistinction(
            name="test",
            description="Test distinction",
            pairs=[pair],
        )

        assert distinction.name == "test"
        assert distinction.description == "Test distinction"
        assert len(distinction.pairs) == 1


class TestContentQuality:
    """Test the quality of the distinction content."""

    def test_chosen_is_longer_than_rejected(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen responses are generally longer and more detailed."""
        all_pairs = distinctions.get_all_pairs()

        for pair in all_pairs:
            assert len(pair.chosen) > len(pair.rejected), (
                f"Chosen should be longer than rejected for: {pair.prompt[:50]}..."
            )

    def test_chosen_contains_bobs_voice(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Chosen responses contain Bob's voice markers."""
        all_pairs = distinctions.get_all_pairs()

        voice_markers = ["I've", "I'd", "Here's", "my", "I've seen", "I've tracked"]

        for pair in all_pairs:
            has_voice = any(marker in pair.chosen for marker in voice_markers)
            assert has_voice, f"Chosen should have Bob's voice: {pair.prompt[:50]}..."

    def test_rejected_is_more_generic(
        self, distinctions: CriticalDistinctions
    ) -> None:
        """Rejected responses are more generic and impersonal."""
        all_pairs = distinctions.get_all_pairs()

        for pair in all_pairs:
            # Rejected should have fewer first-person references
            chosen_i_count = pair.chosen.lower().count(" i ")
            rejected_i_count = pair.rejected.lower().count(" i ")

            assert rejected_i_count <= chosen_i_count, (
                f"Rejected should be less personal: {pair.prompt[:50]}..."
            )
