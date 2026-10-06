import pytest
from pydantic import ValidationError

from triage.tiers import FAULT_NAMES, TIER_TABLE, FaultEntry


def test_twelve_dangerous_four_standard() -> None:
    tiers = [e.tier for e in TIER_TABLE.values()]
    assert tiers.count("dangerous") == 12
    assert tiers.count("standard") == 4
    assert len(TIER_TABLE) == 16


def test_only_tap_is_degraded() -> None:
    assert [e.name for e in TIER_TABLE.values() if e.degraded] == ["dripping tap or tap tight to turn"]


def test_key_matches_entry_name_and_has_a_source() -> None:
    for name, entry in TIER_TABLE.items():
        assert name == entry.name
        assert len(entry.sources) >= 1


def test_entry_without_source_rejected() -> None:
    with pytest.raises(ValidationError):
        FaultEntry(name="x", tier="standard", degraded=False, sources=())


def test_fault_names_are_table_keys_in_order() -> None:
    assert FAULT_NAMES == tuple(TIER_TABLE)


@pytest.mark.parametrize("word", ["dangerous", "standard", "tier", "serious", "emergency", "urgent", "severity"])
def test_fault_names_carry_no_judgment_words(word: str) -> None:
    for name in FAULT_NAMES:
        assert word not in name.lower(), name


def test_tier_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        TIER_TABLE["gas leak"] = TIER_TABLE["roof leak"]  # type: ignore[index]


def test_fault_entry_is_frozen() -> None:
    with pytest.raises(ValidationError):
        TIER_TABLE["gas leak"].tier = "standard"  # type: ignore[misc]
