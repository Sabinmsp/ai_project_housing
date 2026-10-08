import pytest
from pydantic import ValidationError

from triage.tiers import COORDINATOR_SOURCE, FAULT_NAMES, NO_FIT_EMERGENCY, NO_FIT_GENERAL, TIER_TABLE, FaultEntry


def test_twelve_dangerous_four_standard() -> None:
    # The two public authorities' entries; the coordinator-only entries are counted separately.
    tiers = [e.tier for e in TIER_TABLE.values() if not e.coordinator_only]
    assert tiers.count("dangerous") == 12
    assert tiers.count("standard") == 4
    assert len(tiers) == 16


def test_only_tap_is_degraded() -> None:
    assert [e.name for e in TIER_TABLE.values() if e.degraded] == ["dripping tap or tap tight to turn"]


def test_key_matches_entry_name_and_has_a_source() -> None:
    for name, entry in TIER_TABLE.items():
        assert name == entry.name
        assert len(entry.sources) >= 1


def test_entry_without_source_rejected() -> None:
    with pytest.raises(ValidationError):
        FaultEntry(name="x", tier="standard", degraded=False, sources=(), coordinator_only=False)


def test_fault_names_are_the_public_table_keys_in_order() -> None:
    assert FAULT_NAMES == tuple(name for name, e in TIER_TABLE.items() if not e.coordinator_only)


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


# The list the model saw before coordinator-only entries existed. A changed list changes the
# prompt and the recording key, so every saved answer would stop replaying.
PUBLIC_NAMES_BEFORE = (
    "blocked or broken toilet", "blocked drain", "sewage leak", "leaking or burst water main or pipe",
    "exposed electrical wires", "gas leak", "roof leak", "flooding or flood damage", "storm, fire or impact damage",
    "no gas, electricity or water supply", "hot water system not working", "stove or oven not working",
    "dripping tap or tap tight to turn", "stove element not working", "fan not working properly", "power point not working",
)


def test_fault_names_are_exactly_the_list_before_coordinator_entries() -> None:
    assert FAULT_NAMES == PUBLIC_NAMES_BEFORE


def test_the_two_coordinator_only_entries() -> None:
    generic = {n: e for n, e in TIER_TABLE.items() if e.coordinator_only}
    assert {n: (e.tier, e.sources, e.degraded) for n, e in generic.items()} == {
        NO_FIT_EMERGENCY: ("dangerous", (COORDINATOR_SOURCE,), False),
        NO_FIT_GENERAL: ("standard", (COORDINATOR_SOURCE,), False),
    }
    assert NO_FIT_EMERGENCY == "no listed fault fits — emergency" and NO_FIT_GENERAL == "no listed fault fits — general"
    assert not set(generic) & set(FAULT_NAMES)
    assert all(not e.coordinator_only for n, e in TIER_TABLE.items() if n in FAULT_NAMES)


def test_coordinator_only_has_no_default() -> None:
    with pytest.raises(ValidationError):
        FaultEntry(name="x", tier="standard", degraded=False, sources=("nt.gov.au",))
