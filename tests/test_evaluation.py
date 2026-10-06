import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.evaluation import lookup_tier
from triage.tiers import TIER_TABLE

TOILET = "blocked or broken toilet"  # dangerous
GAS = "gas leak"  # dangerous
TAP = "dripping tap or tap tight to turn"  # standard
FAN = "fan not working properly"  # standard


def test_empty_has_no_tier_and_no_flag() -> None:
    result = lookup_tier([])
    assert (result.entries, result.tier, result.flag) == ((), None, None)


def test_single_dangerous() -> None:
    result = lookup_tier([GAS])
    assert result.entries == (TIER_TABLE[GAS],)
    assert (result.tier, result.flag) == ("dangerous", None)


def test_single_standard() -> None:
    result = lookup_tier([TAP])
    assert (result.tier, result.flag) == ("standard", None)


def test_mixed_tier_pair_takes_dangerous_and_flags() -> None:
    result = lookup_tier([TAP, TOILET])
    assert result.tier == "dangerous"
    assert result.entries == (TIER_TABLE[TOILET], TIER_TABLE[TAP])  # table order, not input order
    assert result.flag == (
        f'Fault type unclear — could be "{TOILET}" or "{TAP}". Scored at the highest tier among them (dangerous).'
    )


def test_same_tier_pair_still_flagged() -> None:
    result = lookup_tier([TAP, FAN])
    assert result.tier == "standard"
    assert result.flag is not None and TAP in result.flag and FAN in result.flag


def test_duplicate_name_counted_once() -> None:
    result = lookup_tier([GAS, GAS])
    assert result.entries == (TIER_TABLE[GAS],)
    assert result.flag is None


def test_unknown_name_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="serious roof leak"):
        lookup_tier([GAS, "serious roof leak"])


names = st.sampled_from(list(TIER_TABLE))
# Lists, not sets, so duplicates are exercised too.
matches = st.lists(names, max_size=8)


@given(matches)
def test_tier_is_highest_of_the_subset(match: list[str]) -> None:
    tiers = {TIER_TABLE[n].tier for n in match}
    expected = "dangerous" if "dangerous" in tiers else ("standard" if tiers else None)
    assert lookup_tier(match).tier == expected


@given(matches, st.randoms(use_true_random=False))
def test_result_independent_of_input_order(match: list[str], rng: random.Random) -> None:
    shuffled = match[:]
    rng.shuffle(shuffled)
    assert lookup_tier(shuffled) == lookup_tier(match)


@given(matches)
def test_flag_iff_two_or_more_distinct_names(match: list[str]) -> None:
    assert (lookup_tier(match).flag is not None) == (len(set(match)) >= 2)


@given(matches)
def test_every_candidate_named_in_flag(match: list[str]) -> None:
    flag = lookup_tier(match).flag
    if flag is not None:
        assert all(f'"{n}"' in flag for n in match)
