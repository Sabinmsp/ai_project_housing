import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from triage.evaluation import (
    ALTERNATIVE,
    DEGRADED,
    NO_ALTERNATIVE,
    SIGN,
    UNVERIFIED_ALTERNATIVE,
    UNVERIFIED_SIGN,
    TallyResult,
    compute_tally,
    lookup_tier,
)
from triage.models import ExtractedFacts
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


# --- compute_tally: worked examples (master §3.2.2, §4.3) --------------------

SEWAGE = "sewage leak"  # dangerous
DRAIN = "blocked drain"  # dangerous
ELEMENT = "stove element not working"  # standard
STOVE = "stove or oven not working"  # dangerous
SPAN_FIELDS = ["taxonomy_match", "alternative_mentioned", "coping_mentioned", "fault_or_sign"]


def facts(names: list[str], alternative: bool = False, coping: bool = False, sign: bool = False) -> ExtractedFacts:
    spans = [("taxonomy_match", names)] if names else []
    spans += [(f, on) for f, on in [("alternative_mentioned", alternative), ("coping_mentioned", coping), ("fault_or_sign", sign)]]
    return ExtractedFacts.model_validate({
        "taxonomy_match": names,
        "alternative_mentioned": alternative,
        "coping_mentioned": coping,
        "hazard_status": "none",
        "harm_claimed": False,
        "fault_or_sign": "sign" if sign else "fault",
        "claim_mismatch": None,
        "worsening_mentioned": False,
        "quoted_spans": [{"field": f, "text": "quoted"} for f, on in spans if on],
    })


def tally(names: list[str], unverified: frozenset[str] = frozenset(), **kw: bool) -> TallyResult:
    return compute_tally(lookup_tier(names), facts(names, **kw), unverified)


def test_toilet_blocked_scores_4() -> None:
    result = tally([TOILET])
    assert (result.tally, result.reasons, result.flags) == (4, (NO_ALTERNATIVE,), ())


def test_coping_alone_keeps_4() -> None:
    assert tally([TOILET], coping=True).tally == 4


def test_verified_alternative_scores_3() -> None:
    result = tally([TOILET], alternative=True)
    assert (result.tally, result.reasons) == (3, (ALTERNATIVE,))


@pytest.mark.parametrize("name, expected", [(TAP, 2), (ELEMENT, 3), (STOVE, 4), (DRAIN, 4)])
def test_single_fault_scores(name: str, expected: int) -> None:
    assert tally([name]).tally == expected


def test_tap_reason_is_degraded() -> None:
    assert tally([TAP]).reasons == (DEGRADED,)


def test_verified_sign_scores_3() -> None:
    # "sewage smell": the sign points at a sewage leak; it gets the tier but no +1.
    result = tally([SEWAGE], sign=True)
    assert (result.tally, result.reasons) == (3, (SIGN,))


def test_unverified_alternative_keeps_4_and_flags() -> None:
    result = tally([TOILET], frozenset({"alternative_mentioned"}), alternative=True)
    assert (result.tally, result.flags) == (4, (UNVERIFIED_ALTERNATIVE,))


def test_unverified_sign_keeps_4_and_flags() -> None:
    result = tally([SEWAGE], frozenset({"fault_or_sign"}), sign=True)
    assert (result.tally, result.flags) == (4, (UNVERIFIED_SIGN,))


def test_no_tier_has_no_tally() -> None:
    result = tally([])
    assert (result.tally, result.reasons, result.flags) == (None, (), ())


def test_ambiguous_tap_and_element_takes_max() -> None:
    result = tally([TAP, ELEMENT])
    assert (result.tally, result.reasons) == (3, (NO_ALTERNATIVE,))


def test_unverified_claim_on_degraded_fault_not_flagged() -> None:
    # The tap scores +0 by definition, so "+1 kept" would be false.
    result = tally([TAP], frozenset({"alternative_mentioned"}), alternative=True)
    assert (result.tally, result.flags) == (2, ())


# --- compute_tally: properties ----------------------------------------------

unverified_sets = st.frozensets(st.sampled_from(SPAN_FIELDS))


@given(matches, st.booleans(), st.booleans(), unverified_sets)
def test_coping_never_changes_result(match: list[str], alternative: bool, sign: bool, unverified: frozenset[str]) -> None:
    names = list(dict.fromkeys(match))
    with_coping = tally(names, unverified, alternative=alternative, sign=sign, coping=True)
    assert with_coping == tally(names, unverified, alternative=alternative, sign=sign, coping=False)


@given(matches, st.booleans(), st.booleans(), unverified_sets, st.sampled_from(SPAN_FIELDS))
def test_more_unverified_never_lowers_tally(
    match: list[str], alternative: bool, sign: bool, unverified: frozenset[str], extra: str
) -> None:
    names = list(dict.fromkeys(match))
    before = tally(names, unverified, alternative=alternative, sign=sign).tally
    after = tally(names, unverified | {extra}, alternative=alternative, sign=sign).tally
    assert (before is None and after is None) or (after is not None and before is not None and after >= before)


@given(matches, st.booleans(), st.booleans(), st.booleans(), unverified_sets)
def test_tally_in_range_whenever_tier_set(
    match: list[str], alternative: bool, sign: bool, coping: bool, unverified: frozenset[str]
) -> None:
    names = list(dict.fromkeys(match))
    result = tally(names, unverified, alternative=alternative, sign=sign, coping=coping)
    if lookup_tier(names).tier is None:
        assert result.tally is None
    else:
        assert result.tally in {2, 3, 4}
