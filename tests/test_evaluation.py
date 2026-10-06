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
    NO_HAZARD,
    UNCLEAR_HAZARD,
    UNVERIFIED_HAZARD,
    TallyResult,
    Unverified,
    evaluate,
    compute_safety,
    compute_tally,
    mismatch_flag,
    unverified_flag,
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


Hazard = tuple[str, str | None, str] | None  # (hazard_status, mechanism_type, quote)


def facts(
    names: list[str],
    alternative: bool = False,
    coping: bool = False,
    sign: bool = False,
    hazard: Hazard = None,
    harm: str | None = None,  # harm quote
    mismatch: tuple[str, str, str] | None = None,  # (direction, claim quote, detail quote)
    fault: str | None = None,
) -> ExtractedFacts:
    spans = [{"field": "taxonomy_match", "text": "quoted"}] if names else []
    spans += [{"field": f, "text": "quoted"} for f, on in
              [("alternative_mentioned", alternative), ("coping_mentioned", coping), ("fault_or_sign", sign)] if on]
    status, mechanism, hazard_quote = hazard or ("none", None, "")
    if status != "none":
        spans.append({"field": "hazard", "text": hazard_quote})
    if harm:
        spans.append({"field": "harm_claimed", "text": harm})
    if fault:
        spans.append({"field": "fault_description", "text": fault})
    if mismatch:
        spans += [{"field": "mismatch_claim", "text": mismatch[1]}, {"field": "mismatch_detail", "text": mismatch[2]}]
    return ExtractedFacts.model_validate({
        "fault_description": fault,
        "taxonomy_match": names,
        "alternative_mentioned": alternative,
        "coping_mentioned": coping,
        "hazard_status": status,
        "mechanism_type": mechanism,
        "harm_claimed": harm is not None,
        "fault_or_sign": "sign" if sign else "fault",
        "claim_mismatch": mismatch[0] if mismatch else None,
        "worsening_mentioned": False,
        "quoted_spans": spans,
    })


def tally(names: list[str], unverified: Unverified = frozenset(), **kw: bool) -> TallyResult:
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
    result = tally([TOILET], frozenset({("alternative_mentioned", "quoted")}), alternative=True)
    assert (result.tally, result.flags) == (4, (UNVERIFIED_ALTERNATIVE,))


def test_unverified_sign_keeps_4_and_flags() -> None:
    result = tally([SEWAGE], frozenset({("fault_or_sign", "quoted")}), sign=True)
    assert (result.tally, result.flags) == (4, (UNVERIFIED_SIGN,))


def test_no_tier_has_no_tally() -> None:
    result = tally([])
    assert (result.tally, result.reasons, result.flags) == (None, (), ())


def test_ambiguous_tap_and_element_takes_max() -> None:
    result = tally([TAP, ELEMENT])
    assert (result.tally, result.reasons) == (3, (NO_ALTERNATIVE,))


def test_unverified_claim_on_degraded_fault_not_flagged() -> None:
    # The tap scores +0 by definition, so "+1 kept" would be false.
    result = tally([TAP], frozenset({("alternative_mentioned", "quoted")}), alternative=True)
    assert (result.tally, result.flags) == (2, ())


# --- compute_tally: properties ----------------------------------------------

# compute_tally and compute_safety only read the field, so a placeholder quote is enough.
tally_pairs = st.sampled_from(SPAN_FIELDS).map(lambda f: (f, "quoted"))
unverified_sets = st.frozensets(tally_pairs)


@given(matches, st.booleans(), st.booleans(), unverified_sets)
def test_coping_never_changes_result(match: list[str], alternative: bool, sign: bool, unverified: Unverified) -> None:
    names = list(dict.fromkeys(match))
    with_coping = tally(names, unverified, alternative=alternative, sign=sign, coping=True)
    assert with_coping == tally(names, unverified, alternative=alternative, sign=sign, coping=False)


@given(matches, st.booleans(), st.booleans(), unverified_sets, tally_pairs)
def test_more_unverified_never_lowers_tally(
    match: list[str], alternative: bool, sign: bool, unverified: Unverified, extra: tuple[str, str]
) -> None:
    names = list(dict.fromkeys(match))
    before = tally(names, unverified, alternative=alternative, sign=sign).tally
    after = tally(names, unverified | {extra}, alternative=alternative, sign=sign).tally
    assert (before is None and after is None) or (after is not None and before is not None and after >= before)


@given(matches, st.booleans(), st.booleans(), st.booleans(), unverified_sets)
def test_tally_in_range_whenever_tier_set(
    match: list[str], alternative: bool, sign: bool, coping: bool, unverified: Unverified
) -> None:
    names = list(dict.fromkeys(match))
    result = tally(names, unverified, alternative=alternative, sign=sign, coping=coping)
    if lookup_tier(names).tier is None:
        assert result.tally is None
    else:
        assert result.tally in {2, 3, 4}


# --- compute_safety: §4.5 table and grounds examples --------------------------

LIGHT = "water dripping through the light fitting"
WIRE = "wire hanging, hasn't touched water"
CEILING = "sagging ceiling, we avoid that spot"
ELECTROCUTE = "this tap's gonna electrocute the kids"


def test_described_active_is_2() -> None:
    result = compute_safety(facts([], hazard=("described", "active", LIGHT)), frozenset())
    assert result.level == 2
    assert result.reason == f"Active hazard described: '{LIGHT}' — full override"
    assert result.flags == ()


def test_described_conditional_is_1() -> None:
    result = compute_safety(facts([], hazard=("described", "conditional", WIRE)), frozenset())
    assert (result.level, result.flags) == (1, ())
    assert f"'{WIRE}'" in result.reason


def test_unclear_is_1_and_flagged_despite_self_mitigation() -> None:
    # "we avoid that spot" is self-mitigation: never a field, so it can't lower the gate.
    result = compute_safety(facts([], hazard=("unclear", None, CEILING)), frozenset())
    assert result.level == 1
    assert result.reason == f"Unclear hazard: '{CEILING}' — treated as conditional"
    assert result.flags == (f"Possible hazard — needs a direct look: '{CEILING}'",)
    assert f"'{CEILING}'" in result.flags[0]


def test_harm_claim_without_pathway_is_0_with_g5_flag() -> None:
    result = compute_safety(facts([TAP], harm=ELECTROCUTE), frozenset())
    assert (result.level, result.reason) == (0, NO_HAZARD)
    assert len(result.flags) == 1 and "Safety claim — unconfirmed" in result.flags[0]
    assert f"'{ELECTROCUTE}'" in result.flags[0]


def test_urgent_tone_without_harm_is_0_no_flag() -> None:
    # "tap's the most urgent thing": tone only (Safety G2), no harm named.
    result = compute_safety(facts([TAP]), frozenset())
    assert (result.level, result.reason, result.flags) == (0, NO_HAZARD, ())


def test_unclear_with_harm_has_only_the_unclear_flag() -> None:
    result = compute_safety(facts([], hazard=("unclear", None, CEILING), harm=ELECTROCUTE), frozenset())
    assert (result.level, result.flags) == (1, (UNCLEAR_HAZARD.format(quote=CEILING),))


def test_described_with_harm_has_no_g5_flag() -> None:
    result = compute_safety(facts([], hazard=("described", "active", LIGHT), harm=ELECTROCUTE), frozenset())
    assert (result.level, result.flags) == (2, ())


def test_unverified_hazard_keeps_level_and_flags() -> None:
    result = compute_safety(facts([], hazard=("described", "active", LIGHT)), frozenset({("hazard", LIGHT)}))
    assert (result.level, result.flags) == (2, (UNVERIFIED_HAZARD,))


def test_unverified_harm_still_fires_g5() -> None:
    result = compute_safety(facts([TAP], harm=ELECTROCUTE), frozenset({("harm_claimed", ELECTROCUTE)}))
    assert result.level == 0 and len(result.flags) == 1


# --- compute_safety: properties ----------------------------------------------

hazards = st.sampled_from([None, ("described", "active", LIGHT), ("described", "conditional", WIRE),
                           ("unclear", None, CEILING)])
harms = st.sampled_from([None, ELECTROCUTE])
safety_pairs = st.sampled_from(["hazard", "harm_claimed", "taxonomy_match"]).map(lambda f: (f, "quoted"))
safety_unverified = st.frozensets(safety_pairs)


@given(matches, hazards, harms, safety_unverified)
def test_safety_independent_of_taxonomy_match(match: list[str], hazard: Hazard, harm: str | None, unverified: Unverified) -> None:
    names = list(dict.fromkeys(match))
    assert compute_safety(facts(names, hazard=hazard, harm=harm), unverified) == compute_safety(
        facts([], hazard=hazard, harm=harm), unverified
    )


@given(hazards, harms, safety_unverified, safety_pairs)
def test_more_unverified_never_lowers_level(
    hazard: Hazard, harm: str | None, unverified: Unverified, extra: tuple[str, str]
) -> None:
    f = facts([], hazard=hazard, harm=harm)
    assert compute_safety(f, unverified | {extra}).level >= compute_safety(f, unverified).level


@given(hazards, harms, safety_unverified)
def test_level_2_iff_described_active(hazard: Hazard, harm: str | None, unverified: Unverified) -> None:
    f = facts([], hazard=hazard, harm=harm)
    is_active = f.hazard_status == "described" and f.mechanism_type == "active"
    assert (compute_safety(f, unverified).level == 2) == is_active


# --- mismatch_flag and unverified_flag ------------------------------------------

OVER = ("over", "it's an emergency, flooding everywhere", "small drip under the sink")
UNDER = ("under", "nothing too bad", "sewage coming up through the shower")


def assert_quotes_in(flag: str | None, *quotes: str) -> None:
    assert flag is not None
    for q in quotes:
        assert f"'{q}'" in flag


def test_over_on_standard_tier_flags() -> None:
    flag = mismatch_flag(facts([TAP], mismatch=OVER), lookup_tier([TAP]))
    assert flag is not None and flag.startswith("Claim stronger than the report's own details")
    assert_quotes_in(flag, OVER[1], OVER[2])


def test_under_on_dangerous_tier_flags() -> None:
    flag = mismatch_flag(facts([SEWAGE], mismatch=UNDER), lookup_tier([SEWAGE]))
    assert flag is not None and flag.startswith("Report plays down a fault on the repair-first list")
    assert_quotes_in(flag, UNDER[1], UNDER[2])


def test_under_on_standard_tier_not_flagged() -> None:
    assert mismatch_flag(facts([TAP], mismatch=UNDER), lookup_tier([TAP])) is None


def test_under_with_no_tier_not_flagged() -> None:
    assert mismatch_flag(facts([], mismatch=UNDER), lookup_tier([])) is None


def test_no_mismatch_not_flagged() -> None:
    assert mismatch_flag(facts([SEWAGE]), lookup_tier([SEWAGE])) is None


def test_one_unverified_span_flagged_with_its_quote() -> None:
    f = facts([], hazard=("described", "active", LIGHT), harm=ELECTROCUTE)
    flag = unverified_flag(f, frozenset({("hazard", LIGHT)}))
    assert flag == f"Quoted words not found in the report: '{LIGHT}' — check the reading"


def test_two_unverified_spans_give_one_flag_listing_both() -> None:
    f = facts([], hazard=("described", "active", LIGHT), harm=ELECTROCUTE)
    flag = unverified_flag(f, frozenset({("harm_claimed", ELECTROCUTE), ("hazard", LIGHT)}))
    assert flag == f"Quoted words not found in the report: '{LIGHT}', '{ELECTROCUTE}' — check the reading"


def test_nothing_unverified_not_flagged() -> None:
    f = facts([], hazard=("described", "active", LIGHT), harm=ELECTROCUTE)
    assert unverified_flag(f, frozenset()) is None



def test_only_the_failed_quote_of_a_field_is_listed() -> None:
    data = facts([], hazard=("described", "active", LIGHT)).model_dump()
    data["quoted_spans"].append({"field": "hazard", "text": "wires sparking"})
    f = ExtractedFacts.model_validate(data)
    flag = unverified_flag(f, frozenset({("hazard", "wires sparking")}))
    assert flag == "Quoted words not found in the report: 'wires sparking' — check the reading"


# --- evaluate: worked examples from the grounds docs ------------------------------


def test_roof_collapse_untiered_but_active() -> None:
    quote = "ceiling's come down on the bed"
    result = evaluate(facts([], hazard=("described", "active", quote), fault="roof collapse"), frozenset())
    assert (result.tier.tier, result.tally.tally, result.safety.level) == (None, None, 2)


def test_g2_founding_pair_sign_3_fault_4() -> None:
    smell = evaluate(facts([SEWAGE], sign=True, fault="sewage smell, 2 days"), frozenset())
    blocked = evaluate(facts([TOILET], fault="toilet fully blocked, only toilet"), frozenset())
    assert (smell.tally.tally, blocked.tally.tally) == (3, 4)


def test_played_down_toilet_keeps_4_and_flags() -> None:
    mismatch = ("under", "nothing too bad", "toilet's blocked")
    result = evaluate(facts([TOILET], mismatch=mismatch, fault="toilet's blocked"), frozenset())
    assert result.tally.tally == 4
    assert result.flags == (mismatch_flag(facts([TOILET], mismatch=mismatch), lookup_tier([TOILET])),)


def test_overclaimed_tap_scores_2_with_flag_and_no_safety() -> None:
    # "can't cope" is distress: not interpreted, only routed to a human via the flag.
    mismatch = ("over", "most urgent thing, can't cope", "tap's dripping")
    result = evaluate(facts([TAP], mismatch=mismatch, fault="tap's dripping"), frozenset())
    assert (result.tally.tally, result.safety.level) == (2, 0)
    assert len(result.flags) == 1 and result.flags[0].startswith("Claim stronger than")


def test_electrocute_claim_is_level_0_with_g5_flag() -> None:
    result = evaluate(facts([TAP], harm=ELECTROCUTE, fault="tap"), frozenset())
    assert result.safety.level == 0
    assert len(result.flags) == 1 and "Safety claim — unconfirmed" in result.flags[0]


def test_ambiguous_drain_or_sewage_is_dangerous_and_flagged() -> None:
    result = evaluate(facts([DRAIN, SEWAGE], fault="could be blocked drain or sewage leak"), frozenset())
    assert result.tier.tier == "dangerous"
    assert result.flags == (lookup_tier([DRAIN, SEWAGE]).flag,)


def test_no_fault_named_raises() -> None:
    with pytest.raises(ValueError, match="no fault named"):
        evaluate(facts([]), frozenset())


def test_unverified_pair_not_in_report_raises() -> None:
    with pytest.raises(ValueError, match="not quoted spans"):
        evaluate(facts([TOILET], fault="toilet"), frozenset({("hazard", "made up")}))


def test_flags_ordered_safety_ambiguity_tally_mismatch_unverified() -> None:
    mismatch = ("over", "flooding everywhere", "small drip")
    f = facts([TOILET, DRAIN], alternative=True, hazard=("unclear", None, CEILING), mismatch=mismatch, fault="toilet")
    result = evaluate(f, frozenset({("alternative_mentioned", "quoted")}))
    safety, ambiguity, tally_flag, mismatch_text, unverified_text = result.flags
    assert safety.startswith("Possible hazard")
    assert ambiguity.startswith("Fault type unclear")
    assert tally_flag == UNVERIFIED_ALTERNATIVE
    assert mismatch_text.startswith("Claim stronger than")
    assert unverified_text.startswith("Quoted words not found")


@given(
    matches, hazards, harms, st.booleans(), st.booleans(),
    st.sampled_from([None, OVER, UNDER]), st.data(),
)
def test_evaluate_flags_never_blank_or_duplicated(
    match: list[str], hazard: Hazard, harm: str | None, alternative: bool, sign: bool,
    mismatch: tuple[str, str, str] | None, data: st.DataObject,
) -> None:
    names = list(dict.fromkeys(match))
    f = facts(names, alternative=alternative, sign=sign, hazard=hazard, harm=harm, mismatch=mismatch, fault="fault")
    pairs = sorted({(s.field, s.text) for s in f.quoted_spans})
    unverified = data.draw(st.frozensets(st.sampled_from(pairs)))
    flags = evaluate(f, unverified).flags
    assert all(flag.strip() for flag in flags)
    assert len(flags) == len(set(flags))
